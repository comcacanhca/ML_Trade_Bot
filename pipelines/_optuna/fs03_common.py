from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
RF_MLFLOW_ROOT = RESEARCH_ROOT / "rf_mlflow"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_STORE_PATH = PROCESSED_DIR / "fs03_lags_cycle_feature_store.parquet"
FEATURE_STORE_META_PATH = PROCESSED_DIR / "fs03_lags_cycle_feature_store.meta.json"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"

for path in (RESEARCH_ROOT, RF_MLFLOW_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from research.config import CFG, ensure_dirs  # noqa: E402
from research.features import RobustClipScaler, named_feature_sets  # noqa: E402
from research.rf_mlflow.train_rf_mlflow import _candidate_dataset, _proxy_threshold_table, _score_full_frame, build_cached_dataset  # noqa: E402

FEATURE_SET = "fs03_lags_cycle"
DEFAULT_THRESHOLDS = (0.56, 0.57, 0.58)


def new_run_dir(prefix: str, output_root: Path | None = None) -> Path:
    ensure_dirs()
    root = output_root or (CFG.outputs_dir / "pipelines_optuna")
    run_dir = root / f"{prefix}_{int(time.time())}"
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    return run_dir


def materialize_feature_store(force: bool = False) -> pd.DataFrame:
    """Build or load the FS03 feature store as a DVC-versioned Parquet artifact."""
    ensure_dirs()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_STORE_PATH.exists() and not force:
        return pd.read_parquet(FEATURE_STORE_PATH)

    frame = build_cached_dataset(force=force).sort_values("dates").reset_index(drop=True)
    cols = named_feature_sets(frame)[FEATURE_SET]
    feature_store = frame.copy()
    feature_store.to_parquet(FEATURE_STORE_PATH, index=False)
    dump_json(
        FEATURE_STORE_META_PATH,
        {
            "feature_set": FEATURE_SET,
            "path": str(FEATURE_STORE_PATH),
            "rows": int(len(feature_store)),
            "columns": int(len(feature_store.columns)),
            "feature_columns": cols,
            "train_years": list(CFG.split.train_years),
            "valid_years": list(CFG.split.valid_years),
            "test_years": list(CFG.split.test_years),
            "format": "parquet",
        },
    )
    return feature_store


def parse_thresholds(raw: str | None) -> list[float]:
    if not raw:
        return list(DEFAULT_THRESHOLDS)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def metrics_from_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = (p >= 0.5).astype(int)
    out: dict[str, float | int] = {
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
        out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def load_fs03_splits(force_dataset: bool = False, max_train_rows: int | None = None):
    frame = materialize_feature_store(force=force_dataset).sort_values("dates").reset_index(drop=True)
    cols = named_feature_sets(frame)[FEATURE_SET]
    x_train_raw, y_train, train_meta = _candidate_dataset(frame, cols, CFG.split.train_years)
    x_valid_raw, y_valid, valid_meta = _candidate_dataset(frame, cols, CFG.split.valid_years)
    x_test_raw, y_test, test_meta = _candidate_dataset(frame, cols, CFG.split.test_years)

    if max_train_rows and len(y_train) > max_train_rows:
        rng = np.random.default_rng(1003)
        idx = np.sort(rng.choice(len(y_train), size=max_train_rows, replace=False))
        x_train_raw = x_train_raw[idx]
        y_train = y_train[idx]
        train_meta = train_meta.iloc[idx].reset_index(drop=True)

    scaler = RobustClipScaler().fit(x_train_raw)
    return {
        "frame": frame,
        "feature_columns": cols,
        "scaler": scaler,
        "x_train": scaler.transform(x_train_raw),
        "y_train": y_train,
        "train_meta": train_meta,
        "x_valid": scaler.transform(x_valid_raw),
        "y_valid": y_valid,
        "valid_meta": valid_meta,
        "x_test": scaler.transform(x_test_raw),
        "y_test": y_test,
        "test_meta": test_meta,
    }


def threshold_curves(frame: pd.DataFrame, cols: list[str], scaler: Any, model: Any) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    scores = _score_full_frame(frame, cols, scaler, model)
    valid_curve = _proxy_threshold_table(frame, scores, CFG.split.valid_years)
    test_curve = _proxy_threshold_table(frame, scores, CFG.split.test_years)
    return scores, valid_curve, test_curve


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def mlflow_client(enabled: bool):
    if not enabled:
        return None
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
        artifact_root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", str(MLFLOW_ARTIFACT_ROOT)))
        if not artifact_root.is_absolute():
            artifact_root = PROJECT_ROOT / artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_root.resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(CFG.experiment_name)
        if experiment is None:
            client.create_experiment(CFG.experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None
