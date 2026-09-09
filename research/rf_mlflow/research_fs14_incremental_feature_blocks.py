from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from plots import plot_model_diagnostics, plot_threshold_curve
from research_fs14_overlay_norm_light import RobustClipScalerLite


FEATURE_FAMILY = "fs14_incremental_feature_blocks"
CACHE_FAMILY = "fs14_full_lasso_corr"
META_COLS = {"dates", "candle_index", "entry_index", "entry_price", "year", "label"}
CORE_TOKENS = ("tod_", "dow_", "sig_bb_reversion", "buy_signal", "session_london_ny")


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _candidate_dir() -> Path:
    return CFG.cache_dir / CACHE_FAMILY / "candidates"


def _year_path(year: int) -> Path:
    return _candidate_dir() / f"candidates_{year}.parquet"


def _read_feature_cols() -> list[str]:
    return json.loads((_candidate_dir() / "feature_columns.json").read_text(encoding="utf-8"))


def _load_years(years: tuple[int, ...], columns: list[str]) -> pd.DataFrame:
    frames = []
    for year in years:
        frames.append(pd.read_parquet(_year_path(year), columns=columns))
    return pd.concat(frames, ignore_index=True)


def _feature_order(cols: list[str]) -> list[str]:
    core = [c for c in cols if any(tok in c for tok in CORE_TOKENS)]
    lag = [c for c in cols if ("_lag" in c or "close_diff_lag_" in c or "body_lag_" in c) and c not in core]
    cycle_rank = [c for c in cols if ("cycle" in c or "rank" in c) and c not in core and c not in lag]
    width_roll = [c for c in cols if ("width" in c or "rollmean" in c or "rollstd" in c) and c not in core and c not in lag and c not in cycle_rank]
    rest = [c for c in cols if c not in set(core + lag + cycle_rank + width_roll)]
    return list(dict.fromkeys(core + lag + cycle_rank + width_roll + rest))


def _chunks(cols: list[str], size: int) -> list[list[str]]:
    return [cols[i : i + size] for i in range(0, len(cols), size)]


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    out = {
        "samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
    }
    if len(np.unique(y_true)) == 2:
        out["logloss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    return out


def _threshold_table(frame: pd.DataFrame, prob: np.ndarray, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    year_arr = frame["year"].to_numpy()
    label_arr = frame["label"].to_numpy(np.int8)
    session_arr = frame["session_london_ny"].to_numpy(np.int8) if "session_london_ny" in frame.columns else np.ones(len(frame), dtype=np.int8)
    for threshold in thresholds:
        selected = (prob >= threshold) & (session_arr == 1)
        for year in sorted(frame["year"].unique()):
            ym = year_arr == year
            mask = selected & ym
            wins = int(np.count_nonzero(mask & (label_arr == 1)))
            losses = int(np.count_nonzero(mask & (label_arr == 0)))
            resolved = wins + losses
            rows.append(
                {
                    "threshold": float(threshold),
                    "year": int(year),
                    "resolved": resolved,
                    "wins": wins,
                    "losses": losses,
                    "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _select_threshold(valid_curve: pd.DataFrame) -> float:
    grouped = valid_curve.groupby("threshold", as_index=False).agg(resolved=("resolved", "sum"), wins=("wins", "sum"))
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    viable = grouped[grouped["resolved"] >= 2200].copy()
    if viable.empty:
        viable = grouped.copy()
    viable["score"] = viable["winrate"] - (viable["resolved"] - 3500).abs() / 3500
    return float(viable.sort_values(["score", "winrate"], ascending=[False, False]).iloc[0]["threshold"])


def _fit_block(cols: list[str], n_estimators: int, min_samples_leaf: int) -> tuple[RandomForestClassifier, RobustClipScalerLite, pd.DataFrame, np.ndarray, np.ndarray]:
    read_cols = list(dict.fromkeys(cols + ["label"]))
    train = _load_years(CFG.split.train_years, read_cols)
    x_train = train[cols].to_numpy("float32", copy=True)
    y_train = train["label"].to_numpy("int8", copy=True)
    del train
    gc.collect()

    scaler = RobustClipScalerLite().fit(x_train)
    x_train = scaler.transform_inplace(x_train)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=max(800, min_samples_leaf * 4),
        max_features="sqrt",
        max_samples=0.70,
        class_weight=None,
        bootstrap=True,
        random_state=31001,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    importance = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    del x_train
    gc.collect()
    return model, scaler, importance, y_train, np.array(cols, dtype=object)


def _predict_years(model, scaler, cols: list[str], years: tuple[int, ...], include_meta: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    meta = ["label", "year", "session_london_ny"]
    if include_meta:
        meta = ["dates", "candle_index", "entry_index", "entry_price", "label", "year", "session_london_ny", "buy_signal"]
    read_cols = list(dict.fromkeys(cols + meta))
    frame = _load_years(years, read_cols)
    x = frame[cols].to_numpy("float32", copy=True)
    x = scaler.transform_inplace(x)
    prob = model.predict_proba(x)[:, 1]
    del x
    gc.collect()
    return frame, prob


def _plot_block_history(history: pd.DataFrame, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["block"], history["valid_auc"], marker="o", label="valid_auc")
    ax1.plot(history["block"], history["test_auc"], marker="o", label="test_auc")
    ax1.set_xlabel("Feature block")
    ax1.set_ylabel("AUC")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(history["block"], history["feature_count"], alpha=0.20, color="gray", label="features")
    ax2.set_ylabel("features used")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Incremental 50-feature block selection")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_importance(importance: pd.DataFrame, path: Path) -> Path:
    top = importance.head(25).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Final RF feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run(block_size: int, carry_top_n: int, block_estimators: int, final_estimators: int, min_samples_leaf: int) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    all_features = _feature_order(_read_feature_cols())
    chunks = _chunks(all_features, block_size)
    carry: list[str] = []
    history_rows = []
    block_importance_paths = []

    for i, chunk in enumerate(chunks, start=1):
        active_cols = list(dict.fromkeys(carry + chunk))
        print({"phase": "block_train", "block": i, "blocks": len(chunks), "active_features": len(active_cols), "new_features": len(chunk)}, flush=True)
        model, scaler, importance, _, _ = _fit_block(active_cols, block_estimators, min_samples_leaf)
        valid_frame, p_valid = _predict_years(model, scaler, active_cols, CFG.split.valid_years)
        test_frame, p_test = _predict_years(model, scaler, active_cols, CFG.split.test_years)
        valid_metrics = _metrics(valid_frame["label"].to_numpy("int8"), p_valid)
        test_metrics = _metrics(test_frame["label"].to_numpy("int8"), p_test)
        imp_path = out_dir / f"block_{i:02d}_importance.csv"
        importance.to_csv(imp_path, index=False, encoding="utf-8-sig")
        block_importance_paths.append(imp_path)
        carry = importance.head(carry_top_n)["feature"].tolist()
        history_rows.append(
            {
                "block": i,
                "feature_count": len(active_cols),
                "carry_count": len(carry),
                "valid_auc": valid_metrics["auc"],
                "valid_brier": valid_metrics["brier"],
                "test_auc": test_metrics["auc"],
                "test_brier": test_metrics["brier"],
                "top_features": "|".join(carry),
            }
        )
        del model, scaler, valid_frame, test_frame, p_valid, p_test, importance
        gc.collect()

    final_cols = carry
    print({"phase": "final_train", "features": len(final_cols)}, flush=True)
    final_model, final_scaler, final_importance, _, _ = _fit_block(final_cols, final_estimators, min_samples_leaf)
    valid_frame, p_valid = _predict_years(final_model, final_scaler, final_cols, CFG.split.valid_years)
    test_frame, p_test = _predict_years(final_model, final_scaler, final_cols, CFG.split.test_years, include_meta=True)
    valid_metrics = _metrics(valid_frame["label"].to_numpy("int8"), p_valid)
    test_metrics = _metrics(test_frame["label"].to_numpy("int8"), p_test)
    valid_curve = _threshold_table(valid_frame, p_valid, CFG.threshold_grid)
    test_curve = _threshold_table(test_frame, p_test, CFG.threshold_grid)
    threshold = _select_threshold(valid_curve)

    history = pd.DataFrame(history_rows)
    history_path = out_dir / "block_history.csv"
    final_features_path = out_dir / "final_selected_features.json"
    final_importance_path = out_dir / "final_rf_feature_importance.csv"
    valid_curve_path = out_dir / "threshold_valid.csv"
    test_curve_path = out_dir / "threshold_test.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    final_features_path.write_text(json.dumps(final_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    final_importance.to_csv(final_importance_path, index=False, encoding="utf-8-sig")
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

    chart_paths = []
    chart_paths.append(_plot_block_history(history, chart_dir / "block_history_auc.png"))
    chart_paths.append(_plot_importance(final_importance, chart_dir / "final_feature_importance_top25.png"))
    chart_paths += plot_model_diagnostics(valid_frame["label"].to_numpy("int8"), p_valid, chart_dir, "final_valid")
    chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / "threshold_valid.png", "Final validation threshold curve"))
    chart_paths.append(plot_threshold_curve(test_curve, chart_dir / "threshold_test.png", "Final test threshold curve"))

    model_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_bundle.joblib"
    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
            "feature_columns": final_cols,
            "feature_family": CACHE_FAMILY,
            "feature_set": FEATURE_FAMILY,
            "selected_threshold": threshold,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        model_path,
        compress=3,
    )

    summary = {
        "run_id": run_id,
        "feature_family": FEATURE_FAMILY,
        "source_cache_family": CACHE_FAMILY,
        "all_features": len(all_features),
        "block_size": block_size,
        "carry_top_n": carry_top_n,
        "blocks": len(chunks),
        "final_features": len(final_cols),
        "selected_threshold": threshold,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "elapsed_sec": round(time.time() - started, 2),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}"):
            mlflow.log_params(
                {
                    "feature_family": FEATURE_FAMILY,
                    "source_cache_family": CACHE_FAMILY,
                    "all_features": len(all_features),
                    "block_size": block_size,
                    "carry_top_n": carry_top_n,
                    "blocks": len(chunks),
                    "final_features": len(final_cols),
                    "block_estimators": block_estimators,
                    "final_estimators": final_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "selected_threshold": threshold,
                    "n_jobs": 1,
                }
            )
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            for _, row in history.iterrows():
                mlflow.log_metric("block_valid_auc", float(row["valid_auc"]), step=int(row["block"]))
                mlflow.log_metric("block_test_auc", float(row["test_auc"]), step=int(row["block"]))
            for path in [history_path, final_features_path, final_importance_path, valid_curve_path, test_curve_path, summary_path, model_path, *block_importance_paths]:
                mlflow.log_artifact(str(path))
            for path in chart_paths:
                mlflow.log_artifact(str(path), artifact_path="charts")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--carry-top-n", type=int, default=20)
    parser.add_argument("--block-estimators", type=int, default=70)
    parser.add_argument("--final-estimators", type=int, default=160)
    parser.add_argument("--min-samples-leaf", type=int, default=300)
    args = parser.parse_args()
    run(
        block_size=args.block_size,
        carry_top_n=args.carry_top_n,
        block_estimators=args.block_estimators,
        final_estimators=args.final_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )
