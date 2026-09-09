from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import CFG, PROJECT_ROOT, ensure_dirs
from mophong_adapter import orders_from_scores, simulate_orders
from plots import plot_yearly_mophong
from train_rf_mlflow import _add_train_explainability, _candidate_dataset, build_cached_dataset


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def run(bundle_file: Path, source_run_id: str = "") -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*ChainedAssignmentError.*")
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"complete_bb_family_{run_id}"
    chart_dir = out_dir / "charts"
    explain_dir = out_dir / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    explain_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(bundle_file)
    frame = build_cached_dataset(force=False)
    cols = bundle["feature_columns"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    threshold = float(bundle["selected_threshold"])
    feature_set = bundle.get("feature_set", bundle_file.stem)

    rows = []
    year_arr = frame["year"].to_numpy()
    for year in CFG.split.test_years:
        print({"phase": "mophong_year_start", "year": year}, flush=True)
        mask = year_arr == year
        chunk = frame.loc[mask].copy().reset_index(drop=True)
        chunk_scores = np.full(len(chunk), np.nan, dtype=float)
        needed = (
            (chunk["buy_signal"].to_numpy(np.int8) == 1)
            & (chunk["session_london_ny"].to_numpy(np.int8) == 1)
            & chunk[cols].notna().all(axis=1).to_numpy()
        )
        if np.any(needed):
            x_needed = scaler.transform(chunk.loc[needed, cols].to_numpy(float))
            chunk_scores[needed] = model.predict_proba(x_needed)[:, 1]
        orders = orders_from_scores(chunk, chunk_scores, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"feature_set": feature_set, "year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(out_dir / f"{feature_set}_mophong_deals_{year}.csv", index=False, encoding="utf-8-sig")
        print({"phase": "mophong_year_done", "year": year, **summary}, flush=True)

    yearly = pd.DataFrame(rows)
    yearly_path = out_dir / f"{feature_set}_mophong_yearly.csv"
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    yearly_chart = plot_yearly_mophong(yearly, chart_dir / f"{feature_set}_mophong_yearly.png")

    print({"phase": "explainability_start", "feature_set": feature_set}, flush=True)
    x_train, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
    x_valid, _, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    explain_paths = _add_train_explainability(
        model=model,
        feature_cols=cols,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        out_dir=explain_dir / feature_set,
        pairplot_rows=800,
        shap_rows=500,
        seed=5100,
    )
    print({"phase": "explainability_done", "artifacts": len(explain_paths)}, flush=True)

    meta = {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "bundle_file": str(bundle_file),
        "feature_set": feature_set,
        "threshold": threshold,
        "elapsed_sec": round(time.time() - started, 2),
        "mophong_yearly_path": str(yearly_path),
    }
    meta_path = out_dir / "complete_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name=f"complete_bb_family_artifacts_{feature_set}_{run_id}"):
            mlflow.log_params({k: v for k, v in meta.items() if k not in {"mophong_yearly_path"}})
            for _, row in yearly.iterrows():
                prefix = f"mophong_{int(row['year'])}"
                for key in ["deals", "resolved", "wins", "losses", "no_res", "winrate", "fund", "max_fund", "add_fund", "rut_fund"]:
                    if key in row:
                        mlflow.log_metric(f"{prefix}_{key}", float(row[key]))
            mlflow.log_artifact(str(yearly_path), artifact_path="mophong")
            mlflow.log_artifact(str(yearly_chart), artifact_path="charts")
            mlflow.log_artifact(str(meta_path))
            for path in explain_paths:
                artifact_path = f"explainability/{feature_set}/shap" if "shap" in path.name else f"explainability/{feature_set}"
                mlflow.log_artifact(str(path), artifact_path=artifact_path)

    result = {
        **meta,
        "mophong_yearly": yearly.to_dict(orient="records"),
        "chart": str(yearly_chart),
        "explainability_artifacts": [str(path) for path in explain_paths],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-file", required=True)
    parser.add_argument("--source-run-id", default="")
    args = parser.parse_args()
    run(Path(args.bundle_file), source_run_id=args.source_run_id)
