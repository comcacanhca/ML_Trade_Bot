from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fs03_common import (
    dump_json,
    load_fs03_splits,
    metrics_from_scores,
    mlflow_client,
    new_run_dir,
    parse_thresholds,
    threshold_curves,
)

from research.mophong_adapter import orders_from_scores, simulate_orders


def simulate_thresholds(frame: pd.DataFrame, scores, thresholds: list[float], out_dir: Path) -> pd.DataFrame:
    from fs03_common import CFG

    rows = []
    year_arr = frame["year"].to_numpy()
    deals_dir = out_dir / "deals"
    deals_dir.mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        for year in CFG.split.test_years:
            mask = year_arr == year
            chunk = frame.loc[mask].copy().reset_index(drop=True)
            score_chunk = scores[mask]
            orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
            summary, results = simulate_orders(chunk, orders, tinh_tien=True)
            rows.append({"threshold": threshold, "year": year, **summary})
            pd.DataFrame(results).to_csv(deals_dir / f"deals_t{threshold:.2f}_{year}.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def run(
    model_bundle: Path,
    thresholds: list[float],
    force_dataset: bool,
    skip_simulate: bool,
    enable_mlflow: bool,
    output_root: Path | None = None,
) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    started = time.time()
    run_dir = new_run_dir("fs03_optuna_eval", output_root)
    bundle = joblib.load(model_bundle)
    data = load_fs03_splits(force_dataset=force_dataset)
    model = bundle["model"]
    scaler = bundle["scaler"]
    cols = bundle["feature_columns"]

    p_valid = model.predict_proba(data["x_valid"])[:, 1]
    p_test = model.predict_proba(data["x_test"])[:, 1]
    valid_metrics = metrics_from_scores(data["y_valid"], p_valid)
    test_metrics = metrics_from_scores(data["y_test"], p_test)
    scores, valid_curve, test_curve = threshold_curves(data["frame"], cols, scaler, model)

    valid_curve_path = run_dir / "threshold_valid.csv"
    test_curve_path = run_dir / "threshold_test.csv"
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

    yearly_path = run_dir / "mophong_thresholds_yearly.csv"
    total_path = run_dir / "mophong_thresholds_total.csv"
    simulation_total = []
    if not skip_simulate:
        yearly = simulate_thresholds(data["frame"], scores, thresholds, run_dir)
        total = yearly.groupby("threshold", as_index=False).agg(
            deals=("deals", "sum"),
            resolved=("resolved", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            no_res=("no_res", "sum"),
        )
        resolved = total["resolved"].astype(float).replace(0.0, np.nan)
        total["winrate"] = (total["wins"].astype(float) * 100.0 / resolved).round(4).fillna(0.0)

        yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        total.to_csv(total_path, index=False, encoding="utf-8-sig")
        simulation_total = total.to_dict(orient="records")

    summary = {
        "run_dir": str(run_dir),
        "model_bundle": str(model_bundle),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "thresholds": thresholds,
        "skip_simulate": skip_simulate,
        "simulation_total": simulation_total,
        "elapsed_sec": round(time.time() - started, 2),
    }
    summary_path = run_dir / "summary.json"
    dump_json(summary_path, summary)

    mlflow = mlflow_client(enable_mlflow)
    if mlflow:
        with mlflow.start_run(run_name=f"fs03_optuna_eval_{int(started)}"):
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            for artifact in (summary_path, valid_curve_path, test_curve_path):
                mlflow.log_artifact(str(artifact))
            if not skip_simulate:
                mlflow.log_artifact(str(yearly_path))
                mlflow.log_artifact(str(total_path))

    print(summary_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate/backtest an fs03 Optuna model bundle.")
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.56,0.57,0.58")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--skip-simulate", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run(args.model_bundle, parse_thresholds(args.thresholds), args.force_dataset, args.skip_simulate, args.enable_mlflow, args.output_root)
