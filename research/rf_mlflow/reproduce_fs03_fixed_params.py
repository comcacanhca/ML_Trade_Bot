from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from features import RobustClipScaler, named_feature_sets
from mophong_adapter import orders_from_scores, simulate_orders
from plots import plot_model_diagnostics, plot_threshold_curve, plot_yearly_mophong
from train_rf_mlflow import _candidate_dataset, _proxy_threshold_table, _score_full_frame, build_cached_dataset


FEATURE_SET = "fs03_lags_cycle"
SOURCE_RUN_DIR = CFG.outputs_dir / "research10_1788533519"
FIXED_PARAMS = {
    "n_estimators": 220,
    "min_samples_split": 1200,
    "min_samples_leaf": 250,
    "max_samples": 0.85,
    "max_features": 0.70,
    "max_depth": None,
    "class_weight": None,
}


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    out = {
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
        out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def _total_curve(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve.groupby("threshold", as_index=False).agg(
        resolved=("resolved", "sum"),
        wins=("wins", "sum"),
        losses=("losses", "sum"),
    )
    out["winrate"] = (out["wins"] * 100 / out["resolved"].replace(0, np.nan)).round(4)
    return out


def compare_old_new(new_curve: pd.DataFrame) -> pd.DataFrame:
    old_curve = pd.read_csv(SOURCE_RUN_DIR / "fs03_lags_cycle_threshold_test.csv")
    old_total = _total_curve(old_curve).rename(
        columns={"resolved": "old_resolved", "wins": "old_wins", "losses": "old_losses", "winrate": "old_winrate"}
    )
    new_total = _total_curve(new_curve).rename(
        columns={"resolved": "new_resolved", "wins": "new_wins", "losses": "new_losses", "winrate": "new_winrate"}
    )
    cmp = old_total.merge(new_total, on="threshold", how="outer").sort_values("threshold")
    cmp["resolved_diff"] = cmp["new_resolved"] - cmp["old_resolved"]
    cmp["winrate_diff"] = cmp["new_winrate"] - cmp["old_winrate"]
    return cmp


def simulate_thresholds(frame: pd.DataFrame, scores: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    year_arr = frame["year"].to_numpy()
    for threshold in thresholds:
        for year in CFG.split.test_years:
            mask = year_arr == year
            chunk = frame.loc[mask].copy().reset_index(drop=True)
            score_chunk = scores[mask]
            orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
            summary, results = simulate_orders(chunk, orders, tinh_tien=True)
            rows.append({"threshold": threshold, "year": year, **summary})
            pd.DataFrame(results).to_csv(
                CFG.outputs_dir / f"fs03_fixed_params_deals_t{threshold:.2f}_{year}.csv",
                index=False,
                encoding="utf-8-sig",
            )
    return pd.DataFrame(rows)


def run(simulate_threshold_values: list[float], force_dataset: bool, skip_simulate: bool) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"reproduce_fs03_fixed_params_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    frame = build_cached_dataset(force=force_dataset).sort_values("dates").reset_index(drop=True)
    cols = named_feature_sets(frame)[FEATURE_SET]
    x_train_raw, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
    x_valid_raw, y_valid, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
    x_test_raw, y_test, _ = _candidate_dataset(frame, cols, CFG.split.test_years)
    scaler = RobustClipScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)
    x_test = scaler.transform(x_test_raw)

    model = RandomForestClassifier(
        **FIXED_PARAMS,
        bootstrap=True,
        n_jobs=-1,
        random_state=1003,
    )
    model.fit(x_train, y_train)
    p_valid = model.predict_proba(x_valid)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    valid_metrics = _metrics(y_valid, p_valid)
    test_metrics = _metrics(y_test, p_test)
    scores = _score_full_frame(frame, cols, scaler, model)
    valid_curve = _proxy_threshold_table(frame, scores, CFG.split.valid_years)
    test_curve = _proxy_threshold_table(frame, scores, CFG.split.test_years)
    valid_curve_path = out_dir / "fs03_fixed_threshold_valid.csv"
    test_curve_path = out_dir / "fs03_fixed_threshold_test.csv"
    comparison_path = out_dir / "fs03_fixed_vs_research10_1788533519_test_curve.csv"
    yearly_path = out_dir / "fs03_fixed_mophong_thresholds_yearly.csv"
    sim_total_path = out_dir / "fs03_fixed_mophong_thresholds_total.csv"
    summary_path = out_dir / "fs03_fixed_summary.json"
    model_path = CFG.model_dir / f"fs03_fixed_params_{run_id}_bundle.joblib"
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
    comparison = compare_old_new(test_curve)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": cols, "params": FIXED_PARAMS}, model_path, compress=3)

    chart_paths = []
    chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, "fs03_fixed_valid")
    chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / "fs03_fixed_threshold_valid.png", "fs03 fixed validation threshold"))
    chart_paths.append(plot_threshold_curve(test_curve, chart_dir / "fs03_fixed_threshold_test.png", "fs03 fixed test threshold"))
    if skip_simulate:
        yearly = pd.DataFrame()
        sim_total = pd.DataFrame()
    else:
        yearly = simulate_thresholds(frame, scores, simulate_threshold_values)
        sim_total = yearly.groupby("threshold", as_index=False).agg(
            deals=("deals", "sum"),
            resolved=("resolved", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            no_res=("no_res", "sum"),
        )
        sim_total["winrate"] = (sim_total["wins"] * 100 / sim_total["resolved"].replace(0, np.nan)).round(4)
        yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        sim_total.to_csv(sim_total_path, index=False, encoding="utf-8-sig")
        for threshold in simulate_threshold_values:
            chart_paths.append(plot_yearly_mophong(yearly[yearly["threshold"] == threshold], chart_dir / f"fs03_fixed_mophong_t{threshold:.2f}.png"))

    summary = {
        "run_id": run_id,
        "source_run": "research10_1788533519",
        "feature_set": FEATURE_SET,
        "params": FIXED_PARAMS,
        "random_state": 1003,
        "skip_simulate": skip_simulate,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "simulated_thresholds": simulate_threshold_values,
        "simulation_total": sim_total.to_dict(orient="records") if not sim_total.empty else [],
        "curve_comparison_key_thresholds": comparison[comparison["threshold"].isin(simulate_threshold_values)].to_dict(orient="records"),
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - started, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    mlflow = _mlflow()
    if mlflow:
        with mlflow.start_run(run_name=f"reproduce_fs03_fixed_params_{run_id}"):
            mlflow.log_param("source_run", "research10_1788533519")
            mlflow.log_param("feature_set", FEATURE_SET)
            mlflow.log_param("random_state", 1003)
            mlflow.log_params({f"fixed_{k}": v for k, v in FIXED_PARAMS.items()})
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            if not sim_total.empty:
                for _, row in sim_total.iterrows():
                    t = str(row["threshold"]).replace(".", "_")
                    mlflow.log_metric(f"mophong_t{t}_winrate", float(row["winrate"]))
                    mlflow.log_metric(f"mophong_t{t}_deals", float(row["deals"]))
            for p in [valid_curve_path, test_curve_path, comparison_path, summary_path, model_path]:
                mlflow.log_artifact(str(p))
            if not skip_simulate:
                for p in [yearly_path, sim_total_path]:
                    mlflow.log_artifact(str(p))
            for p in chart_paths:
                mlflow.log_artifact(str(p), artifact_path="charts")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-thresholds", default="0.56,0.57,0.58")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--skip-simulate", action="store_true")
    args = parser.parse_args()
    thresholds = [float(item.strip()) for item in args.simulate_thresholds.split(",") if item.strip()]
    run(thresholds, args.force_dataset, args.skip_simulate)
