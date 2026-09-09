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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config import CFG, PROJECT_ROOT, ensure_dirs
from features import RobustClipScaler, named_feature_sets
from mophong_adapter import orders_from_scores, simulate_orders
from plots import plot_model_diagnostics, plot_threshold_curve, plot_yearly_mophong
from train_rf_mlflow import _candidate_dataset, _proxy_threshold_table, _score_full_frame, build_cached_dataset


FEATURE_SET = "fs03_lags_cycle"
OLD_RUN = "research10_feature_sets_1788533519"
OLD_RUN_DIR = CFG.outputs_dir / "research10_1788533519"


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def old_param_distributions() -> dict:
    """Exact param distribution used before the lightweight patch."""

    return {
        "n_estimators": [220, 320, 420, 560],
        "max_depth": [6, 8, 10, 12, 14, None],
        "min_samples_leaf": [80, 150, 250, 400, 700, 1000],
        "min_samples_split": [200, 500, 800, 1200, 1800, 2600],
        "max_features": ["sqrt", "log2", 0.35, 0.50, 0.70],
        "max_samples": [0.55, 0.70, 0.85],
        "class_weight": ["balanced_subsample", "balanced", None],
    }


def _sample_for_search(x, y, max_samples: int, seed: int):
    if len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    half = max_samples // 2
    pos_n = min(len(pos), half)
    neg_n = min(len(neg), max_samples - pos_n)
    idx = np.concatenate(
        [
            rng.choice(pos, pos_n, replace=False),
            rng.choice(neg, neg_n, replace=False),
        ]
    )
    rng.shuffle(idx)
    return x[idx], y[idx]


def fit_old_setup(x_train, y_train, n_iter: int, max_train_samples: int):
    seed = 1003  # fs03 index=3 in original research_10_feature_sets.py
    xs, ys = _sample_for_search(x_train, y_train, max_train_samples, seed)
    base = RandomForestClassifier(bootstrap=True, n_jobs=-1, random_state=seed)
    search = RandomizedSearchCV(
        base,
        param_distributions=old_param_distributions(),
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=1,
        cv=StratifiedKFold(n_splits=3, shuffle=False),
        verbose=1,
        random_state=seed,
        refit=True,
    )
    search.fit(xs, ys)
    best = search.best_estimator_
    # Original run refit selected params on the complete train dataset.
    best.set_params(n_jobs=-1, random_state=seed)
    best.fit(x_train, y_train)
    return best, search.best_params_, float(search.best_score_), pd.DataFrame(search.cv_results_)


def metrics(y, p) -> dict:
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


def total_curve(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve.groupby("threshold", as_index=False).agg(
        resolved=("resolved", "sum"),
        wins=("wins", "sum"),
        losses=("losses", "sum"),
    )
    out["winrate"] = (out["wins"] * 100 / out["resolved"].replace(0, np.nan)).round(4)
    return out


def compare_with_old(new_curve: pd.DataFrame) -> pd.DataFrame:
    old_path = OLD_RUN_DIR / "fs03_lags_cycle_threshold_test.csv"
    old_curve = pd.read_csv(old_path)
    old_total = total_curve(old_curve).rename(
        columns={"resolved": "old_resolved", "wins": "old_wins", "losses": "old_losses", "winrate": "old_winrate"}
    )
    new_total = total_curve(new_curve).rename(
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
                CFG.outputs_dir / f"repro_1788533519_fs03_deals_t{threshold:.2f}_{year}.csv",
                index=False,
                encoding="utf-8-sig",
            )
    return pd.DataFrame(rows)


def run(n_iter: int, max_train_samples: int, force_dataset: bool, simulate_threshold_values: list[float]) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"reproduce_1788533519_fs03_{run_id}"
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

    model, best_params, cv_auc, cv_results = fit_old_setup(x_train, y_train, n_iter, max_train_samples)
    p_valid = model.predict_proba(x_valid)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    valid_metrics = metrics(y_valid, p_valid)
    test_metrics = metrics(y_test, p_test)
    scores = _score_full_frame(frame, cols, scaler, model)
    valid_curve = _proxy_threshold_table(frame, scores, CFG.split.valid_years)
    test_curve = _proxy_threshold_table(frame, scores, CFG.split.test_years)
    comparison = compare_with_old(test_curve)
    yearly = simulate_thresholds(frame, scores, simulate_threshold_values)

    cv_path = out_dir / "fs03_repro_cv_results.csv"
    valid_curve_path = out_dir / "fs03_repro_threshold_valid.csv"
    test_curve_path = out_dir / "fs03_repro_threshold_test.csv"
    comparison_path = out_dir / "fs03_repro_vs_1788533519_test_curve.csv"
    yearly_path = out_dir / "fs03_repro_mophong_thresholds_yearly.csv"
    summary_path = out_dir / "fs03_repro_summary.json"
    model_path = CFG.model_dir / f"fs03_repro_1788533519_{run_id}_bundle.joblib"
    cv_results.to_csv(cv_path, index=False, encoding="utf-8-sig")
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": cols,
            "best_params": best_params,
            "source_setup": OLD_RUN,
        },
        model_path,
        compress=3,
    )

    chart_paths = []
    chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, "fs03_repro_valid")
    chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / "fs03_repro_threshold_valid.png", "fs03 repro validation threshold"))
    chart_paths.append(plot_threshold_curve(test_curve, chart_dir / "fs03_repro_threshold_test.png", "fs03 repro test threshold"))
    for threshold in simulate_threshold_values:
        subset = yearly[yearly["threshold"] == threshold]
        chart_paths.append(plot_yearly_mophong(subset, chart_dir / f"fs03_repro_mophong_t{threshold:.2f}.png"))

    sim_total = yearly.groupby("threshold", as_index=False).agg(deals=("deals", "sum"), resolved=("resolved", "sum"), wins=("wins", "sum"), losses=("losses", "sum"), no_res=("no_res", "sum"))
    sim_total["winrate"] = (sim_total["wins"] * 100 / sim_total["resolved"].replace(0, np.nan)).round(4)

    summary = {
        "run_id": run_id,
        "source_setup": OLD_RUN,
        "feature_set": FEATURE_SET,
        "n_iter": n_iter,
        "max_train_samples": max_train_samples,
        "refit_full": True,
        "seed": 1003,
        "best_params": best_params,
        "cv_auc": cv_auc,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "simulated_thresholds": simulate_threshold_values,
        "simulation_total": sim_total.to_dict(orient="records"),
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - started, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    mlflow = _mlflow()
    if mlflow:
        with mlflow.start_run(run_name=f"reproduce_1788533519_fs03_{run_id}"):
            mlflow.log_param("source_setup", OLD_RUN)
            mlflow.log_param("feature_set", FEATURE_SET)
            mlflow.log_param("n_iter", n_iter)
            mlflow.log_param("max_train_samples", max_train_samples)
            mlflow.log_param("refit_full", True)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("cv_auc", cv_auc)
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            for _, row in sim_total.iterrows():
                t = str(row["threshold"]).replace(".", "_")
                mlflow.log_metric(f"mophong_t{t}_winrate", float(row["winrate"]))
                mlflow.log_metric(f"mophong_t{t}_deals", float(row["deals"]))
            for p in [cv_path, valid_curve_path, test_curve_path, comparison_path, yearly_path, summary_path, model_path]:
                mlflow.log_artifact(str(p))
            for p in chart_paths:
                mlflow.log_artifact(str(p), artifact_path="charts")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=90000)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--simulate-thresholds", default="0.56,0.57,0.58")
    args = parser.parse_args()
    thresholds = [float(item.strip()) for item in args.simulate_thresholds.split(",") if item.strip()]
    run(args.n_iter, args.max_train_samples, args.force_dataset, thresholds)
