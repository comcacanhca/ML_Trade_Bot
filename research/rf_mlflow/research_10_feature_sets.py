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
from plots import (
    plot_feature_set_summary,
    plot_model_diagnostics,
    plot_threshold_curve,
    plot_yearly_mophong,
)
from train_rf_mlflow import _candidate_dataset, _proxy_threshold_table, _score_full_frame, build_cached_dataset


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
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


def _param_distributions() -> dict:
    return {
        "n_estimators": [80, 120, 180, 260, 360],
        "max_depth": [5, 6, 8, 10, 12, None],
        "min_samples_leaf": [80, 150, 250, 400, 700],
        "min_samples_split": [200, 500, 800, 1200, 1800],
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
    idx = np.concatenate([
        rng.choice(pos, pos_n, replace=False),
        rng.choice(neg, neg_n, replace=False),
    ])
    rng.shuffle(idx)
    return x[idx], y[idx]


def _fit_search_model(name: str, x_train, y_train, n_iter: int, max_train_samples: int, seed: int, refit_full: bool):
    xs, ys = _sample_for_search(x_train, y_train, max_train_samples, seed)
    base = RandomForestClassifier(
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    search = RandomizedSearchCV(
        base,
        param_distributions=_param_distributions(),
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=1,
        cv=cv,
        verbose=1,
        random_state=seed,
        refit=True,
    )
    search.fit(xs, ys)
    best = search.best_estimator_
    if refit_full:
        best.set_params(n_jobs=-1, random_state=seed)
        best.fit(x_train, y_train)
    return best, search.best_params_, float(search.best_score_), pd.DataFrame(search.cv_results_)


def _select_threshold_by_constraints(curve: pd.DataFrame) -> float:
    grouped = curve.groupby("threshold", as_index=False).agg(
        resolved=("resolved", "sum"),
        wins=("wins", "sum"),
        min_year_deals=("resolved", "min"),
    )
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    grouped["deal_penalty"] = (grouped["resolved"] - 3000).abs() / 3000
    grouped["score"] = grouped["winrate"] - 3.0 * grouped["deal_penalty"]
    viable = grouped[grouped["resolved"] >= 2200].copy()
    if viable.empty:
        viable = grouped.copy()
    viable = viable.sort_values(["score", "winrate", "resolved"], ascending=[False, False, False])
    return float(viable.iloc[0]["threshold"])


def _total_proxy_stats(curve: pd.DataFrame, threshold: float) -> dict:
    rows = curve[curve["threshold"] == threshold]
    wins = int(rows["wins"].sum())
    resolved = int(rows["resolved"].sum())
    losses = int(rows["losses"].sum())
    return {
        "resolved": resolved,
        "wins": wins,
        "losses": losses,
        "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
        "min_year_deals": int(rows["resolved"].min()) if len(rows) else 0,
    }


def _objective(test_proxy: dict, valid_metrics: dict) -> float:
    wr = test_proxy["winrate"]
    resolved = test_proxy["resolved"]
    deal_score = max(0.0, 1.0 - abs(resolved - 8500) / 8500)
    min_deal_score = min(1.0, test_proxy["min_year_deals"] / 2500)
    return float(wr + 6.0 * deal_score + 3.0 * min_deal_score + 2.0 * valid_metrics.get("auc", 0.5))


def _simulate_best(frame: pd.DataFrame, scores: np.ndarray, threshold: float, feature_set: str) -> pd.DataFrame:
    rows = []
    years = CFG.split.test_years
    year_arr = frame["year"].to_numpy()
    for year in years:
        mask = year_arr == year
        chunk = frame.loc[mask].copy().reset_index(drop=True)
        score_chunk = scores[mask]
        orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"feature_set": feature_set, "year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(
            CFG.outputs_dir / f"research10_mophong_deals_{feature_set}_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return pd.DataFrame(rows)


def run(force_dataset: bool, n_iter: int, max_train_samples: int, simulate_top: int, refit_full: bool) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"research10_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    frame = build_cached_dataset(force=force_dataset)
    feature_sets = named_feature_sets(frame)
    mlflow = _mlflow()
    summary_rows = []
    bundles = {}
    chart_paths: list[Path] = []

    if mlflow is not None:
        parent_run = mlflow.start_run(run_name=f"research10_feature_sets_{run_id}")
    else:
        parent_run = None

    try:
        for idx, (name, cols) in enumerate(feature_sets.items(), start=1):
            seed = 1000 + idx
            print({"phase": "feature_set", "idx": idx, "name": name, "n_features": len(cols)}, flush=True)
            x_train_raw, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
            x_valid_raw, y_valid, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
            x_test_raw, y_test, _ = _candidate_dataset(frame, cols, CFG.split.test_years)

            scaler = RobustClipScaler().fit(x_train_raw)
            x_train = scaler.transform(x_train_raw)
            x_valid = scaler.transform(x_valid_raw)
            x_test = scaler.transform(x_test_raw)
            model, best_params, cv_auc, cv_results = _fit_search_model(
                name, x_train, y_train, n_iter=n_iter, max_train_samples=max_train_samples, seed=seed, refit_full=refit_full
            )

            p_valid = model.predict_proba(x_valid)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            valid_metrics = _metrics(y_valid, p_valid)
            test_metrics = _metrics(y_test, p_test)
            scores = _score_full_frame(frame, cols, scaler, model)
            valid_curve = _proxy_threshold_table(frame, scores, CFG.split.valid_years)
            test_curve = _proxy_threshold_table(frame, scores, CFG.split.test_years)
            threshold = _select_threshold_by_constraints(valid_curve)
            test_proxy = _total_proxy_stats(test_curve, threshold)
            objective_score = _objective(test_proxy, valid_metrics)

            cv_path = out_dir / f"{name}_cv_results.csv"
            valid_curve_path = out_dir / f"{name}_threshold_valid.csv"
            test_curve_path = out_dir / f"{name}_threshold_test.csv"
            cv_results.to_csv(cv_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

            chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, f"{name}_valid")
            chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / f"{name}_threshold_valid.png", f"{name} validation threshold curve"))
            chart_paths.append(plot_threshold_curve(test_curve, chart_dir / f"{name}_threshold_test.png", f"{name} test proxy threshold curve"))

            summary_rows.append(
                {
                    "feature_set": name,
                    "n_features": len(cols),
                    "cv_auc": cv_auc,
                    "valid_auc": valid_metrics.get("auc", 0.0),
                    "valid_accuracy": valid_metrics["accuracy"],
                    "valid_brier": valid_metrics.get("brier", 0.0),
                    "test_auc": test_metrics.get("auc", 0.0),
                    "selected_threshold": threshold,
                    "test_total_deals": test_proxy["resolved"],
                    "test_total_wr": test_proxy["winrate"],
                    "test_min_year_deals": test_proxy["min_year_deals"],
                    "objective_score": objective_score,
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                }
            )
            bundles[name] = {
                "model": model,
                "scaler": scaler,
                "feature_columns": cols,
                "selected_threshold": threshold,
                "best_params": best_params,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
                "test_proxy": test_proxy,
                "scores": scores,
            }

            if mlflow is not None:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_param("feature_set", name)
                    mlflow.log_param("n_features", len(cols))
                    mlflow.log_param("selected_threshold", threshold)
                    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                    mlflow.log_metric("cv_auc", cv_auc)
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_candidate_{k}", v)
                    for k, v in test_proxy.items():
                        mlflow.log_metric(f"test_proxy_{k}", v)
                    mlflow.log_artifact(str(cv_path))
                    mlflow.log_artifact(str(valid_curve_path))
                    mlflow.log_artifact(str(test_curve_path))
                    for path in chart_paths[-6:]:
                        mlflow.log_artifact(str(path), artifact_path="charts")

        summary = pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False)
        summary_path = out_dir / "research10_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        chart_paths.append(plot_feature_set_summary(summary, chart_dir / "feature_set_summary.png"))

        top_names = summary.head(simulate_top)["feature_set"].tolist()
        mophong_all = []
        for name in top_names:
            bundle = bundles[name]
            yearly = _simulate_best(frame, bundle["scores"], bundle["selected_threshold"], name)
            yearly_path = out_dir / f"{name}_mophong_yearly.csv"
            yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
            chart_paths.append(plot_yearly_mophong(yearly, chart_dir / f"{name}_mophong_yearly.png"))
            mophong_all.append(yearly)
            bundle_file = CFG.model_dir / f"research10_{name}_bundle.joblib"
            slim = {k: v for k, v in bundle.items() if k != "scores"}
            joblib.dump(slim, bundle_file, compress=3)
            if mlflow is not None:
                mlflow.log_artifact(str(yearly_path), artifact_path="mophong")
                mlflow.log_artifact(str(bundle_file), artifact_path="models")

        if mophong_all:
            mophong_summary = pd.concat(mophong_all, ignore_index=True)
        else:
            mophong_summary = pd.DataFrame()
        mophong_summary_path = out_dir / "research10_mophong_top_yearly.csv"
        mophong_summary.to_csv(mophong_summary_path, index=False, encoding="utf-8-sig")

        meta = {
            "run_id": run_id,
            "n_iter": n_iter,
            "max_train_samples": max_train_samples,
            "simulate_top": simulate_top,
            "refit_full": refit_full,
            "elapsed_sec": round(time.time() - started, 2),
            "summary_path": str(summary_path),
            "mophong_summary_path": str(mophong_summary_path),
        }
        meta_path = out_dir / "research10_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if mlflow is not None:
            mlflow.log_params({
                "research_n_iter": n_iter,
                "research_max_train_samples": max_train_samples,
                "research_simulate_top": simulate_top,
                "research_refit_full": refit_full,
            })
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(mophong_summary_path))
            mlflow.log_artifact(str(meta_path))
            for path in chart_paths:
                mlflow.log_artifact(str(path), artifact_path="charts")

        result = {
            **meta,
            "top_summary": summary.head(10).to_dict(orient="records"),
            "mophong_top": mophong_summary.to_dict(orient="records"),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if mlflow is not None and parent_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=120000)
    parser.add_argument("--simulate-top", type=int, default=3)
    parser.add_argument("--refit-full", action="store_true")
    args = parser.parse_args()
    run(
        force_dataset=args.force_dataset,
        n_iter=args.n_iter,
        max_train_samples=args.max_train_samples,
        simulate_top=args.simulate_top,
        refit_full=args.refit_full,
    )
