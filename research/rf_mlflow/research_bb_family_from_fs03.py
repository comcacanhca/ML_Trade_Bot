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

from config import CFG, PROJECT_ROOT, ensure_dirs
from features import RobustClipScaler, named_feature_sets
from mophong_adapter import orders_from_scores, simulate_orders
from plots import plot_feature_set_summary, plot_model_diagnostics, plot_threshold_curve, plot_yearly_mophong
from research_10_feature_sets import (
    _fit_search_model,
    _metrics,
    _objective,
    _sample_for_search,
    _select_threshold_by_constraints,
    _total_proxy_stats,
)
from train_rf_mlflow import _add_train_explainability, _candidate_dataset, _proxy_threshold_table, _score_full_frame, build_cached_dataset


BB_FEATURE_SETS = (
    "fs11_fs03_bb_core",
    "fs12_bb_reversion_multiband",
    "fs13_bb_width_regime",
)

FS03_FIXED_PARAMS = {
    "n_estimators": 220,
    "min_samples_split": 1200,
    "min_samples_leaf": 250,
    "max_samples": 0.85,
    "max_features": 0.7,
    "max_depth": None,
    "class_weight": None,
}


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _fit_fixed(x_train: np.ndarray, y_train: np.ndarray, seed: int, fit_sample: int = 0) -> RandomForestClassifier:
    model = RandomForestClassifier(
        **FS03_FIXED_PARAMS,
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )
    if fit_sample and len(y_train) > fit_sample:
        x_fit, y_fit = _sample_for_search(x_train, y_train, fit_sample, seed)
    else:
        x_fit, y_fit = x_train, y_train
    model.fit(x_fit, y_fit)
    return model


def _simulate_yearly(frame: pd.DataFrame, scores: np.ndarray, threshold: float, feature_set: str, out_dir: Path) -> pd.DataFrame:
    rows = []
    year_arr = frame["year"].to_numpy()
    for year in CFG.split.test_years:
        mask = year_arr == year
        chunk = frame.loc[mask].copy().reset_index(drop=True)
        orders = orders_from_scores(chunk, scores[mask], threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"feature_set": feature_set, "year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(
            out_dir / f"{feature_set}_mophong_deals_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return pd.DataFrame(rows)


def run(
    force_dataset: bool,
    search: bool,
    n_iter: int,
    max_train_samples: int,
    simulate_top: int,
    explainability_top: int,
    feature_sets: tuple[str, ...],
    fixed_fit_sample: int,
) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"research_bb_family_{run_id}"
    chart_dir = out_dir / "charts"
    explain_dir = out_dir / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    explain_dir.mkdir(parents=True, exist_ok=True)

    frame = build_cached_dataset(force=force_dataset)
    all_sets = named_feature_sets(frame)
    selected_sets = feature_sets or BB_FEATURE_SETS
    missing = [name for name in selected_sets if name not in all_sets]
    if missing:
        raise KeyError(f"Missing feature sets: {missing}")

    mlflow = _mlflow()
    parent_run = mlflow.start_run(run_name=f"research_bb_family_from_fs03_{run_id}") if mlflow is not None else None
    summary_rows: list[dict] = []
    bundles: dict[str, dict] = {}
    all_chart_paths: list[Path] = []

    try:
        for idx, name in enumerate(selected_sets, start=1):
            seed = 3100 + idx
            cols = all_sets[name]
            print({"phase": "feature_set", "name": name, "n_features": len(cols)}, flush=True)

            x_train_raw, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
            x_valid_raw, y_valid, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
            x_test_raw, y_test, _ = _candidate_dataset(frame, cols, CFG.split.test_years)

            scaler = RobustClipScaler().fit(x_train_raw)
            x_train = scaler.transform(x_train_raw)
            x_valid = scaler.transform(x_valid_raw)
            x_test = scaler.transform(x_test_raw)

            cv_auc = np.nan
            cv_results = pd.DataFrame()
            if search:
                model, best_params, cv_auc, cv_results = _fit_search_model(
                    name,
                    x_train,
                    y_train,
                    n_iter=n_iter,
                    max_train_samples=max_train_samples,
                    seed=seed,
                    refit_full=True,
                )
            else:
                model = _fit_fixed(x_train, y_train, seed=1003, fit_sample=fixed_fit_sample)
                best_params = FS03_FIXED_PARAMS.copy()
                if fixed_fit_sample:
                    best_params["fixed_fit_sample"] = fixed_fit_sample

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
            if len(cv_results):
                cv_results.to_csv(cv_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

            chart_paths = []
            chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, f"{name}_valid")
            chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / f"{name}_threshold_valid.png", f"{name} valid threshold curve"))
            chart_paths.append(plot_threshold_curve(test_curve, chart_dir / f"{name}_threshold_test.png", f"{name} test proxy threshold curve"))
            all_chart_paths += chart_paths

            model_file = CFG.model_dir / f"bb_family_{run_id}_{name}_bundle.joblib"
            bundle = {
                "model": model,
                "scaler": scaler,
                "feature_columns": cols,
                "selected_threshold": threshold,
                "best_params": best_params,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
                "test_proxy": test_proxy,
                "feature_set": name,
                "run_id": run_id,
            }
            joblib.dump(bundle, model_file, compress=3)
            bundles[name] = {**bundle, "scores": scores, "x_train": x_train, "y_train": y_train, "x_valid": x_valid}

            summary_rows.append(
                {
                    "feature_set": name,
                    "n_features": len(cols),
                    "mode": "search" if search else "fs03_fixed_params",
                    "cv_auc": cv_auc,
                    "valid_auc": valid_metrics.get("auc", 0.0),
                    "valid_brier": valid_metrics.get("brier", 0.0),
                    "test_auc": test_metrics.get("auc", 0.0),
                    "selected_threshold": threshold,
                    "test_total_deals": test_proxy["resolved"],
                    "test_total_wr": test_proxy["winrate"],
                    "test_min_year_deals": test_proxy["min_year_deals"],
                    "objective_score": objective_score,
                    "model_file": str(model_file),
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                }
            )

            if mlflow is not None:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_param("feature_set", name)
                    mlflow.log_param("n_features", len(cols))
                    mlflow.log_param("mode", "search" if search else "fs03_fixed_params")
                    mlflow.log_param("selected_threshold", threshold)
                    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                    if np.isfinite(cv_auc):
                        mlflow.log_metric("cv_auc", float(cv_auc))
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_candidate_{k}", v)
                    for k, v in test_proxy.items():
                        mlflow.log_metric(f"test_proxy_{k}", v)
                    if len(cv_results):
                        mlflow.log_artifact(str(cv_path))
                    mlflow.log_artifact(str(valid_curve_path))
                    mlflow.log_artifact(str(test_curve_path))
                    mlflow.log_artifact(str(model_file), artifact_path="models")
                    for path in chart_paths:
                        mlflow.log_artifact(str(path), artifact_path="charts")

        summary = pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False)
        summary_path = out_dir / "bb_family_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        all_chart_paths.append(plot_feature_set_summary(summary, chart_dir / "bb_family_feature_set_summary.png"))

        mophong_frames = []
        for name in summary.head(simulate_top)["feature_set"].tolist():
            bundle = bundles[name]
            yearly = _simulate_yearly(frame, bundle["scores"], bundle["selected_threshold"], name, out_dir)
            yearly_path = out_dir / f"{name}_mophong_yearly.csv"
            yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
            year_chart = plot_yearly_mophong(yearly, chart_dir / f"{name}_mophong_yearly.png")
            all_chart_paths.append(year_chart)
            mophong_frames.append(yearly)
            if mlflow is not None:
                mlflow.log_artifact(str(yearly_path), artifact_path="mophong")
                mlflow.log_artifact(str(year_chart), artifact_path="charts")

        explainability_paths: list[Path] = []
        for name in summary.head(explainability_top)["feature_set"].tolist():
            bundle = bundles[name]
            paths = _add_train_explainability(
                model=bundle["model"],
                feature_cols=bundle["feature_columns"],
                x_train=bundle["x_train"],
                y_train=bundle["y_train"],
                x_valid=bundle["x_valid"],
                out_dir=explain_dir / name,
                pairplot_rows=800,
                shap_rows=500,
                seed=4100,
            )
            explainability_paths += paths
            if mlflow is not None:
                for path in paths:
                    artifact_path = f"explainability/{name}/shap" if "shap" in path.name else f"explainability/{name}"
                    mlflow.log_artifact(str(path), artifact_path=artifact_path)

        mophong_summary = pd.concat(mophong_frames, ignore_index=True) if mophong_frames else pd.DataFrame()
        mophong_summary_path = out_dir / "bb_family_mophong_top_yearly.csv"
        mophong_summary.to_csv(mophong_summary_path, index=False, encoding="utf-8-sig")

        meta = {
            "run_id": run_id,
            "mode": "search" if search else "fs03_fixed_params",
            "force_dataset": force_dataset,
            "n_iter": n_iter,
            "max_train_samples": max_train_samples,
            "simulate_top": simulate_top,
            "explainability_top": explainability_top,
            "feature_sets": ",".join(selected_sets),
            "fixed_fit_sample": fixed_fit_sample,
            "elapsed_sec": round(time.time() - started, 2),
            "summary_path": str(summary_path),
            "mophong_summary_path": str(mophong_summary_path),
        }
        meta_path = out_dir / "bb_family_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if mlflow is not None:
            mlflow.log_params({k: v for k, v in meta.items() if k not in {"summary_path", "mophong_summary_path"}})
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(mophong_summary_path))
            mlflow.log_artifact(str(meta_path))
            for path in all_chart_paths:
                mlflow.log_artifact(str(path), artifact_path="charts")

        result = {
            **meta,
            "top_summary": summary.to_dict(orient="records"),
            "mophong_top": mophong_summary.to_dict(orient="records"),
            "explainability_artifacts": [str(path) for path in explainability_paths],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if mlflow is not None and parent_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=120000)
    parser.add_argument("--simulate-top", type=int, default=1)
    parser.add_argument("--explainability-top", type=int, default=1)
    parser.add_argument("--feature-sets", default=",".join(BB_FEATURE_SETS))
    parser.add_argument("--fixed-fit-sample", type=int, default=180000)
    args = parser.parse_args()
    feature_sets = tuple(item.strip() for item in args.feature_sets.split(",") if item.strip())
    run(
        force_dataset=args.force_dataset,
        search=args.search,
        n_iter=args.n_iter,
        max_train_samples=args.max_train_samples,
        simulate_top=args.simulate_top,
        explainability_top=args.explainability_top,
        feature_sets=feature_sets,
        fixed_fit_sample=args.fixed_fit_sample,
    )
