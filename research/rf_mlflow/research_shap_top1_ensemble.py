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

from config import CFG, ensure_dirs
from plots import plot_model_diagnostics, plot_threshold_curve
from research_fs14_overlay_norm_light import RobustClipScalerLite
from research_random50_initial_features import (
    BASELINE,
    _load_years,
    _metrics,
    _mlflow,
    _plot_importance,
    _plot_pairplot,
    _plot_shap,
    _sample_train,
    _score_objective,
    _select_threshold,
    _threshold_table,
    _total_at_threshold,
    build_cache,
)


FEATURE_FAMILY = "shap_top1_ensemble_initial_non_bb"
MODEL_NAME = "rf_shap_top1_ensemble"


def _is_excluded_feature(feature: str) -> bool:
    low = feature.lower()
    return low.startswith(("bb", "bband")) or "_bb" in low or "boll" in low


def collect_top1_shap_features(outputs_dir: Path, available_cols: set[str]) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict] = []
    selected: list[str] = []
    seen: set[str] = set()
    for path in sorted(outputs_dir.rglob("*shap_importance.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append({"path": str(path), "feature": "", "mean_abs_shap": np.nan, "status": f"read_error:{exc!r}"})
            continue
        if "feature" not in df.columns:
            rows.append({"path": str(path), "feature": "", "mean_abs_shap": np.nan, "status": "missing_feature_column"})
            continue
        value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else None
        if value_col is None:
            candidates = [c for c in df.columns if c != "feature" and pd.api.types.is_numeric_dtype(df[c])]
            value_col = candidates[0] if candidates else None
        if value_col is None:
            rows.append({"path": str(path), "feature": "", "mean_abs_shap": np.nan, "status": "missing_shap_value_column"})
            continue
        ranked = df.dropna(subset=["feature"]).copy()
        ranked = ranked[~ranked["feature"].astype(str).map(_is_excluded_feature)]
        ranked = ranked[ranked["feature"].astype(str).isin(available_cols)]
        if ranked.empty:
            rows.append({"path": str(path), "feature": "", "mean_abs_shap": np.nan, "status": "no_available_non_bb_feature"})
            continue
        ranked = ranked.sort_values(value_col, ascending=False)
        feature = str(ranked.iloc[0]["feature"])
        value = float(ranked.iloc[0][value_col])
        status = "selected_new" if feature not in seen else "selected_duplicate"
        rows.append({"path": str(path), "feature": feature, "mean_abs_shap": value, "status": status})
        if feature not in seen:
            seen.add(feature)
            selected.append(feature)

    for feature in BASELINE:
        if feature in available_cols and feature not in seen:
            selected.insert(0, feature)
            seen.add(feature)

    return selected, pd.DataFrame(rows)


def _plot_selected_feature_votes(top1: pd.DataFrame, out_path: Path) -> Path:
    valid = top1[top1["feature"].astype(str).ne("")].copy()
    if valid.empty:
        out_path.write_text("No selected features", encoding="utf-8")
        return out_path
    counts = valid.groupby("feature", as_index=False).agg(votes=("path", "count"), max_mean_abs_shap=("mean_abs_shap", "max"))
    counts = counts.sort_values(["votes", "max_mean_abs_shap"], ascending=False).head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(counts))))
    ax.barh(counts["feature"], counts["votes"])
    ax.set_xlabel("Top-1 SHAP votes across previous models")
    ax.set_title("Selected feature votes from previous SHAP importance files")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run(force_cache: bool, max_train_rows: int, shap_rows: int, pairplot_rows: int, seed: int) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    run_id = int(time.time())
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    model_dir = out_dir / "explainability" / MODEL_NAME
    chart_dir = out_dir / "charts"
    model_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    all_cols = build_cache(force_cache)
    available_cols = set(all_cols)
    cols, top1_table = collect_top1_shap_features(CFG.outputs_dir, available_cols)
    if not cols:
        raise RuntimeError("No usable SHAP top-1 features found.")

    feature_path = out_dir / "selected_features.json"
    top1_path = out_dir / "source_top1_shap_features.csv"
    feature_path.write_text(json.dumps({"features": cols, "n_features": len(cols)}, ensure_ascii=False, indent=2), encoding="utf-8")
    top1_table.to_csv(top1_path, index=False, encoding="utf-8-sig")
    vote_chart = _plot_selected_feature_votes(top1_table, chart_dir / "top1_shap_feature_votes.png")

    params = {
        "n_estimators": 90,
        "max_depth": 10,
        "min_samples_leaf": 500,
        "min_samples_split": 1200,
        "max_features": "sqrt",
        "max_samples": 0.75,
        "class_weight": "balanced_subsample",
    }

    read_cols = list(dict.fromkeys(cols + ["label", "year", "session_london_ny"]))
    print({"phase": "load_data", "n_features": len(cols)}, flush=True)
    train = _sample_train(_load_years(CFG.split.train_years, read_cols), max_train_rows, seed)
    valid = _load_years(CFG.split.valid_years, read_cols)
    test = _load_years(CFG.split.test_years, read_cols)

    x_train = train[cols].to_numpy("float32", copy=True)
    y_train = train["label"].to_numpy("int8", copy=True)
    x_valid = valid[cols].to_numpy("float32", copy=True)
    y_valid = valid["label"].to_numpy("int8", copy=True)
    x_test = test[cols].to_numpy("float32", copy=True)
    y_test = test["label"].to_numpy("int8", copy=True)

    scaler = RobustClipScalerLite().fit(x_train)
    x_train = scaler.transform_inplace(x_train)
    x_valid = scaler.transform_inplace(x_valid)
    x_test = scaler.transform_inplace(x_test)

    model = RandomForestClassifier(**params, bootstrap=True, random_state=seed, n_jobs=1)
    print({"phase": "fit_rf", "params": params}, flush=True)
    model.fit(x_train, y_train)

    p_valid = model.predict_proba(x_valid)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    valid_metrics = _metrics(y_valid, p_valid)
    test_metrics = _metrics(y_test, p_test)
    valid_curve = _threshold_table(valid, p_valid, CFG.threshold_grid)
    test_curve = _threshold_table(test, p_test, CFG.threshold_grid)
    threshold = _select_threshold(valid_curve)
    test_total = _total_at_threshold(test_curve, threshold)
    objective = _score_objective(test_total, test_curve, threshold, test_metrics["auc"])

    imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    imp_path = model_dir / f"{MODEL_NAME}_rf_importance.csv"
    valid_curve_path = model_dir / f"{MODEL_NAME}_threshold_valid.csv"
    test_curve_path = model_dir / f"{MODEL_NAME}_threshold_test.csv"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

    chart_paths = [vote_chart]
    chart_paths += plot_model_diagnostics(y_valid, p_valid, model_dir, f"{MODEL_NAME}_valid")
    chart_paths.append(plot_threshold_curve(valid_curve, model_dir / f"{MODEL_NAME}_threshold_valid.png", f"{MODEL_NAME} valid threshold"))
    chart_paths.append(plot_threshold_curve(test_curve, model_dir / f"{MODEL_NAME}_threshold_test.png", f"{MODEL_NAME} test threshold"))
    chart_paths.append(_plot_importance(imp, model_dir / f"{MODEL_NAME}_rf_importance_top25.png", f"{MODEL_NAME} RF importance"))
    chart_paths += _plot_shap(model, x_valid, cols, model_dir, MODEL_NAME, shap_rows, seed)
    chart_paths.append(_plot_pairplot(model, x_train, y_train, cols, model_dir, MODEL_NAME, pairplot_rows, seed))

    bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_{MODEL_NAME}_bundle.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": cols,
            "feature_family": FEATURE_FAMILY,
            "selected_threshold": threshold,
            "params": params,
        },
        bundle_path,
        compress=3,
    )

    row = {
        "model_name": MODEL_NAME,
        "n_features": len(cols),
        "threshold": threshold,
        "valid_auc": valid_metrics["auc"],
        "valid_brier": valid_metrics["brier"],
        "test_auc": test_metrics["auc"],
        "test_brier": test_metrics["brier"],
        "test_total_resolved": test_total["resolved"],
        "test_total_wr": test_total["winrate"],
        "test_min_year_deals": test_total["min_year_deals"],
        "objective_score": objective,
        "top10_features": "|".join(imp.head(10)["feature"].tolist()),
        "model_path": str(bundle_path),
    }
    summary = pd.DataFrame([row])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps({"run_id": run_id, **row}, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow()
    if mlflow:
        with mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}"):
            mlflow.log_params({"model_name": MODEL_NAME, "n_features": len(cols), **params, "threshold": threshold})
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.log_metric("test_total_wr", test_total["winrate"])
            mlflow.log_metric("test_total_resolved", test_total["resolved"])
            mlflow.log_metric("test_min_year_deals", test_total["min_year_deals"])
            mlflow.log_metric("objective_score", objective)
            for path in [summary_path, feature_path, top1_path, imp_path, valid_curve_path, test_curve_path, bundle_path, *chart_paths]:
                if Path(path).exists():
                    mlflow.log_artifact(str(path), artifact_path=MODEL_NAME)

    del train, valid, test, x_train, y_train, x_valid, y_valid, x_test, y_test, p_valid, p_test, model, scaler
    gc.collect()
    print(json.dumps({"out_dir": str(out_dir), **row}, ensure_ascii=False, indent=2), flush=True)
    return {"out_dir": str(out_dir), **row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=180000)
    parser.add_argument("--shap-rows", type=int, default=800)
    parser.add_argument("--pairplot-rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=53010)
    args = parser.parse_args()
    run(args.force_cache, args.max_train_rows, args.shap_rows, args.pairplot_rows, args.seed)


if __name__ == "__main__":
    main()
