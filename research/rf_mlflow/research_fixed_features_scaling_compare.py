from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import CFG, PROJECT_ROOT, ensure_dirs

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method.OverlayIndicatorsNormalization import (
    rolling_minmax_scale,
    rolling_rank_scale,
    rolling_wma_zscore,
    rolling_zscore,
)
from plots import plot_model_diagnostics, plot_threshold_curve
from research_random50_initial_features import (
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


FEATURE_FAMILY = "fixed_features_scaling_compare"
BASE_FEATURES = [
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    #"atr14",
    "session_ny",
    'h4_volatility_6',
    'h4_volatility_12',
    "sig_bb_reversion_buy",
    "sig_pullback_trend_buy",
]
PASSTHROUGH_FEATURES = {
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "session_ny",
    "sig_bb_reversion_buy",
    "sig_pullback_trend_buy",
}


class RollingFunctionScaler:
    def __init__(
        self,
        method: str,
        feature_columns: list[str],
        window: int = 14,
        min_periods: int = 1,
        use_past_only: bool = False,
        eps: float = 1e-12,
    ):
        self.method = method
        self.feature_columns = feature_columns
        self.window = window
        self.min_periods = min_periods
        self.use_past_only = use_past_only
        self.eps = eps

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype="float32")
        out = np.empty_like(arr, dtype="float32")
        for i in range(arr.shape[1]):
            values = arr[:, i]
            if self.feature_columns[i] in PASSTHROUGH_FEATURES:
                out[:, i] = values
                continue
            if self.method == "rolling_zscore":
                scaled = rolling_zscore(
                    values,
                    window=self.window,
                    min_periods=self.min_periods,
                    use_past_only=self.use_past_only,
                    eps=self.eps,
                )
            elif self.method == "rolling_wma_zscore":
                scaled = rolling_wma_zscore(
                    values,
                    window=self.window,
                    min_periods=self.min_periods,
                    use_past_only=self.use_past_only,
                    eps=self.eps,
                )
            elif self.method == "rolling_minmax":
                scaled = rolling_minmax_scale(
                    values,
                    window=self.window,
                    min_periods=self.min_periods,
                    use_past_only=self.use_past_only,
                    eps=self.eps,
                )
            elif self.method == "rolling_rank":
                scaled = rolling_rank_scale(
                    values,
                    window=self.window,
                    min_periods=self.min_periods,
                    use_past_only=self.use_past_only,
                )
            else:
                raise ValueError(f"Unsupported rolling scaler: {self.method}")
            out[:, i] = scaled.to_numpy(dtype="float32", copy=False)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _make_scalers(n_train: int, feature_columns: list[str]) -> dict:
    return {
        "rolling_zscore": RollingFunctionScaler("rolling_zscore", feature_columns, window=500, min_periods=100, use_past_only=True),
        #"rolling_wma_zscore": RollingFunctionScaler("rolling_wma_zscore", feature_columns, window=14, min_periods=1, use_past_only=False),
        "rolling_minmax": RollingFunctionScaler("rolling_minmax", feature_columns, window=500, min_periods=100, use_past_only=True),
        "rolling_rank": RollingFunctionScaler("rolling_rank", feature_columns, window=500, min_periods=100, use_past_only=True),
    }


def _fit_transform_scaler(scaler, x_train, x_valid, x_test):
    scaler.fit(x_train)
    return scaler, scaler.transform(x_train.copy()).astype("float32"), scaler.transform(x_valid.copy()).astype("float32"), scaler.transform(x_test.copy()).astype("float32")


def _plot_scaler_feature_box(data: dict[str, np.ndarray], cols: list[str], out_path: Path, title: str) -> Path:
    top_cols = cols[: min(16, len(cols))]
    frames = []
    rng = np.random.default_rng(42)
    for split, arr in data.items():
        idx = np.arange(arr.shape[0])
        if len(idx) > 5000:
            idx = rng.choice(idx, size=5000, replace=False)
        small = pd.DataFrame(arr[idx, : len(cols)], columns=cols)
        melted = small[top_cols].melt(var_name="feature", value_name="value")
        melted["split"] = split
        frames.append(melted)
    frame = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    frame.boxplot(column="value", by=["feature", "split"], ax=ax, rot=90, showfliers=False)
    fig.suptitle("")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("scaled value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_compare_summary(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    ordered = summary.sort_values("test_total_wr", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ordered["scaler"], ordered["test_total_wr"], color=["#2ca02c" if v >= 58 else "#ff7f0e" for v in ordered["test_total_wr"]])
    ax.axvline(58, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Test WR % at valid-selected threshold")
    ax.set_title("Scaling comparison: WR")
    fig.tight_layout()
    p = out_dir / "scaling_compare_wr.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ordered["scaler"], ordered["test_total_resolved"])
    ax.axvspan(7000, 10000, color="green", alpha=0.08)
    ax.set_xlabel("Resolved deals 2024-2026")
    ax.set_title("Scaling comparison: deal count")
    fig.tight_layout()
    p = out_dir / "scaling_compare_deals.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def run(max_train_rows: int, shap_rows: int, pairplot_rows: int, seed: int, force_cache: bool, exclude_features: list[str] | None = None) -> dict:
    ensure_dirs()
    run_id = int(time.time())
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    explain_dir = out_dir / "explainability"
    chart_dir = out_dir / "charts"
    explain_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    exclude_features = exclude_features or []
    cols = [c for c in BASE_FEATURES if c not in set(exclude_features)]
    all_cols = set(build_cache(force_cache))
    missing = [c for c in cols if c not in all_cols]
    if missing:
        raise RuntimeError(f"Missing required features: {missing}")
    passthrough_cols = [c for c in cols if c in PASSTHROUGH_FEATURES]
    rolling_scaled_cols = [c for c in cols if c not in PASSTHROUGH_FEATURES]
    (out_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "features": cols,
                "n_features": len(cols),
                "excluded_features": exclude_features,
                "rolling_scaled_features": rolling_scaled_cols,
                "passthrough_features": passthrough_cols,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    read_cols = list(dict.fromkeys(cols + ["label", "year", "session_london_ny"]))
    print({"phase": "load_data", "features": len(cols)}, flush=True)
    train = _sample_train(_load_years(CFG.split.train_years, read_cols), max_train_rows, seed)
    valid = _load_years(CFG.split.valid_years, read_cols)
    test = _load_years(CFG.split.test_years, read_cols)
    x_train_raw = train[cols].to_numpy("float32", copy=True)
    y_train = train["label"].to_numpy("int8", copy=True)
    x_valid_raw = valid[cols].to_numpy("float32", copy=True)
    y_valid = valid["label"].to_numpy("int8", copy=True)
    x_test_raw = test[cols].to_numpy("float32", copy=True)
    y_test = test["label"].to_numpy("int8", copy=True)

    params = {
        "n_estimators": 90,
        "max_depth": 10,
        "min_samples_leaf": 500,
        "min_samples_split": 1200,
        "max_features": "sqrt",
        "max_samples": 0.75,
        "class_weight": "balanced_subsample",
    }

    mlflow = _mlflow()
    parent = mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}") if mlflow else None
    summary_rows = []
    try:
        for scaler_name, scaler in _make_scalers(len(x_train_raw), cols).items():
            print({"phase": "scaler", "name": scaler_name}, flush=True)
            model_dir = explain_dir / scaler_name
            model_dir.mkdir(parents=True, exist_ok=True)
            scaler, x_train, x_valid, x_test = _fit_transform_scaler(scaler, x_train_raw, x_valid_raw, x_test_raw)

            model = RandomForestClassifier(**params, bootstrap=True, random_state=seed, n_jobs=1)
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
            imp_path = model_dir / f"{scaler_name}_rf_importance.csv"
            valid_curve_path = model_dir / f"{scaler_name}_threshold_valid.csv"
            test_curve_path = model_dir / f"{scaler_name}_threshold_test.csv"
            imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

            chart_paths = []
            chart_paths += plot_model_diagnostics(y_valid, p_valid, model_dir, f"{scaler_name}_valid")
            chart_paths.append(plot_threshold_curve(valid_curve, model_dir / f"{scaler_name}_threshold_valid.png", f"{scaler_name} valid threshold"))
            chart_paths.append(plot_threshold_curve(test_curve, model_dir / f"{scaler_name}_threshold_test.png", f"{scaler_name} test threshold"))
            chart_paths.append(_plot_importance(imp, model_dir / f"{scaler_name}_rf_importance_top25.png", f"{scaler_name} RF importance"))
            chart_paths += _plot_shap(model, x_valid, cols, model_dir, scaler_name, shap_rows, seed)
            chart_paths.append(_plot_pairplot(model, x_train, y_train, cols, model_dir, scaler_name, pairplot_rows, seed))
            chart_paths.append(_plot_scaler_feature_box({"train": x_train, "valid": x_valid, "test": x_test}, cols, model_dir / f"{scaler_name}_scaled_boxplot.png", f"{scaler_name} scaled feature distribution"))

            bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_{scaler_name}_bundle.joblib"
            joblib.dump({"model": model, "scaler": scaler, "feature_columns": cols, "feature_family": FEATURE_FAMILY, "selected_threshold": threshold, "params": params}, bundle_path, compress=3)

            row = {
                "scaler": scaler_name,
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
            summary_rows.append(row)
            pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False).to_csv(out_dir / "summary_checkpoint.csv", index=False, encoding="utf-8-sig")

            if mlflow:
                with mlflow.start_run(run_name=f"{scaler_name}_{run_id}", nested=True):
                    mlflow.log_params({"scaler": scaler_name, "n_features": len(cols), **params, "threshold": threshold})
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_{k}", v)
                    mlflow.log_metric("test_total_wr", test_total["winrate"])
                    mlflow.log_metric("test_total_resolved", test_total["resolved"])
                    mlflow.log_metric("test_min_year_deals", test_total["min_year_deals"])
                    mlflow.log_metric("objective_score", objective)
                    for path in [imp_path, valid_curve_path, test_curve_path, bundle_path, *chart_paths]:
                        mlflow.log_artifact(str(path), artifact_path=f"scalers/{scaler_name}")

            del x_train, x_valid, x_test, model, p_valid, p_test, scaler
            gc.collect()

        summary = pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        compare_charts = _plot_compare_summary(summary, chart_dir)
        (out_dir / "summary.json").write_text(json.dumps({"run_id": run_id, "rows": summary_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        if mlflow:
            for path in [summary_path, out_dir / "selected_features.json", out_dir / "summary.json", *compare_charts]:
                mlflow.log_artifact(str(path), artifact_path="summary")
    finally:
        if mlflow and parent:
            mlflow.end_run()

    print(json.dumps({"out_dir": str(out_dir), "best": summary.iloc[0].to_dict()}, ensure_ascii=False, indent=2), flush=True)
    return {"out_dir": str(out_dir), "best": summary.iloc[0].to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--shap-rows", type=int, default=600)
    parser.add_argument("--pairplot-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=54010)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--exclude-features", nargs="*", default=[])
    args = parser.parse_args()
    run(args.max_train_rows, args.shap_rows, args.pairplot_rows, args.seed, args.force_cache, args.exclude_features)


if __name__ == "__main__":
    main()
