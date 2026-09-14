from __future__ import annotations

import argparse
import json
import gc
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
RF_MLFLOW_DIR = RESEARCH_DIR / "rf_mlflow"
VENDOR_SCJ_DIR = REPO_ROOT / "vendor" / "scj"
for path in (REPO_ROOT, RESEARCH_DIR, RF_MLFLOW_DIR, VENDOR_SCJ_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import log as _vendor_log
    from log.MyLogger import MyLogger as _VendorMyLogger

    if not hasattr(getattr(_vendor_log, "MyLogger", None), "get_logger"):
        _vendor_log.MyLogger = _VendorMyLogger
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from research.config import CFG, ensure_dirs
from research.data_io import DATE, load_year, year_path
from research.features import RobustClipScaler, build_features, named_feature_sets
from research.labels import buy_labels_next_open
from research.mophong_adapter import orders_from_scores, simulate_orders
from research.plots import plot_model_diagnostics, plot_threshold_curve, plot_yearly_mophong
from train_rf_mlflow import _candidate_dataset, _proxy_threshold_table, _score_full_frame


FEATURE_SET = "fs03_lags_cycle"
YEAR_CACHE_WARMUP_ROWS = 20_000


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _years_all() -> tuple[int, ...]:
    return CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years


def _cache_file(name: str) -> Path:
    return RF_MLFLOW_DIR / "cache" / name


def _year_cache_file(year: int) -> Path:
    return _cache_file(f"dataset_features_labels_{year}.joblib")


def _load_optional_year(year: int) -> pd.DataFrame | None:
    if not year_path(year).exists():
        return None
    return load_year(year)


def _build_year_dataset(year: int) -> pd.DataFrame:
    """Build one target year with only boundary context loaded.

    The previous-year tail preserves rolling/EMA/HTF features at the start of
    the target year. The next-year head preserves labels near year-end, because
    labels can look forward up to ``CFG.label_max_forward`` M1 rows.
    """

    parts: list[pd.DataFrame] = []

    prev_frame = _load_optional_year(year - 1)
    if prev_frame is not None and not prev_frame.empty:
        parts.append(prev_frame.tail(YEAR_CACHE_WARMUP_ROWS))

    target_frame = load_year(year)
    parts.append(target_frame)

    next_frame = _load_optional_year(year + 1)
    if next_frame is not None and not next_frame.empty:
        parts.append(next_frame.head(int(CFG.label_max_forward) + 5))

    raw = pd.concat(parts, ignore_index=True).sort_values(DATE).reset_index(drop=True)
    frame = build_features(raw)
    frame = buy_labels_next_open(frame)
    frame["year"] = pd.to_datetime(frame[DATE]).dt.year.astype(int)
    frame = frame.loc[frame["year"] == year].copy().reset_index(drop=True)

    del raw, parts, target_frame, prev_frame, next_frame
    gc.collect()
    return frame


def build_cached_dataset(force: bool = False) -> pd.DataFrame:
    """Build/read fs03 old-scaling dataset cache by year to avoid RAM spikes.

    This intentionally avoids the old all-years build path from
    ``train_rf_mlflow.build_cached_dataset()``, which loaded every raw year
    before feature engineering.
    """

    ensure_dirs()
    _cache_file("dummy").parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    built_years: list[int] = []

    for year in _years_all():
        cache_file = _year_cache_file(year)
        if cache_file.exists() and not force:
            print({"phase": "load_year_cache", "year": year, "path": str(cache_file)}, flush=True)
            frame = joblib.load(cache_file)
        else:
            print({"phase": "build_year_cache", "year": year, "path": str(cache_file)}, flush=True)
            frame = _build_year_dataset(year)
            joblib.dump(frame, cache_file, compress=3)
            built_years.append(year)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True).sort_values(DATE).reset_index(drop=True)
    print(
        {
            "phase": "dataset_cache_ready",
            "years": list(_years_all()),
            "built_years": built_years,
            "rows": int(len(combined)),
            "cache_mode": "per_year",
        },
        flush=True,
    )
    return combined


def _param_distributions() -> dict:
    return {
        "n_estimators": [120, 180, 260, 360, 520],
        "max_depth": [5, 6, 8, 10, 12, None],
        "min_samples_leaf": [80, 150, 250, 400, 700, 1000],
        "min_samples_split": [200, 500, 800, 1200, 1800, 2600],
        "max_features": ["sqrt", "log2", 0.35, 0.50, 0.70],
        "max_samples": [0.55, 0.70, 0.85],
        "class_weight": ["balanced_subsample", "balanced", None],
    }


def _sample_balanced(x, y, max_samples: int, seed: int):
    if len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    half = max_samples // 2
    pos_n = min(len(pos), half)
    neg_n = min(len(neg), max_samples - pos_n)
    idx = np.concatenate([rng.choice(pos, pos_n, replace=False), rng.choice(neg, neg_n, replace=False)])
    rng.shuffle(idx)
    return x[idx], y[idx]


def _fit_search(x_train, y_train, n_iter: int, max_train_samples: int, seed: int):
    xs, ys = _sample_balanced(x_train, y_train, max_train_samples, seed)
    model = RandomForestClassifier(bootstrap=True, n_jobs=-1, random_state=seed)
    search = RandomizedSearchCV(
        model,
        param_distributions=_param_distributions(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=False),
        random_state=seed,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    search.fit(xs, ys)
    return search.best_estimator_, search.best_params_, float(search.best_score_), pd.DataFrame(search.cv_results_)


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


def _select_threshold_by_valid(curve: pd.DataFrame) -> float:
    grouped = curve.groupby("threshold", as_index=False).agg(
        resolved=("resolved", "sum"),
        wins=("wins", "sum"),
    )
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    grouped["deal_penalty"] = (grouped["resolved"] - 3000).abs() / 3000
    grouped["score"] = grouped["winrate"] - 3.0 * grouped["deal_penalty"]
    viable = grouped[grouped["resolved"] >= 2200].copy()
    if viable.empty:
        viable = grouped.copy()
    viable = viable.sort_values(["score", "winrate", "resolved"], ascending=[False, False, False])
    return float(viable.iloc[0]["threshold"])


def _proxy_total(curve: pd.DataFrame, threshold: float) -> dict:
    rows = curve[curve["threshold"] == threshold]
    wins = int(rows["wins"].sum())
    resolved = int(rows["resolved"].sum())
    return {
        "resolved": resolved,
        "wins": wins,
        "losses": int(rows["losses"].sum()),
        "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
        "min_year_deals": int(rows["resolved"].min()) if len(rows) else 0,
    }


def _transformed_frame(x: np.ndarray, cols: list[str], y: np.ndarray | None = None) -> pd.DataFrame:
    out = pd.DataFrame(x, columns=cols)
    if y is not None:
        out["label"] = y
    return out


def plot_pairplot_top10(x_train: np.ndarray, y_train: np.ndarray, cols: list[str], model, out_path: Path, max_rows: int, seed: int) -> Path:
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    top10 = importances.head(10).index.tolist()
    frame = _transformed_frame(x_train, cols, y_train)[top10 + ["label"]].dropna()
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    frame["label"] = frame["label"].map({0: "Lose", 1: "Win"})
    grid = sns.pairplot(frame, vars=top10, hue="label", corner=True, plot_kws={"s": 8, "alpha": 0.35}, diag_kws={"common_norm": False})
    grid.fig.suptitle("fs03 old-scaling top 10 RF features", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(grid.fig)
    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        out_path.with_name(out_path.stem + "_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return out_path


def plot_shap(model, x_valid: np.ndarray, cols: list[str], out_dir: Path, max_rows: int, seed: int) -> list[Path]:
    import shap

    frame = _transformed_frame(x_valid, cols).dropna()
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, frame, plot_type="bar", show=False, max_display=20)
    bar = out_dir / "fs03_old_scaling_shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(bar)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, frame, show=False, max_display=20)
    bee = out_dir / "fs03_old_scaling_shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(bee, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(bee)
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp_path = out_dir / "fs03_old_scaling_shap_importance.csv"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
    paths.append(imp_path)
    return paths


def simulate_yearly(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    year_arr = frame["year"].to_numpy()
    for year in CFG.split.test_years:
        mask = year_arr == year
        chunk = frame.loc[mask].copy().reset_index(drop=True)
        score_chunk = scores[mask]
        orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"feature_set": FEATURE_SET, "scaling": "old_RobustClipScaler", "year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(CFG.outputs_dir / f"fs03_old_scaling_mophong_deals_{year}.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def run(n_iter: int, max_train_samples: int, pairplot_rows: int, shap_rows: int, force_dataset: bool) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"fs03_old_scaling_{run_id}"
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

    model, best_params, cv_auc, cv_results = _fit_search(x_train, y_train, n_iter, max_train_samples, seed=3303)
    p_valid = model.predict_proba(x_valid)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    valid_metrics = _metrics(y_valid, p_valid)
    test_metrics = _metrics(y_test, p_test)

    scores = _score_full_frame(frame, cols, scaler, model)
    valid_curve = _proxy_threshold_table(frame, scores, CFG.split.valid_years)
    test_curve = _proxy_threshold_table(frame, scores, CFG.split.test_years)
    threshold = _select_threshold_by_valid(valid_curve)
    proxy = _proxy_total(test_curve, threshold)
    yearly = simulate_yearly(frame, scores, threshold)

    cv_path = out_dir / "fs03_old_scaling_cv_results.csv"
    valid_curve_path = out_dir / "fs03_old_scaling_threshold_valid.csv"
    test_curve_path = out_dir / "fs03_old_scaling_threshold_test.csv"
    yearly_path = out_dir / "fs03_old_scaling_mophong_yearly.csv"
    summary_path = out_dir / "fs03_old_scaling_summary.json"
    model_path = CFG.model_dir / f"fs03_old_scaling_{run_id}_bundle.joblib"
    cv_results.to_csv(cv_path, index=False, encoding="utf-8-sig")
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": cols, "selected_threshold": threshold, "best_params": best_params}, model_path, compress=3)

    chart_paths = []
    chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, "fs03_old_scaling_valid")
    chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / "fs03_old_scaling_threshold_valid.png", "fs03 old scaling validation threshold"))
    chart_paths.append(plot_threshold_curve(test_curve, chart_dir / "fs03_old_scaling_threshold_test.png", "fs03 old scaling test proxy threshold"))
    chart_paths.append(plot_yearly_mophong(yearly, chart_dir / "fs03_old_scaling_mophong_yearly.png"))
    pairplot = plot_pairplot_top10(x_train, y_train, cols, model, chart_dir / "fs03_old_scaling_pairplot_top10.png", pairplot_rows, 3303)
    chart_paths.append(pairplot)
    fi_path = pairplot.with_name(pairplot.stem + "_feature_importance.csv")
    shap_paths = plot_shap(model, x_valid, cols, chart_dir, shap_rows, 3303)
    chart_paths += shap_paths

    total_wr = round(yearly["wins"].sum() * 100 / max(yearly["resolved"].sum(), 1), 4)
    summary = {
        "run_id": run_id,
        "feature_set": FEATURE_SET,
        "scaling": "old_RobustClipScaler",
        "n_features": len(cols),
        "cv_auc": cv_auc,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "selected_threshold": threshold,
        "test_proxy": proxy,
        "mophong_total_deals": int(yearly["deals"].sum()),
        "mophong_total_resolved": int(yearly["resolved"].sum()),
        "mophong_total_wr": total_wr,
        "best_params": best_params,
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - started, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    mlflow = _mlflow()
    if mlflow:
        with mlflow.start_run(run_name=f"fs03_old_scaling_{run_id}"):
            mlflow.log_param("feature_set", FEATURE_SET)
            mlflow.log_param("scaling", "old_RobustClipScaler")
            mlflow.log_param("n_features", len(cols))
            mlflow.log_param("selected_threshold", threshold)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("cv_auc", cv_auc)
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_candidate_{k}", v)
            for k, v in proxy.items():
                mlflow.log_metric(f"proxy_test_{k}", v)
            mlflow.log_metric("mophong_total_wr", total_wr)
            mlflow.log_metric("mophong_total_deals", int(yearly["deals"].sum()))
            for _, row in yearly.iterrows():
                mlflow.log_metric(f"mophong_{int(row['year'])}_winrate", float(row["winrate"]))
                mlflow.log_metric(f"mophong_{int(row['year'])}_deals", float(row["deals"]))
            for p in [cv_path, valid_curve_path, test_curve_path, yearly_path, summary_path, model_path, fi_path]:
                mlflow.log_artifact(str(p))
            for p in chart_paths:
                mlflow.log_artifact(str(p), artifact_path="shap" if "shap_" in p.name else "charts")

    print(json.dumps(summary | {"yearly": yearly.to_dict(orient="records")}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=60000)
    parser.add_argument("--pairplot-rows", type=int, default=1000)
    parser.add_argument("--shap-rows", type=int, default=500)
    parser.add_argument("--force-dataset", action="store_true")
    args = parser.parse_args()
    run(args.n_iter, args.max_train_samples, args.pairplot_rows, args.shap_rows, args.force_dataset)
