from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config import CFG, PROJECT_ROOT, ensure_dirs
from features import named_feature_sets
from mophong_adapter import orders_from_scores, simulate_orders
from plots import plot_model_diagnostics, plot_threshold_curve, plot_yearly_mophong
from train_rf_mlflow import _candidate_dataset, _proxy_threshold_table, _score_full_frame, build_cached_dataset

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_FEATURE_SETS = ("fs03_lags_cycle", "fs07_bb_rsi_reversion", "fs10_mtf_full_all")


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _is_binary_like(series: pd.Series) -> bool:
    vals = pd.Series(series).dropna().unique()
    if len(vals) > 3:
        return False
    return set(vals).issubset({0, 1, 0.0, 1.0, -1, -1.0})


def _should_keep_unscaled(frame: pd.DataFrame, col: str) -> bool:
    if col.startswith("sig_") or col in {"buy_signal", "session_london_ny", "session_ny"}:
        return True
    if col.startswith("tod_") or col.startswith("dow_"):
        return True
    return _is_binary_like(frame[col])


def apply_wma_scaling(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    """Create strict-causal WMA z-score columns.

    Scaling parameters are rolling and past-only:
    row t uses x[t-14:t-1], never the current row's future context or test-year fit.
    """

    out = frame.copy()
    scaled_cols: list[str] = []
    kept_cols: list[str] = []
    for col in cols:
        if _should_keep_unscaled(out, col):
            scaled_cols.append(col)
            kept_cols.append(col)
            continue
        new_col = f"{col}__wmaz14p"
        out[new_col] = fast_wma_zscore_14_past(out[col].astype(float))
        scaled_cols.append(new_col)
    meta = {
        "scaling": "rolling_wma_zscore",
        "window": 14,
        "min_periods": 14,
        "use_past_only": True,
        "kept_unscaled_count": len(kept_cols),
        "scaled_count": len(cols) - len(kept_cols),
        "kept_unscaled": kept_cols,
    }
    return out, scaled_cols, meta


def fast_wma_zscore_14_past(values: pd.Series) -> pd.Series:
    """Vectorized equivalent for rolling_wma_zscore(window=14, min_periods=14, use_past_only=True).

    It uses the previous 14 rows only. No future row and not the current row are used.
    """

    x = pd.Series(values).reset_index(drop=True).replace([np.inf, -np.inf], np.nan).astype(float)
    ref = x.shift(1)
    arr = ref.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if n < 14:
        return pd.Series(out)

    windows = np.lib.stride_tricks.sliding_window_view(arr, 14)
    valid = np.isfinite(windows).all(axis=1)
    weights = np.arange(1, 15, dtype=float)
    wma = np.full(len(windows), np.nan, dtype=float)
    wma[valid] = windows[valid].dot(weights) / weights.sum()
    std = ref.rolling(14, min_periods=14).std().to_numpy(dtype=float)[13:]
    current = x.to_numpy(dtype=float)[13:]
    denom = np.where(np.abs(std) > 1e-12, std, np.nan)
    z = (current - wma) / denom
    same_value = np.abs(current - wma) <= 1e-12
    z = np.where(np.isfinite(denom), z, np.where(same_value, 0.0, np.nan))
    out[13:] = z
    return pd.Series(out)


def _param_distributions() -> dict:
    return {
        "n_estimators": [120, 180, 260, 360],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [80, 150, 250, 400, 700],
        "min_samples_split": [200, 500, 800, 1200, 1800],
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


def _select_threshold(curve: pd.DataFrame) -> float:
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


def plot_pairplot_top10(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    model: RandomForestClassifier,
    out_path: Path,
    max_rows: int,
    seed: int,
) -> Path:
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top10 = importances.head(10).index.tolist()
    sample_cols = top10 + ["label"]
    sample = train_frame[sample_cols].dropna()
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=seed)
    sample["label"] = sample["label"].map({0.0: "Lose", 1.0: "Win", 0: "Lose", 1: "Win"})
    grid = sns.pairplot(sample, vars=top10, hue="label", corner=True, plot_kws={"s": 8, "alpha": 0.35}, diag_kws={"common_norm": False})
    grid.fig.suptitle("Pair plot of top 10 RF features on train sample", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(grid.fig)
    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        out_path.with_name(out_path.stem + "_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return out_path


def plot_shap_diagnostics(
    model: RandomForestClassifier,
    x_frame: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    prefix: str,
    max_rows: int,
    seed: int,
) -> list[Path]:
    """Create SHAP feature charts after train.

    Uses a sampled validation/candidate frame to keep TreeSHAP runtime bounded.
    """

    import shap

    sample = x_frame[feature_cols].dropna()
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=seed)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, sample, plot_type="bar", show=False, max_display=20)
    bar_path = out_dir / f"{prefix}_shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(bar_path)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, sample, show=False, max_display=20)
    beeswarm_path = out_dir / f"{prefix}_shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(beeswarm_path)

    mean_abs = np.abs(values).mean(axis=0)
    shap_importance = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    imp_path = out_dir / f"{prefix}_shap_importance.csv"
    shap_importance.to_csv(imp_path, index=False, encoding="utf-8-sig")
    paths.append(imp_path)
    return paths


def simulate_yearly(frame: pd.DataFrame, scores: np.ndarray, threshold: float, feature_set: str) -> pd.DataFrame:
    rows = []
    year_arr = frame["year"].to_numpy()
    for year in CFG.split.test_years:
        mask = year_arr == year
        chunk = frame.loc[mask].copy().reset_index(drop=True)
        score_chunk = scores[mask]
        orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"feature_set": feature_set, "year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(
            CFG.outputs_dir / f"scaled_{feature_set}_mophong_deals_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return pd.DataFrame(rows)


def run(
    n_iter: int,
    max_train_samples: int,
    pairplot_rows: int,
    shap_rows: int,
    force_dataset: bool,
    feature_sets: tuple[str, ...] = TARGET_FEATURE_SETS,
) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"scaled_selected_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    frame = build_cached_dataset(force=force_dataset).sort_values("dates").reset_index(drop=True)
    fs = named_feature_sets(frame)
    mlflow = _mlflow()
    parent = mlflow.start_run(run_name=f"scaled_selected_{run_id}") if mlflow else None
    summary_rows = []
    yearly_frames = []

    try:
        for idx, name in enumerate(feature_sets, start=1):
            if name not in fs:
                raise KeyError(f"Unknown feature set {name}. Available: {list(fs)}")
            seed = 7400 + idx
            raw_cols = fs[name]
            scaled_frame, cols, scale_meta = apply_wma_scaling(frame, raw_cols)
            print({"phase": "scaled_feature_set", "name": name, "features": len(cols), "scale_meta": scale_meta}, flush=True)

            x_train, y_train, _ = _candidate_dataset(scaled_frame, cols, CFG.split.train_years)
            x_valid, y_valid, _ = _candidate_dataset(scaled_frame, cols, CFG.split.valid_years)
            x_test, y_test, _ = _candidate_dataset(scaled_frame, cols, CFG.split.test_years)

            model, best_params, cv_auc, cv_results = _fit_search(x_train, y_train, n_iter, max_train_samples, seed)
            p_valid = model.predict_proba(x_valid)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            valid_metrics = _metrics(y_valid, p_valid)
            test_metrics = _metrics(y_test, p_test)

            scores = _score_full_frame(scaled_frame, cols, scaler=_IdentityScaler(), model=model)
            valid_curve = _proxy_threshold_table(scaled_frame, scores, CFG.split.valid_years)
            test_curve = _proxy_threshold_table(scaled_frame, scores, CFG.split.test_years)
            threshold = _select_threshold(valid_curve)
            proxy = _proxy_total(test_curve, threshold)
            yearly = simulate_yearly(scaled_frame, scores, threshold, name)
            yearly_frames.append(yearly)

            prefix = f"{name}_wmaz14p"
            cv_path = out_dir / f"{prefix}_cv_results.csv"
            valid_curve_path = out_dir / f"{prefix}_threshold_valid.csv"
            test_curve_path = out_dir / f"{prefix}_threshold_test.csv"
            yearly_path = out_dir / f"{prefix}_mophong_yearly.csv"
            model_path = CFG.model_dir / f"{prefix}_bundle.joblib"
            meta_path = out_dir / f"{prefix}_meta.json"
            pairplot_path = chart_dir / f"{prefix}_pairplot_top10.png"

            cv_results.to_csv(cv_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
            yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
            joblib.dump(
                {
                    "model": model,
                    "feature_columns": cols,
                    "raw_feature_columns": raw_cols,
                    "selected_threshold": threshold,
                    "best_params": best_params,
                    "scaling": scale_meta,
                },
                model_path,
                compress=3,
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_set": name,
                        "scale_meta": scale_meta,
                        "best_params": best_params,
                        "cv_auc": cv_auc,
                        "valid_metrics": valid_metrics,
                        "test_metrics": test_metrics,
                        "test_proxy": proxy,
                        "selected_threshold": threshold,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            chart_paths = []
            chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, f"{prefix}_valid")
            chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / f"{prefix}_threshold_valid.png", f"{prefix} validation threshold"))
            chart_paths.append(plot_threshold_curve(test_curve, chart_dir / f"{prefix}_threshold_test.png", f"{prefix} test proxy threshold"))
            chart_paths.append(plot_yearly_mophong(yearly, chart_dir / f"{prefix}_mophong_yearly.png"))

            train_mask = (
                scaled_frame["year"].isin(CFG.split.train_years)
                & (scaled_frame["buy_signal"] == 1)
                & scaled_frame["label"].notna()
                & scaled_frame[cols].notna().all(axis=1)
            )
            train_for_plot = scaled_frame.loc[train_mask, cols + ["label"]].copy()
            chart_paths.append(plot_pairplot_top10(train_for_plot, cols, model, pairplot_path, pairplot_rows, seed))
            fi_path = pairplot_path.with_name(pairplot_path.stem + "_feature_importance.csv")

            valid_plot_mask = (
                scaled_frame["year"].isin(CFG.split.valid_years)
                & (scaled_frame["buy_signal"] == 1)
                & scaled_frame["label"].notna()
                & scaled_frame[cols].notna().all(axis=1)
            )
            valid_for_shap = scaled_frame.loc[valid_plot_mask, cols].copy()
            shap_paths = plot_shap_diagnostics(
                model=model,
                x_frame=valid_for_shap,
                feature_cols=cols,
                out_dir=chart_dir,
                prefix=prefix,
                max_rows=shap_rows,
                seed=seed,
            )
            chart_paths += shap_paths

            summary_rows.append(
                {
                    "feature_set": name,
                    "scaling": "rolling_wma_zscore_14_past",
                    "n_features": len(cols),
                    "cv_auc": cv_auc,
                    "valid_auc": valid_metrics.get("auc", 0.0),
                    "test_auc": test_metrics.get("auc", 0.0),
                    "selected_threshold": threshold,
                    "proxy_test_resolved": proxy["resolved"],
                    "proxy_test_wr": proxy["winrate"],
                    "mophong_total_deals": int(yearly["deals"].sum()),
                    "mophong_total_resolved": int(yearly["resolved"].sum()),
                    "mophong_total_wr": round(yearly["wins"].sum() * 100 / max(yearly["resolved"].sum(), 1), 4),
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                }
            )

            if mlflow:
                with mlflow.start_run(run_name=prefix, nested=True):
                    mlflow.log_param("feature_set", name)
                    mlflow.log_param("scaling", "rolling_wma_zscore_14_past")
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
                    for _, row in yearly.iterrows():
                        mlflow.log_metric(f"mophong_{int(row['year'])}_winrate", float(row["winrate"]))
                        mlflow.log_metric(f"mophong_{int(row['year'])}_deals", float(row["deals"]))
                    for p in [cv_path, valid_curve_path, test_curve_path, yearly_path, meta_path, fi_path, model_path]:
                        mlflow.log_artifact(str(p))
                    for p in chart_paths:
                        artifact_path = "shap" if "shap_" in p.name else "charts"
                        mlflow.log_artifact(str(p), artifact_path=artifact_path)

        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / "scaled_fs03_fs07_fs10_summary.csv"
        yearly_all_path = out_dir / "scaled_fs03_fs07_fs10_mophong_yearly_all.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        yearly_all = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
        yearly_all.to_csv(yearly_all_path, index=False, encoding="utf-8-sig")

        if mlflow:
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(yearly_all_path))

        result = {
            "out_dir": str(out_dir),
            "summary": summary.to_dict(orient="records"),
            "yearly": yearly_all.to_dict(orient="records"),
            "elapsed_sec": round(time.time() - started, 2),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if mlflow and parent:
            mlflow.end_run()


class _IdentityScaler:
    def transform(self, x):
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--pairplot-rows", type=int, default=2500)
    parser.add_argument("--shap-rows", type=int, default=1500)
    parser.add_argument("--feature-sets", default=",".join(TARGET_FEATURE_SETS))
    parser.add_argument("--force-dataset", action="store_true")
    args = parser.parse_args()
    selected = tuple(item.strip() for item in args.feature_sets.split(",") if item.strip())
    run(args.n_iter, args.max_train_samples, args.pairplot_rows, args.shap_rows, args.force_dataset, selected)
