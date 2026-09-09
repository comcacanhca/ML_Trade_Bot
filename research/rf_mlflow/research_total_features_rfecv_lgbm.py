from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit

from config import CFG, ensure_dirs
from research_random50_initial_features import (
    META_COLS,
    _metrics,
    _mlflow,
    _plot_importance,
    _plot_shap,
    _sample_train,
    _select_threshold,
    _threshold_table,
    _total_at_threshold,
)
from research_rolling_zscore_50_h4h1 import (
    DEFAULT_CACHE_FAMILY,
    DROP_DEFAULT,
    PASSTHROUGH_FEATURES,
    PREDICTION_META_COLS,
    _cache_year_path,
    _extra_probability_metrics,
    _load_cache_columns,
    _make_diagnostic_artifacts,
    _rolling_zscore_frame,
)


FEATURE_FAMILY = "total_features_rfecv_lgbm"
TIME_FIXED_FEATURES = [
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "session_london_ny",
    "session_ny",
    "session_asia",
]


def _available_feature_columns(all_cols: list[str]) -> list[str]:
    return [c for c in all_cols if c not in META_COLS and c not in DROP_DEFAULT]


def _thin_ordered(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    idx = np.linspace(0, len(frame) - 1, num=max_rows, dtype=np.int64)
    return frame.iloc[idx].reset_index(drop=True)


def _load_scaled_years(
    years: tuple[int, ...],
    feature_columns: list[str],
    cache_family: str,
    max_rows: int = 0,
    ordered_thin: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    read_cols = list(dict.fromkeys(feature_columns + PREDICTION_META_COLS))
    frames = []
    per_year = int(np.ceil(max_rows / max(1, len(years)))) if max_rows else 0
    for year in years:
        frame = pd.read_parquet(_cache_year_path(cache_family, year), columns=read_cols)
        frame = _rolling_zscore_frame(frame, feature_columns)
        if per_year:
            if ordered_thin:
                frame = _thin_ordered(frame, per_year)
            else:
                frame = _sample_train(frame, per_year, seed + int(year))
        frames.append(frame)
        gc.collect()
    out = pd.concat(frames, ignore_index=True, copy=False)
    if max_rows:
        out = _thin_ordered(out, max_rows) if ordered_thin else _sample_train(out, max_rows, seed)
    return out


def _lgbm_classifier(seed: int, n_estimators: int, n_jobs: int, learning_rate: float, num_leaves: int, max_depth: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.00,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _predict_proba_batched(model: lgb.LGBMClassifier, x: np.ndarray, batch_size: int = 50000) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        out[start:stop] = model.predict_proba(x[start:stop])[:, 1].astype("float32", copy=False)
    return out


def _constant_filter(train: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows = []
    kept = []
    for col in feature_columns:
        x = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nunique = int(x.nunique(dropna=False))
        std = float(x.std())
        keep = nunique > 1 and std > 1e-12
        rows.append({"feature": col, "nunique": nunique, "std": std, "status": "keep" if keep else "constant"})
        if keep:
            kept.append(col)
    return kept, pd.DataFrame(rows)


def _label_corr_scores(train: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    y = train["label"].to_numpy("float32", copy=False)
    y_std = float(np.std(y))
    rows = []
    for col in feature_columns:
        x = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float32", copy=False)
        x_std = float(np.std(x))
        corr = 0.0
        if x_std > 1e-12 and y_std > 1e-12:
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append({"feature": col, "label_corr": corr, "abs_label_corr": abs(corr)})
    return pd.DataFrame(rows)


def _correlation_prune(
    train: pd.DataFrame,
    feature_columns: list[str],
    protected: set[str],
    threshold: float,
    max_corr_rows: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    if len(feature_columns) <= 1:
        return feature_columns, pd.DataFrame()
    data = train[feature_columns + ["label"]]
    if max_corr_rows and len(data) > max_corr_rows:
        data = _sample_train(data, max_corr_rows, seed)
    scores = _label_corr_scores(data, feature_columns).set_index("feature")["abs_label_corr"].to_dict()
    ordered = sorted(feature_columns, key=lambda c: (c not in protected, -scores.get(c, 0.0), c))
    corr = data[ordered].corr(method="pearson").abs().fillna(0.0)
    dropped: set[str] = set()
    rows = []
    for i, a in enumerate(ordered):
        if a in dropped:
            continue
        for b in ordered[i + 1 :]:
            if b in dropped:
                continue
            value = float(corr.loc[a, b])
            if value < threshold:
                continue
            if a in protected and b not in protected:
                keep, drop = a, b
            elif b in protected and a not in protected:
                keep, drop = b, a
            elif scores.get(a, 0.0) >= scores.get(b, 0.0):
                keep, drop = a, b
            else:
                keep, drop = b, a
            dropped.add(drop)
            rows.append({"keep": keep, "drop": drop, "abs_corr": value, "drop_score": scores.get(drop, 0.0), "keep_score": scores.get(keep, 0.0)})
            if drop == a:
                break
    kept = [c for c in feature_columns if c not in dropped]
    return kept, pd.DataFrame(rows)


def _preselect_features(
    train: pd.DataFrame,
    feature_columns: list[str],
    top_n: int,
    corr_threshold: float,
    max_corr_rows: int,
    mutual_info_rows: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    protected: list[str],
    out_dir: Path,
) -> tuple[list[str], list[Path]]:
    paths: list[Path] = []
    protected_set = {c for c in protected if c in feature_columns}

    kept_constant, constant_report = _constant_filter(train, feature_columns)
    constant_path = out_dir / "preselect_constant_filter.csv"
    constant_report.to_csv(constant_path, index=False, encoding="utf-8-sig")
    paths.append(constant_path)

    kept_corr, corr_report = _correlation_prune(train, kept_constant, protected_set, corr_threshold, max_corr_rows, seed)
    corr_path = out_dir / "preselect_correlation_dropped.csv"
    corr_report.to_csv(corr_path, index=False, encoding="utf-8-sig")
    paths.append(corr_path)

    x = train[kept_corr].to_numpy("float32", copy=True)
    y = train["label"].to_numpy("int8", copy=True)
    pre_model = _lgbm_classifier(seed, n_estimators, n_jobs, learning_rate, num_leaves, max_depth)
    pre_model.fit(x, y)
    lgb_importance = pd.Series(pre_model.feature_importances_, index=kept_corr, dtype="float64")

    label_corr = _label_corr_scores(train, kept_corr).set_index("feature")

    mi_sample = train[kept_corr + ["label"]]
    if mutual_info_rows and len(mi_sample) > mutual_info_rows:
        mi_sample = _sample_train(mi_sample, mutual_info_rows, seed + 991)
    mi_values = mutual_info_classif(
        mi_sample[kept_corr].to_numpy("float32", copy=True),
        mi_sample["label"].to_numpy("int8", copy=True),
        discrete_features=False,
        random_state=seed,
    )
    mutual_info = pd.Series(mi_values, index=kept_corr, dtype="float64")

    score = pd.DataFrame(
        {
            "feature": kept_corr,
            "lgb_importance": [float(lgb_importance.get(c, 0.0)) for c in kept_corr],
            "abs_label_corr": [float(label_corr.loc[c, "abs_label_corr"]) if c in label_corr.index else 0.0 for c in kept_corr],
            "mutual_info": [float(mutual_info.get(c, 0.0)) for c in kept_corr],
            "protected": [c in protected_set for c in kept_corr],
        }
    )
    for metric in ["lgb_importance", "abs_label_corr", "mutual_info"]:
        score[f"{metric}_rank"] = score[metric].rank(method="average", ascending=False)
    score["combined_rank"] = score[["lgb_importance_rank", "abs_label_corr_rank", "mutual_info_rank"]].mean(axis=1)
    score = score.sort_values(["protected", "combined_rank", "feature"], ascending=[False, True, True])
    score_path = out_dir / "preselect_score.csv"
    score.to_csv(score_path, index=False, encoding="utf-8-sig")
    paths.append(score_path)

    if top_n and top_n > 0:
        selected = score.head(max(top_n, len(protected_set)))["feature"].tolist()
    else:
        selected = score["feature"].tolist()
    selected = list(dict.fromkeys([*protected, *selected]))
    selected = [c for c in selected if c in kept_corr]

    selected_path = out_dir / "preselected_features.json"
    selected_path.write_text(
        json.dumps(
            {
                "preselected_features": selected,
                "preselected_count": len(selected),
                "input_features": len(feature_columns),
                "after_constant_filter": len(kept_constant),
                "after_correlation_prune": len(kept_corr),
                "top_n": top_n,
                "protected_features": [c for c in protected if c in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.append(selected_path)
    return selected, paths


def _save_rfecv_artifacts(selector: RFECV, feature_columns: list[str], out_dir: Path) -> tuple[list[str], list[Path]]:
    paths: list[Path] = []
    support = np.asarray(selector.support_, dtype=bool)
    ranking = np.asarray(selector.ranking_, dtype=np.int32)
    selected = [c for c, keep in zip(feature_columns, support) if keep]
    support_df = pd.DataFrame({"feature": feature_columns, "selected": support, "ranking": ranking}).sort_values(
        ["ranking", "feature"], ascending=[True, True]
    )
    support_path = out_dir / "rfecv_feature_ranking.csv"
    support_df.to_csv(support_path, index=False, encoding="utf-8-sig")
    paths.append(support_path)

    selected_path = out_dir / "selected_features.json"
    selected_path.write_text(json.dumps({"selected_features": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(selected_path)

    cv_results = getattr(selector, "cv_results_", None)
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_path = out_dir / "rfecv_cv_results.csv"
        cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
        paths.append(cv_path)
    return selected, paths


def run(
    run_id: int | None,
    cache_family: str,
    force_cache: bool,
    max_train_rows: int,
    rfecv_rows: int,
    preselect_top_n: int,
    preselect_corr_threshold: float,
    preselect_corr_rows: int,
    preselect_mutual_info_rows: int,
    preselect_n_estimators: int,
    min_features_to_select: int,
    rfecv_step: int,
    cv_splits: int,
    seed: int,
    n_estimators_rfecv: int,
    n_estimators_final: int,
    n_jobs: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    protect_time_features: bool,
    protect_passthrough: bool,
    enable_mlflow: bool,
    save_predictions: bool,
    log_diagnostics: bool,
    shap_rows: int,
) -> dict:
    ensure_dirs()
    run_id = int(run_id or time.time())
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    model_dir = out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_cols = _load_cache_columns(cache_family, force_cache)
    feature_columns = _available_feature_columns(all_cols)
    if not feature_columns:
        raise ValueError("No usable feature columns.")

    config = {
        "run_id": run_id,
        "feature_family": FEATURE_FAMILY,
        "cache_family": cache_family,
        "total_features": len(feature_columns),
        "scaling": "rolling_zscore",
        "rfecv_rows": rfecv_rows,
        "max_train_rows": max_train_rows,
        "preselect_top_n": preselect_top_n,
        "preselect_corr_threshold": preselect_corr_threshold,
        "preselect_corr_rows": preselect_corr_rows,
        "preselect_mutual_info_rows": preselect_mutual_info_rows,
        "preselect_n_estimators": preselect_n_estimators,
        "min_features_to_select": min_features_to_select,
        "rfecv_step": rfecv_step,
        "cv_splits": cv_splits,
        "seed": seed,
        "model": "LightGBM",
        "n_estimators_rfecv": n_estimators_rfecv,
        "n_estimators_final": n_estimators_final,
        "protect_time_features": protect_time_features,
        "time_fixed_features": [c for c in TIME_FIXED_FEATURES if c in feature_columns],
        "protect_passthrough": protect_passthrough,
    }
    config_path = out_dir / "run_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    protected = []
    if protect_time_features:
        protected.extend(c for c in TIME_FIXED_FEATURES if c in feature_columns)
    if protect_passthrough:
        protected.extend(c for c in PASSTHROUGH_FEATURES if c in feature_columns)
    protected = list(dict.fromkeys(protected))

    print({"phase": "load_rfecv_train", "features": len(feature_columns), "rows": rfecv_rows}, flush=True)
    rfecv_train = _load_scaled_years(
        CFG.split.train_years,
        feature_columns,
        cache_family,
        max_rows=rfecv_rows,
        ordered_thin=True,
        seed=seed,
    )

    print({"phase": "preselect_features", "input_features": len(feature_columns), "top_n": preselect_top_n}, flush=True)
    preselected_features, artifact_paths = _preselect_features(
        train=rfecv_train,
        feature_columns=feature_columns,
        top_n=preselect_top_n,
        corr_threshold=preselect_corr_threshold,
        max_corr_rows=preselect_corr_rows,
        mutual_info_rows=preselect_mutual_info_rows,
        seed=seed,
        n_estimators=preselect_n_estimators,
        n_jobs=n_jobs,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        protected=protected,
        out_dir=out_dir,
    )
    print({"phase": "preselect_done", "selected_features": len(preselected_features)}, flush=True)

    x_rfecv = rfecv_train[preselected_features].to_numpy("float32", copy=True)
    y_rfecv = rfecv_train["label"].to_numpy("int8", copy=True)

    estimator = _lgbm_classifier(seed, n_estimators_rfecv, n_jobs, learning_rate, num_leaves, max_depth)
    cv = TimeSeriesSplit(n_splits=cv_splits)
    selector = RFECV(
        estimator=estimator,
        step=rfecv_step,
        min_features_to_select=min_features_to_select,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        importance_getter="feature_importances_",
        verbose=1,
    )
    print({"phase": "fit_rfecv", "rows": len(rfecv_train), "features": len(preselected_features)}, flush=True)
    selector.fit(x_rfecv, y_rfecv)
    selected_features, rfecv_paths = _save_rfecv_artifacts(selector, preselected_features, out_dir)
    artifact_paths.extend(rfecv_paths)

    if protected:
        selected_features = list(dict.fromkeys([*protected, *selected_features]))
        selected_path = out_dir / "selected_features_with_protected.json"
        selected_path.write_text(
            json.dumps(
                {
                    "selected_features": selected_features,
                    "protected_features": protected,
                    "protected_time_features": [c for c in TIME_FIXED_FEATURES if c in protected],
                    "protected_passthrough_features": [c for c in PASSTHROUGH_FEATURES if c in protected],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_paths.append(selected_path)

    print({"phase": "load_final_train_valid", "selected_features": len(selected_features)}, flush=True)
    train = _load_scaled_years(CFG.split.train_years, selected_features, cache_family, max_rows=max_train_rows, seed=seed)
    valid = _load_scaled_years(CFG.split.valid_years, selected_features, cache_family)
    x_train = train[selected_features].to_numpy("float32", copy=True)
    y_train = train["label"].to_numpy("int8", copy=True)
    x_valid = valid[selected_features].to_numpy("float32", copy=True)
    y_valid = valid["label"].to_numpy("int8", copy=True)

    model = _lgbm_classifier(seed, n_estimators_final, n_jobs, learning_rate, num_leaves, max_depth)
    print({"phase": "fit_final_lgbm", "rows": len(train), "features": len(selected_features)}, flush=True)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)],
    )

    p_valid = _predict_proba_batched(model, x_valid)
    valid_metrics = _metrics(y_valid, p_valid)
    valid_metrics.update(_extra_probability_metrics(y_valid, p_valid))
    valid_curve = _threshold_table(valid[["label", "year", "session_london_ny"]], p_valid, CFG.threshold_grid)
    threshold = _select_threshold(valid_curve)

    test_frames = []
    test_probs = []
    test_labels = []
    test_curves = []
    for year in CFG.split.test_years:
        print({"phase": "predict_test_year", "year": year}, flush=True)
        test_year = _load_scaled_years((year,), selected_features, cache_family)
        x_test_year = test_year[selected_features].to_numpy("float32", copy=True)
        p_test_year = _predict_proba_batched(model, x_test_year)
        test_frames.append(test_year[[c for c in PREDICTION_META_COLS if c in test_year.columns]].copy())
        test_probs.append(p_test_year)
        test_labels.append(test_year["label"].to_numpy("int8", copy=True))
        test_curves.append(_threshold_table(test_year[["label", "year", "session_london_ny"]], p_test_year, CFG.threshold_grid))
        del test_year, x_test_year
        gc.collect()

    p_test = np.concatenate(test_probs).astype("float32", copy=False)
    y_test = np.concatenate(test_labels).astype("int8", copy=False)
    test_meta = pd.concat(test_frames, ignore_index=True, copy=False)
    test_meta["prob"] = p_test
    test_curve = pd.concat(test_curves, ignore_index=True, copy=False)
    test_metrics = _metrics(y_test, p_test)
    test_metrics.update(_extra_probability_metrics(y_test, p_test))
    test_total = _total_at_threshold(test_curve, threshold)

    importance = pd.DataFrame({"feature": selected_features, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    importance_path = model_dir / "lgbm_feature_importance.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    artifact_paths.append(importance_path)
    artifact_paths.append(_plot_importance(importance, model_dir / "lgbm_feature_importance_top40.png", "LightGBM feature importance"))

    valid_curve_path = model_dir / "threshold_valid.csv"
    test_curve_path = model_dir / "threshold_test.csv"
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
    artifact_paths.extend([valid_curve_path, test_curve_path])

    artifact_paths.extend(
        _make_diagnostic_artifacts(
            model_dir=model_dir,
            name="total_features_rfecv_lgbm",
            valid_all=valid,
            p_valid=p_valid,
            y_valid=y_valid,
            test_pred=test_meta,
            p_test=p_test,
            y_test=y_test,
            threshold=threshold,
            valid_curve=valid_curve,
            test_curve=test_curve,
            save_predictions=save_predictions,
            log_diagnostics=log_diagnostics,
        )
    )
    artifact_paths.extend(_plot_shap(model, x_valid, selected_features, model_dir, "total_features_rfecv_lgbm", shap_rows, seed))

    bundle_path = model_dir / "total_features_rfecv_lgbm_bundle.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": selected_features,
            "feature_family": FEATURE_FAMILY,
            "selected_threshold": threshold,
            "cache_family": cache_family,
            "scaling": "rolling_zscore",
        },
        bundle_path,
        compress=3,
    )
    artifact_paths.append(bundle_path)

    summary = {
        **config,
        "preselected_features": len(preselected_features),
        "selected_features": len(selected_features),
        "threshold": float(threshold),
        "valid_auc": float(valid_metrics["auc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_total_wr": float(test_total["winrate"]),
        "test_total_resolved": int(test_total["resolved"]),
        "test_min_year_deals": int(test_total["min_year_deals"]),
        "top20_features": importance.head(20)["feature"].tolist(),
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"valid": valid_metrics, "test": test_metrics, "test_total": test_total}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths.extend([summary_path, metrics_path])

    mlflow = _mlflow() if enable_mlflow else None
    if mlflow:
        with mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}"):
            mlflow.log_params(config)
            mlflow.log_param("preselected_features", len(preselected_features))
            mlflow.log_param("selected_features", len(selected_features))
            mlflow.log_param("threshold", float(threshold))
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", float(v))
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v))
            mlflow.log_metric("test_total_wr", float(test_total["winrate"]))
            mlflow.log_metric("test_total_resolved", int(test_total["resolved"]))
            mlflow.log_metric("test_min_year_deals", int(test_total["min_year_deals"]))
            for path in artifact_paths:
                if path and Path(path).exists():
                    mlflow.log_artifact(str(path))

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--cache-family", type=str, default=DEFAULT_CACHE_FAMILY)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--rfecv-rows", type=int, default=50000)
    parser.add_argument("--preselect-top-n", type=int, default=300)
    parser.add_argument("--preselect-corr-threshold", type=float, default=0.98)
    parser.add_argument("--preselect-corr-rows", type=int, default=30000)
    parser.add_argument("--preselect-mutual-info-rows", type=int, default=20000)
    parser.add_argument("--preselect-n-estimators", type=int, default=250)
    parser.add_argument("--min-features-to-select", type=int, default=60)
    parser.add_argument("--rfecv-step", type=int, default=50)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=860906)
    parser.add_argument("--n-estimators-rfecv", type=int, default=250)
    parser.add_argument("--n-estimators-final", type=int, default=1200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--no-protect-time-features", action="store_true")
    parser.add_argument("--protect-passthrough", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--log-diagnostics", action="store_true")
    parser.add_argument("--shap-rows", type=int, default=0)
    args = parser.parse_args()
    run(
        run_id=args.run_id or None,
        cache_family=args.cache_family,
        force_cache=args.force_cache,
        max_train_rows=args.max_train_rows,
        rfecv_rows=args.rfecv_rows,
        preselect_top_n=args.preselect_top_n,
        preselect_corr_threshold=args.preselect_corr_threshold,
        preselect_corr_rows=args.preselect_corr_rows,
        preselect_mutual_info_rows=args.preselect_mutual_info_rows,
        preselect_n_estimators=args.preselect_n_estimators,
        min_features_to_select=args.min_features_to_select,
        rfecv_step=args.rfecv_step,
        cv_splits=args.cv_splits,
        seed=args.seed,
        n_estimators_rfecv=args.n_estimators_rfecv,
        n_estimators_final=args.n_estimators_final,
        n_jobs=args.n_jobs,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        protect_time_features=not args.no_protect_time_features,
        protect_passthrough=args.protect_passthrough,
        enable_mlflow=args.enable_mlflow,
        save_predictions=args.save_predictions,
        log_diagnostics=args.log_diagnostics,
        shap_rows=args.shap_rows,
    )


if __name__ == "__main__":
    main()
