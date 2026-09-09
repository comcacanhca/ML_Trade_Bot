from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import CFG, ensure_dirs
from research_random50_initial_features import META_COLS, _mlflow, _sample_train, _year_path, build_cache
from research_rolling_zscore_50_h4h1 import (
    DEFAULT_CACHE_FAMILY,
    DROP_DEFAULT,
    PASSTHROUGH_FEATURES,
    PREDICTION_META_COLS,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    _rolling_zscore_frame,
)


FEATURE_SELECTION_FAMILY = "feature_selection_rolling_zscore"
PROTECTED_FEATURES = [
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "session_ny",
    "sig_bb_reversion_buy",
    #"sig_pullback_trend_buy",
]


def _available_feature_columns(all_cols: list[str]) -> list[str]:
    return [c for c in all_cols if c not in META_COLS and c not in DROP_DEFAULT]


def _cache_year_path(cache_family: str, year: int) -> Path:
    return CFG.cache_dir / cache_family / f"candidates_{year}.parquet"


def _load_cache_columns(cache_family: str, force_cache: bool) -> list[str]:
    if cache_family == DEFAULT_CACHE_FAMILY:
        return build_cache(force_cache)
    cols_path = CFG.cache_dir / cache_family / "feature_columns.json"
    if not cols_path.exists():
        raise FileNotFoundError(f"Missing {cols_path}. Build this cache family first.")
    return json.loads(cols_path.read_text(encoding="utf-8"))


def _load_scaled_train(feature_columns: list[str], max_train_rows: int, seed: int, cache_family: str) -> pd.DataFrame:
    read_cols = list(dict.fromkeys(feature_columns + PREDICTION_META_COLS))
    per_train_year = max(1000, int(np.ceil(max_train_rows / max(1, len(CFG.split.train_years)))))
    frames = []
    for year in CFG.split.train_years:
        frame = pd.read_parquet(_cache_year_path(cache_family, year), columns=read_cols)
        frame = _rolling_zscore_frame(frame, feature_columns)
        frame = _sample_train(frame, per_train_year, seed + int(year))
        frames.append(frame)
        gc.collect()
    train = pd.concat(frames, ignore_index=True, copy=False)
    return _sample_train(train, max_train_rows, seed)


def _remove_constant_features(train: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows = []
    kept = []
    for col in feature_columns:
        values = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nunique = int(values.nunique(dropna=False))
        std = float(values.std())
        status = "keep" if nunique > 1 and std > 1e-12 else "constant"
        rows.append({"feature": col, "nunique": nunique, "std": std, "status": status})
        if status == "keep":
            kept.append(col)
    return kept, pd.DataFrame(rows)


def _label_corr_scores(train: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    y = train["label"].to_numpy("float32")
    y_std = float(np.std(y))
    rows = []
    for col in feature_columns:
        x = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float32")
        x_std = float(np.std(x))
        corr = 0.0
        if x_std > 1e-12 and y_std > 1e-12:
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append({"feature": col, "abs_label_corr": abs(corr), "label_corr": corr})
    return pd.DataFrame(rows).sort_values("abs_label_corr", ascending=False)


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
    rng = np.random.default_rng(seed)
    data = train[feature_columns + ["label"]].copy()
    if len(data) > max_corr_rows:
        data = data.iloc[rng.choice(np.arange(len(data)), size=max_corr_rows, replace=False)].reset_index(drop=True)
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
                drop = b
                keep = a
            elif b in protected and a not in protected:
                drop = a
                keep = b
            elif scores.get(a, 0.0) >= scores.get(b, 0.0):
                drop = b
                keep = a
            else:
                drop = a
                keep = b
            dropped.add(drop)
            rows.append(
                {
                    "kept_feature": keep,
                    "dropped_feature": drop,
                    "abs_corr": value,
                    "kept_abs_label_corr": scores.get(keep, 0.0),
                    "dropped_abs_label_corr": scores.get(drop, 0.0),
                }
            )
            if drop == a:
                break
    kept = [c for c in feature_columns if c not in dropped]
    return kept, pd.DataFrame(rows)


def _l1_select(train: pd.DataFrame, feature_columns: list[str], protected: set[str], c_value: float, seed: int) -> tuple[list[str], pd.DataFrame]:
    x = train[feature_columns].to_numpy("float32", copy=True)
    y = train["label"].to_numpy("int8", copy=True)
    model = LogisticRegression(
        penalty="l1",
        C=c_value,
        solver="saga",
        class_weight="balanced",
        max_iter=2500,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, y)
    coef = model.coef_[0]
    table = pd.DataFrame({"feature": feature_columns, "l1_coef": coef, "abs_l1_coef": np.abs(coef)})
    selected = table.loc[table["abs_l1_coef"] > 1e-9, "feature"].tolist()
    selected = list(dict.fromkeys([c for c in feature_columns if c in protected] + selected))
    return selected, table.sort_values("abs_l1_coef", ascending=False)


def _rf_importance_select(
    train: pd.DataFrame,
    feature_columns: list[str],
    protected: set[str],
    top_n: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    x = train[feature_columns].to_numpy("float32", copy=True)
    y = train["label"].to_numpy("int8", copy=True)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=500,
        min_samples_split=900,
        max_features="sqrt",
        max_samples=0.75,
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, y)
    table = pd.DataFrame({"feature": feature_columns, "rf_importance": model.feature_importances_}).sort_values("rf_importance", ascending=False)
    selected = list(dict.fromkeys([c for c in feature_columns if c in protected] + table.head(top_n)["feature"].tolist()))
    return selected, table


def _featurewiz_select(
    train: pd.DataFrame,
    feature_columns: list[str],
    protected: set[str],
    corr_limit: float,
    max_rows: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    try:
        from featurewiz import featurewiz
    except Exception as exc:
        raise RuntimeError(
            "featurewiz is not installed. Install it with: python -m pip install featurewiz"
        ) from exc

    rng = np.random.default_rng(seed)
    data = train[feature_columns + ["label"]].copy()
    if len(data) > max_rows:
        idx = rng.choice(np.arange(len(data)), size=max_rows, replace=False)
        data = data.iloc[idx].reset_index(drop=True)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        result = featurewiz(
            dataname=data,
            target="label",
            corr_limit=corr_limit,
            verbose=0,
            sep=",",
            header=0,
            test_data="",
            feature_engg="",
            category_encoders="",
            dask_xgboost_flag=False,
            nrows=None,
        )
    except TypeError:
        result = featurewiz(data, "label", corr_limit=corr_limit, verbose=0)

    if isinstance(result, tuple):
        selected_raw = result[0]
    else:
        selected_raw = result
    if isinstance(selected_raw, pd.DataFrame):
        selected = [c for c in selected_raw.columns if c != "label"]
    else:
        selected = [str(c) for c in list(selected_raw)]
    selected = [c for c in selected if c in feature_columns]
    selected = list(dict.fromkeys([c for c in feature_columns if c in protected] + selected))
    table = pd.DataFrame(
        {
            "feature": feature_columns,
            "featurewiz_selected": [c in selected for c in feature_columns],
            "featurewiz_rank": [selected.index(c) + 1 if c in selected else None for c in feature_columns],
        }
    ).sort_values(["featurewiz_selected", "featurewiz_rank", "feature"], ascending=[False, True, True])
    return selected, table


def _plot_bar(table: pd.DataFrame, value_col: str, path: Path, title: str, top_n: int = 40) -> Path:
    data = table.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(5, len(data) * 0.24)))
    plt.barh(data["feature"], data[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_featurewiz_selected(table: pd.DataFrame, path: Path, title: str, top_n: int = 60) -> Path | None:
    if table.empty or "featurewiz_selected" not in table.columns:
        return None
    data = table.loc[table["featurewiz_selected"] == True].copy()
    if data.empty:
        return None
    data["rank_score"] = top_n + 1 - pd.to_numeric(data["featurewiz_rank"], errors="coerce").fillna(top_n + 1)
    data = data.sort_values("featurewiz_rank").head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(5, len(data) * 0.24)))
    plt.barh(data["feature"], data["rank_score"])
    plt.title(title)
    plt.xlabel("FeatureWiz rank score")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def run(
    max_train_rows: int,
    seed: int,
    force_cache: bool,
    run_id: int | None,
    corr_threshold: float,
    max_corr_rows: int,
    l1_c: float,
    rf_top_n: int,
    final_top_n: int,
    method: str,
    featurewiz_corr_limit: float,
    featurewiz_rows: int,
    cache_family: str,
    enable_mlflow: bool,
) -> dict:
    ensure_dirs()
    run_id = int(run_id or time.time())
    out_dir = CFG.outputs_dir / f"{FEATURE_SELECTION_FAMILY}_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    all_cols = _load_cache_columns(cache_family, force_cache)
    candidates = _available_feature_columns(all_cols)
    protected = {c for c in PROTECTED_FEATURES if c in candidates}
    train = _load_scaled_train(candidates, max_train_rows, seed, cache_family)

    kept_non_constant, constant_table = _remove_constant_features(train, candidates)
    corr_kept, corr_drop_table = _correlation_prune(train, kept_non_constant, protected, corr_threshold, max_corr_rows, seed)
    label_corr_table = _label_corr_scores(train, corr_kept)
    l1_selected: list[str] = []
    rf_selected: list[str] = []
    featurewiz_selected: list[str] = []
    l1_table = pd.DataFrame({"feature": corr_kept, "l1_coef": 0.0, "abs_l1_coef": 0.0})
    rf_table = pd.DataFrame({"feature": corr_kept, "rf_importance": 0.0})
    featurewiz_table = pd.DataFrame({"feature": corr_kept, "featurewiz_selected": False, "featurewiz_rank": None})

    if method in {"hybrid", "featurewiz_hybrid"}:
        l1_selected, l1_table = _l1_select(train, corr_kept, protected, l1_c, seed)
        rf_selected, rf_table = _rf_importance_select(train, corr_kept, protected, rf_top_n, seed)
    if method in {"featurewiz", "featurewiz_hybrid"}:
        featurewiz_selected, featurewiz_table = _featurewiz_select(train, corr_kept, protected, featurewiz_corr_limit, featurewiz_rows, seed)

    l1_rank = {f: i for i, f in enumerate(l1_table["feature"].tolist(), start=1)}
    rf_rank = {f: i for i, f in enumerate(rf_table["feature"].tolist(), start=1)}
    featurewiz_rank = {f: i for i, f in enumerate(featurewiz_table.loc[featurewiz_table["featurewiz_selected"], "feature"].tolist(), start=1)}
    corr_rank = {f: i for i, f in enumerate(label_corr_table["feature"].tolist(), start=1)}
    union = list(dict.fromkeys([*PROTECTED_FEATURES, *featurewiz_selected, *l1_selected, *rf_selected]))
    union = [f for f in union if f in corr_kept]
    score_rows = []
    for feature in union:
        score = 0.0
        if feature in protected:
            score += 1000.0
        score += max(0.0, 300.0 - featurewiz_rank.get(feature, 9999))
        score += max(0.0, 200.0 - l1_rank.get(feature, 9999))
        score += max(0.0, 200.0 - rf_rank.get(feature, 9999))
        score += max(0.0, 100.0 - corr_rank.get(feature, 9999))
        score_rows.append(
            {
                "feature": feature,
                "selection_score": score,
                "protected": feature in protected,
                "featurewiz_rank": featurewiz_rank.get(feature),
                "l1_rank": l1_rank.get(feature),
                "rf_rank": rf_rank.get(feature),
                "label_corr_rank": corr_rank.get(feature),
            }
        )
    selection_table = pd.DataFrame(score_rows).sort_values(["protected", "selection_score", "feature"], ascending=[False, False, True])
    if method == "featurewiz":
        selected_features = list(dict.fromkeys([f for f in PROTECTED_FEATURES if f in corr_kept] + featurewiz_selected))
        selected_features = selected_features[: max(final_top_n, len(protected))]
    else:
        selected_features = selection_table.head(max(final_top_n, len(protected)))["feature"].tolist()
    selected_features = list(dict.fromkeys([f for f in PROTECTED_FEATURES if f in selected_features] + selected_features))

    paths = []
    selected_path = out_dir / "selected_features.json"
    selected_path.write_text(json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(selected_path)
    for table, name in [
        (constant_table, "constant_filter.csv"),
        (corr_drop_table, "correlation_dropped.csv"),
        (featurewiz_table, "featurewiz_selected.csv"),
        (l1_table, "l1_coef.csv"),
        (rf_table, "rf_importance.csv"),
        (label_corr_table, "label_correlation.csv"),
        (selection_table, "selection_score.csv"),
    ]:
        path = out_dir / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)

    fw_chart = _plot_featurewiz_selected(featurewiz_table, chart_dir / "featurewiz_selected_top60.png", "FeatureWiz selected features")
    if fw_chart is not None:
        paths.append(fw_chart)
    if method in {"hybrid", "featurewiz_hybrid"}:
        paths.append(_plot_bar(l1_table, "abs_l1_coef", chart_dir / "l1_coef_top40.png", "L1 Logistic absolute coefficients"))
        paths.append(_plot_bar(rf_table, "rf_importance", chart_dir / "rf_importance_top40.png", "RF feature importance selection model"))
    paths.append(_plot_bar(selection_table, "selection_score", chart_dir / "selection_score_top40.png", "Final feature selection score"))

    report = {
        "run_id": run_id,
        "family": FEATURE_SELECTION_FAMILY,
        "scaling": {
            "name": "rolling_zscore",
            "window": ROLLING_WINDOW,
            "min_periods": ROLLING_MIN_PERIODS,
            "use_past_only": True,
            "passthrough_features": sorted(PASSTHROUGH_FEATURES),
        },
        "split": {
            "train_years": CFG.split.train_years,
            "valid_years_not_used": CFG.split.valid_years,
            "test_years_not_used": CFG.split.test_years,
        },
        "params": {
            "max_train_rows": max_train_rows,
            "seed": seed,
            "corr_threshold": corr_threshold,
            "max_corr_rows": max_corr_rows,
            "l1_c": l1_c,
            "rf_top_n": rf_top_n,
            "final_top_n": final_top_n,
            "method": method,
            "featurewiz_corr_limit": featurewiz_corr_limit,
            "featurewiz_rows": featurewiz_rows,
            "cache_family": cache_family,
        },
        "counts": {
            "candidate_features": len(candidates),
            "non_constant_features": len(kept_non_constant),
            "corr_kept_features": len(corr_kept),
            "featurewiz_selected_features": len(featurewiz_selected),
            "l1_selected_features": len(l1_selected),
            "rf_selected_features": len(rf_selected),
            "final_selected_features": len(selected_features),
            "train_rows": int(len(train)),
        },
        "selected_features_path": str(selected_path),
        "selected_features": selected_features,
    }
    report_path = out_dir / "feature_selection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(report_path)

    mlflow = _mlflow() if enable_mlflow else None
    if mlflow:
        with mlflow.start_run(run_name=f"{FEATURE_SELECTION_FAMILY}_{run_id}"):
            mlflow.log_params(report["params"])
            for key, value in report["counts"].items():
                mlflow.log_metric(key, value)
            for path in paths:
                mlflow.log_artifact(str(path), artifact_path="feature_selection")

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=860906)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--max-corr-rows", type=int, default=50000)
    parser.add_argument("--l1-c", type=float, default=0.03)
    parser.add_argument("--rf-top-n", type=int, default=60)
    parser.add_argument("--final-top-n", type=int, default=45)
    parser.add_argument("--method", choices=["hybrid", "featurewiz", "featurewiz_hybrid"], default="featurewiz")
    parser.add_argument("--featurewiz-corr-limit", type=float, default=0.95)
    parser.add_argument("--featurewiz-rows", type=int, default=50000)
    parser.add_argument("--cache-family", type=str, default=DEFAULT_CACHE_FAMILY)
    parser.add_argument("--enable-mlflow", action="store_true")
    args = parser.parse_args()
    run(
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        force_cache=args.force_cache,
        run_id=args.run_id,
        corr_threshold=args.corr_threshold,
        max_corr_rows=args.max_corr_rows,
        l1_c=args.l1_c,
        rf_top_n=args.rf_top_n,
        final_top_n=args.final_top_n,
        method=args.method,
        featurewiz_corr_limit=args.featurewiz_corr_limit,
        featurewiz_rows=args.featurewiz_rows,
        cache_family=args.cache_family,
        enable_mlflow=args.enable_mlflow,
    )


if __name__ == "__main__":
    main()
