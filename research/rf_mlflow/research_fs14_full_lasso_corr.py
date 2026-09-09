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
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from config import CFG, PROJECT_ROOT, ensure_dirs
from mophong_adapter import orders_from_scores, simulate_orders
from plots import plot_model_diagnostics, plot_threshold_curve, plot_yearly_mophong
from research_fs14_overlay_norm_light import FS14_NAME, RobustClipScalerLite, _build_year_candidates, _psi


FEATURE_FAMILY = "fs14_full_lasso_corr"
FORCE_KEEP_TOKENS = ("tod_", "dow_", "sig_", "buy_signal", "close_diff_lag_", "body_lag_")


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _candidate_dir() -> Path:
    return CFG.cache_dir / FEATURE_FAMILY / "candidates"


def _year_path(year: int) -> Path:
    return _candidate_dir() / f"candidates_{year}.parquet"


def _write_feature_cols(cols: list[str]) -> None:
    path = _candidate_dir() / "feature_columns.json"
    path.write_text(json.dumps(cols, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_feature_cols() -> list[str]:
    path = _candidate_dir() / "feature_columns.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_full_candidate_cache(force: bool = False) -> list[str]:
    _candidate_dir().mkdir(parents=True, exist_ok=True)
    years = CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years
    feature_cols: list[str] | None = None
    for year in years:
        path = _year_path(year)
        if path.exists() and not force:
            print({"phase": "candidate_cache_exists", "year": year, "path": str(path)}, flush=True)
            continue
        print({"phase": "build_full_candidates_year", "year": year}, flush=True)
        data_year, cols = _build_year_candidates(year, max_rows=0, seed=17000 + year, as_dict=True)
        feature_cols = feature_cols or cols
        table = pa.Table.from_pydict(data_year)
        pq.write_table(table, path, compression="zstd")
        print({"phase": "build_full_candidates_year_done", "year": year, "rows": table.num_rows, "path": str(path)}, flush=True)
        del data_year, table
        gc.collect()
    if feature_cols is None:
        feature_cols = _read_feature_cols()
    else:
        _write_feature_cols(feature_cols)
    return feature_cols


def _load_years(years: tuple[int, ...], columns: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for year in years:
        path = _year_path(year)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True)


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    out = {
        "samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
    }
    if len(np.unique(y_true)) == 2:
        out["logloss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    return out


def _concat_year_column(years: tuple[int, ...], col: str, dtype: str = "float32") -> np.ndarray:
    chunks = []
    for year in years:
        chunks.append(pd.read_parquet(_year_path(year), columns=[col])[col].to_numpy(dtype=dtype, copy=False))
    return np.concatenate(chunks)


def _corr_select_from_cache(feature_cols: list[str], top_n: int, max_pair_corr: float, pair_sample_step: int = 5) -> tuple[list[str], pd.DataFrame]:
    y = _concat_year_column(CFG.split.train_years, "label", dtype="float32")
    rows = []
    for col in feature_cols:
        x = _concat_year_column(CFG.split.train_years, col, dtype="float32")
        ok = np.isfinite(x) & np.isfinite(y)
        if int(ok.sum()) < 50 or np.nanstd(x[ok]) <= 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append({"feature": col, "corr": corr, "abs_corr": abs(corr)})
        del x
    corr_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    del y
    gc.collect()

    force_keep = [c for c in feature_cols if any(token in c for token in FORCE_KEEP_TOKENS)]
    candidates = list(dict.fromkeys(force_keep + corr_df.head(top_n)["feature"].tolist()))
    selected: list[str] = []
    sampled: dict[str, np.ndarray] = {}
    for col in candidates:
        x = _concat_year_column(CFG.split.train_years, col, dtype="float32")[::pair_sample_step]
        sampled[col] = x
    for col in candidates:
        if col in selected:
            continue
        if not selected:
            selected.append(col)
            continue
        x = sampled[col]
        max_corr = 0.0
        for prev in selected:
            z = sampled[prev]
            ok = np.isfinite(x) & np.isfinite(z)
            if int(ok.sum()) < 50:
                continue
            c = float(np.corrcoef(x[ok], z[ok])[0, 1])
            if np.isfinite(c):
                max_corr = max(max_corr, abs(c))
        if not np.isfinite(max_corr) or max_corr < max_pair_corr or col in force_keep:
            selected.append(col)
    del sampled
    gc.collect()
    return selected, corr_df


def _iter_xy_batches(years: tuple[int, ...], cols: list[str], batch_rows: int = 60000):
    read_cols = cols + ["label"]
    for year in years:
        df = pd.read_parquet(_year_path(year), columns=read_cols)
        n = len(df)
        for start in range(0, n, batch_rows):
            end = min(start + batch_rows, n)
            x = df.iloc[start:end][cols].to_numpy("float32", copy=False)
            y = df.iloc[start:end]["label"].to_numpy("int8", copy=False)
            yield x, y
        del df
        gc.collect()


def _lasso_select_stream(
    candidate_cols: list[str],
    c_values: list[float],
    min_features: int,
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    scaler = StandardScaler()
    for x, _ in _iter_xy_batches(CFG.split.train_years, candidate_cols):
        scaler.partial_fit(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))

    valid_x_parts = []
    valid_y_parts = []
    for x, y in _iter_xy_batches(CFG.split.valid_years, candidate_cols):
        valid_x_parts.append(scaler.transform(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)).astype("float32", copy=False))
        valid_y_parts.append(y)
    x_valid = np.vstack(valid_x_parts)
    y_valid = np.concatenate(valid_y_parts)
    del valid_x_parts, valid_y_parts

    rows = []
    coef_by_c: dict[float, np.ndarray] = {}
    for c in c_values:
        alpha = 1.0 / max(float(c) * 1_000_000.0, 1.0)
        print({"phase": "lasso_sgd", "C_proxy": c, "alpha": alpha, "candidate_features": len(candidate_cols)}, flush=True)
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l1",
            alpha=alpha,
            max_iter=1,
            tol=None,
            random_state=19000,
            learning_rate="optimal",
            average=True,
        )
        first = True
        for epoch in range(3):
            for x, y in _iter_xy_batches(CFG.split.train_years, candidate_cols):
                xb = scaler.transform(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)).astype("float32", copy=False)
                if first:
                    clf.partial_fit(xb, y, classes=np.array([0, 1], dtype=np.int8))
                    first = False
                else:
                    clf.partial_fit(xb, y)
                del xb
            gc.collect()
        prob = clf.predict_proba(x_valid)[:, 1]
        coef = clf.coef_[0].astype("float32", copy=False)
        nonzero = int(np.count_nonzero(np.abs(coef) > 1e-8))
        rows.append(
            {
                "C": c,
                "alpha": alpha,
                "nonzero_features": nonzero,
                "valid_auc": float(roc_auc_score(y_valid, prob)),
                "valid_brier": float(brier_score_loss(y_valid, prob)),
            }
        )
        coef_by_c[float(c)] = coef.copy()
        del clf, prob
        gc.collect()

    lasso_df = pd.DataFrame(rows).sort_values(["valid_auc", "nonzero_features"], ascending=[False, True])
    viable = lasso_df[(lasso_df["nonzero_features"] >= min_features) & (lasso_df["nonzero_features"] <= max_features)]
    best_row = viable.iloc[0] if len(viable) else lasso_df.iloc[0]
    coef = coef_by_c[float(best_row["C"])]
    coef_df = pd.DataFrame({"feature": candidate_cols, "coef": coef, "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False)
    selected = coef_df[coef_df["abs_coef"] > 1e-8].head(max_features)["feature"].tolist()
    force_keep = [c for c in candidate_cols if any(token in c for token in FORCE_KEEP_TOKENS)]
    selected = list(dict.fromkeys(force_keep + selected))
    if len(selected) < min_features:
        selected = list(dict.fromkeys(selected + coef_df.head(min_features)["feature"].tolist()))
    return selected[:max_features], lasso_df


def _lasso_select(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidate_cols: list[str],
    c_values: list[float],
    min_features: int,
    max_features: int,
) -> tuple[list[str], pd.DataFrame, RobustClipScalerLite]:
    x_train_raw = train[candidate_cols].to_numpy("float32")
    y_train = train["label"].to_numpy("int8")
    x_valid_raw = valid[candidate_cols].to_numpy("float32")
    y_valid = valid["label"].to_numpy("int8")
    scaler = RobustClipScalerLite().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)

    rows = []
    coef_by_c: dict[float, np.ndarray] = {}
    for c in c_values:
        print({"phase": "lasso_logistic", "C": c, "candidate_features": len(candidate_cols)}, flush=True)
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float(c),
            max_iter=350,
            tol=0.003,
            n_jobs=-1,
            random_state=19000,
            class_weight=None,
        )
        clf.fit(x_train, y_train)
        coef = clf.coef_[0]
        prob = clf.predict_proba(x_valid)[:, 1]
        nonzero = int(np.count_nonzero(np.abs(coef) > 1e-8))
        rows.append(
            {
                "C": c,
                "nonzero_features": nonzero,
                "valid_auc": float(roc_auc_score(y_valid, prob)),
                "valid_brier": float(brier_score_loss(y_valid, prob)),
            }
        )
        coef_by_c[float(c)] = coef
        del clf, prob
        gc.collect()

    lasso_df = pd.DataFrame(rows).sort_values(["valid_auc", "nonzero_features"], ascending=[False, True])
    viable = lasso_df[(lasso_df["nonzero_features"] >= min_features) & (lasso_df["nonzero_features"] <= max_features)]
    best_row = viable.iloc[0] if len(viable) else lasso_df.iloc[0]
    best_c = float(best_row["C"])
    coef = coef_by_c[best_c]
    coef_df = pd.DataFrame({"feature": candidate_cols, "coef": coef, "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False)
    selected = coef_df[coef_df["abs_coef"] > 1e-8].head(max_features)["feature"].tolist()
    force_keep = [c for c in candidate_cols if any(token in c for token in FORCE_KEEP_TOKENS)]
    selected = list(dict.fromkeys(force_keep + selected))
    if len(selected) < min_features:
        selected = list(dict.fromkeys(selected + coef_df.head(min_features)["feature"].tolist()))
    return selected[:max_features], lasso_df, scaler


def _threshold_table(frame: pd.DataFrame, prob: np.ndarray, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        selected = (prob >= threshold) & (frame["session_london_ny"].to_numpy(np.int8) == 1)
        for year in sorted(frame["year"].unique()):
            ym = frame["year"].to_numpy() == year
            mask = selected & ym
            wins = int(np.count_nonzero(mask & (frame["label"].to_numpy(np.int8) == 1)))
            losses = int(np.count_nonzero(mask & (frame["label"].to_numpy(np.int8) == 0)))
            resolved = wins + losses
            rows.append(
                {
                    "threshold": float(threshold),
                    "year": int(year),
                    "resolved": resolved,
                    "wins": wins,
                    "losses": losses,
                    "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _select_threshold(valid_curve: pd.DataFrame) -> float:
    grouped = valid_curve.groupby("threshold", as_index=False).agg(resolved=("resolved", "sum"), wins=("wins", "sum"))
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    viable = grouped[grouped["resolved"] >= 2200].copy()
    if viable.empty:
        viable = grouped.copy()
    viable["score"] = viable["winrate"] - (viable["resolved"] - 3500).abs() / 3500
    return float(viable.sort_values(["score", "winrate"], ascending=[False, False]).iloc[0]["threshold"])


def _plot_importance(imp: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    top = imp.head(25).sort_values("importance")
    ax.barh(top["feature"], top["importance"])
    ax.set_title("RF selected feature importance top 25")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _simulate_from_probs(test: pd.DataFrame, prob: np.ndarray, threshold: float, out_dir: Path) -> pd.DataFrame:
    rows = []
    for year in CFG.split.test_years:
        mask = test["year"].to_numpy() == year
        chunk = test.loc[mask].copy().reset_index(drop=True)
        scores = prob[mask]
        orders = orders_from_scores(chunk, scores, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(out_dir / f"mophong_deals_{year}.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def run(
    force_candidates: bool,
    corr_top_n: int,
    max_pair_corr: float,
    lasso_c_values: list[float],
    min_features: int,
    max_features: int,
    n_estimators: int,
    skip_mophong: bool,
) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = build_full_candidate_cache(force=force_candidates)
    read_cols = feature_cols + ["label", "year", "dates", "session_london_ny", "buy_signal"]
    read_cols = list(dict.fromkeys(read_cols))

    print({"phase": "correlation_selection"}, flush=True)
    corr_selected, corr_df = _corr_select_from_cache(feature_cols, top_n=corr_top_n, max_pair_corr=max_pair_corr)
    corr_path = out_dir / "correlation_ranking.csv"
    corr_selected_path = out_dir / "correlation_selected_features.json"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    corr_selected_path.write_text(json.dumps(corr_selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print({"phase": "lasso_selection", "corr_selected": len(corr_selected)}, flush=True)
    selected_cols, lasso_df = _lasso_select_stream(
        candidate_cols=corr_selected,
        c_values=lasso_c_values,
        min_features=min_features,
        max_features=max_features,
    )
    lasso_path = out_dir / "lasso_selection_summary.csv"
    selected_path = out_dir / "selected_features_lasso_corr.json"
    lasso_df.to_csv(lasso_path, index=False, encoding="utf-8-sig")
    selected_path.write_text(json.dumps(selected_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    gc.collect()

    print({"phase": "load_selected_datasets", "selected_features": len(selected_cols)}, flush=True)
    read_selected = selected_cols + ["label", "year", "dates", "session_london_ny", "buy_signal"]
    read_selected = list(dict.fromkeys(read_selected))
    train = _load_years(CFG.split.train_years, columns=read_selected)
    valid = _load_years(CFG.split.valid_years, columns=read_selected)
    test = _load_years(CFG.split.test_years, columns=read_selected)

    x_train_raw = train[selected_cols].to_numpy("float32")
    y_train = train["label"].to_numpy("int8")
    x_valid_raw = valid[selected_cols].to_numpy("float32")
    y_valid = valid["label"].to_numpy("int8")
    x_test_raw = test[selected_cols].to_numpy("float32")
    y_test = test["label"].to_numpy("int8")

    scaler = RobustClipScalerLite().fit(x_train_raw)
    x_train = scaler.transform_inplace(x_train_raw)
    x_valid = scaler.transform_inplace(x_valid_raw)
    x_test = scaler.transform_inplace(x_test_raw)

    print({"phase": "train_rf", "rows_train": len(y_train), "selected_features": len(selected_cols)}, flush=True)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=250,
        min_samples_split=1200,
        max_features=0.70 if len(selected_cols) <= 80 else "sqrt",
        max_samples=0.85,
        class_weight=None,
        bootstrap=True,
        random_state=21003,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    p_valid = model.predict_proba(x_valid)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    valid_metrics = _metrics(y_valid, p_valid)
    test_metrics = _metrics(y_test, p_test)

    valid_curve = _threshold_table(valid, p_valid, CFG.threshold_grid)
    test_curve = _threshold_table(test, p_test, CFG.threshold_grid)
    threshold = _select_threshold(valid_curve)

    valid_curve_path = out_dir / "threshold_valid.csv"
    test_curve_path = out_dir / "threshold_test.csv"
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

    imp = pd.DataFrame({"feature": selected_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    imp_path = out_dir / "rf_feature_importance.csv"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")

    chart_paths = []
    chart_paths += plot_model_diagnostics(y_valid, p_valid, chart_dir, "rf_selected_valid")
    chart_paths.append(plot_threshold_curve(valid_curve, chart_dir / "threshold_valid.png", "Validation threshold curve"))
    chart_paths.append(plot_threshold_curve(test_curve, chart_dir / "threshold_test.png", "Test threshold curve"))
    chart_paths.append(_plot_importance(imp, chart_dir / "rf_feature_importance_top25.png"))

    mophong_yearly = pd.DataFrame()
    if not skip_mophong:
        print({"phase": "mophong", "threshold": threshold}, flush=True)
        mophong_yearly = _simulate_from_probs(test, p_test, threshold, out_dir)
        mophong_path = out_dir / "mophong_yearly.csv"
        mophong_yearly.to_csv(mophong_path, index=False, encoding="utf-8-sig")
        chart_paths.append(plot_yearly_mophong(mophong_yearly, chart_dir / "mophong_yearly.png"))
    else:
        mophong_path = out_dir / "mophong_yearly.csv"
        mophong_yearly.to_csv(mophong_path, index=False, encoding="utf-8-sig")

    model_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_bundle.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": selected_cols,
            "feature_family": FS14_NAME,
            "feature_set": FEATURE_FAMILY,
            "selected_threshold": threshold,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        model_path,
        compress=3,
    )

    summary = {
        "run_id": run_id,
        "feature_family": FEATURE_FAMILY,
        "all_features": len(feature_cols),
        "corr_selected": len(corr_selected),
        "lasso_selected": len(selected_cols),
        "selected_threshold": threshold,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "mophong_yearly": mophong_yearly.to_dict(orient="records"),
        "skip_mophong": skip_mophong,
        "elapsed_sec": round(time.time() - started, 2),
        "out_dir": str(out_dir),
        "model_path": str(model_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}"):
            mlflow.log_params(
                {
                    "feature_family": FEATURE_FAMILY,
                    "all_features": len(feature_cols),
                    "corr_top_n": corr_top_n,
                    "max_pair_corr": max_pair_corr,
                    "corr_selected": len(corr_selected),
                    "lasso_selected": len(selected_cols),
                    "lasso_c_values": ",".join(str(x) for x in lasso_c_values),
                    "n_estimators": n_estimators,
                    "selected_threshold": threshold,
                    "skip_mophong": skip_mophong,
                }
            )
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            if not mophong_yearly.empty:
                for _, row in mophong_yearly.iterrows():
                    year = int(row["year"])
                    mlflow.log_metric(f"mophong_{year}_winrate", float(row["winrate"]))
                    mlflow.log_metric(f"mophong_{year}_deals", float(row["deals"]))
                    mlflow.log_metric(f"mophong_{year}_resolved", float(row["resolved"]))
            for path in [corr_path, corr_selected_path, lasso_path, selected_path, valid_curve_path, test_curve_path, imp_path, mophong_path, summary_path, model_path]:
                mlflow.log_artifact(str(path))
            for path in chart_paths:
                mlflow.log_artifact(str(path), artifact_path="charts")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-candidates", action="store_true")
    parser.add_argument("--corr-top-n", type=int, default=180)
    parser.add_argument("--max-pair-corr", type=float, default=0.985)
    parser.add_argument("--lasso-c-values", default="0.01,0.03,0.06,0.1")
    parser.add_argument("--min-features", type=int, default=24)
    parser.add_argument("--max-features", type=int, default=70)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--skip-mophong", action="store_true")
    args = parser.parse_args()
    run(
        force_candidates=args.force_candidates,
        corr_top_n=args.corr_top_n,
        max_pair_corr=args.max_pair_corr,
        lasso_c_values=[float(x.strip()) for x in args.lasso_c_values.split(",") if x.strip()],
        min_features=args.min_features,
        max_features=args.max_features,
        n_estimators=args.n_estimators,
        skip_mophong=args.skip_mophong,
    )
