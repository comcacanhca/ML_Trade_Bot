from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from config import CFG, ensure_dirs
from data_io import DATE, load_year
from features import RobustClipScaler, build_features, named_feature_sets
from labels import buy_labels_next_open
from train_rf_mlflow import _candidate_dataset, build_cached_dataset


def _sample_xy(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(y)), size=max_rows, replace=False)
    return x[idx], y[idx]


def _candidate_frame_by_years(years: tuple[int, ...], feature_set: str, max_rows_per_year: int | None, seed: int) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    cols_ref: list[str] | None = None
    rng = np.random.default_rng(seed)
    for year in years:
        print({"phase": "build_year", "year": year}, flush=True)
        raw = load_year(year)
        frame = buy_labels_next_open(build_features(raw))
        frame["year"] = pd.to_datetime(frame[DATE]).dt.year.astype(int)
        cols = named_feature_sets(frame)[feature_set]
        if cols_ref is None:
            cols_ref = cols
        keep = (
            (frame["buy_signal"] == 1)
            & frame["label"].notna()
            & frame[cols].notna().all(axis=1)
        )
        cand = frame.loc[keep, cols + ["label", "year", DATE]].copy()
        if max_rows_per_year and len(cand) > max_rows_per_year:
            cand = cand.iloc[rng.choice(np.arange(len(cand)), size=max_rows_per_year, replace=False)].sort_index()
        frames.append(cand)
        print({"phase": "build_year_done", "year": year, "candidate_rows": len(cand)}, flush=True)
    if cols_ref is None:
        raise ValueError("No years loaded")
    return pd.concat(frames, ignore_index=True), cols_ref


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins * 5 or len(actual) < bins * 5:
        return np.nan
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _decile_lift(x: np.ndarray, y: np.ndarray, cols: list[str], top_features: list[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    frame = pd.DataFrame(x, columns=cols)
    frame["label"] = y
    for feature in top_features:
        if frame[feature].nunique(dropna=True) < 5:
            continue
        try:
            frame["_bin"] = pd.qcut(frame[feature], q=10, duplicates="drop")
        except ValueError:
            continue
        grouped = frame.groupby("_bin", observed=True)["label"].agg(["count", "mean"]).reset_index()
        grouped["feature"] = feature
        grouped["decile"] = np.arange(1, len(grouped) + 1)
        rows.append(grouped[["feature", "decile", "count", "mean"]])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(grouped["decile"], grouped["mean"] * 100, marker="o")
        ax.axhline(frame["label"].mean() * 100, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Train decile winrate: {feature}")
        ax.set_xlabel("Feature decile")
        ax.set_ylabel("Winrate %")
        fig.tight_layout()
        fig.savefig(out_dir / f"decile_lift_{feature}.png", dpi=140)
        plt.close(fig)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(force_dataset: bool, max_mi_rows: int, max_probe_rows: int, stream_years: bool, max_rows_per_year: int) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"behavior_fs14_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    if stream_years:
        print({"phase": "load_streamed_candidate_years"}, flush=True)
        train_frame, cols = _candidate_frame_by_years(CFG.split.train_years, "fs14_bb_overlay_norm_behavior", max_rows_per_year, 710)
        valid_frame, _ = _candidate_frame_by_years(CFG.split.valid_years, "fs14_bb_overlay_norm_behavior", max_rows_per_year, 711)
        test_frame, _ = _candidate_frame_by_years(CFG.split.test_years, "fs14_bb_overlay_norm_behavior", max_rows_per_year, 712)
        x_train_raw = train_frame[cols].to_numpy(float)
        y_train = train_frame["label"].astype(int).to_numpy()
        x_valid_raw = valid_frame[cols].to_numpy(float)
        y_valid = valid_frame["label"].astype(int).to_numpy()
        x_test_raw = test_frame[cols].to_numpy(float)
        y_test = test_frame["label"].astype(int).to_numpy()
        frame = pd.concat([train_frame, valid_frame, test_frame], ignore_index=True)
    else:
        print({"phase": "load_dataset", "force_dataset": force_dataset}, flush=True)
        frame = build_cached_dataset(force=force_dataset)
        cols = named_feature_sets(frame)["fs14_bb_overlay_norm_behavior"]
        x_train_raw, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
        x_valid_raw, y_valid, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
        x_test_raw, y_test, _ = _candidate_dataset(frame, cols, CFG.split.test_years)

    scaler = RobustClipScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)
    x_test = scaler.transform(x_test_raw)

    print({"phase": "mutual_info", "rows": min(len(y_train), max_mi_rows), "n_features": len(cols)}, flush=True)
    x_mi, y_mi = _sample_xy(x_train, y_train, max_mi_rows, 614)
    mi = mutual_info_classif(x_mi, y_mi, discrete_features=False, random_state=614)
    mi_df = pd.DataFrame({"feature": cols, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    mi_path = out_dir / "fs14_mutual_info.csv"
    mi_df.to_csv(mi_path, index=False, encoding="utf-8-sig")

    top_features = mi_df.head(15)["feature"].tolist()
    print({"phase": "decile_lift", "top_features": top_features[:5]}, flush=True)
    lift_df = _decile_lift(x_mi, y_mi, cols, top_features[:10], chart_dir)
    lift_path = out_dir / "fs14_decile_lift_top10.csv"
    lift_df.to_csv(lift_path, index=False, encoding="utf-8-sig")

    print({"phase": "probe_rf"}, flush=True)
    x_probe, y_probe = _sample_xy(x_train, y_train, max_probe_rows, 615)
    probe = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=250,
        min_samples_split=700,
        max_features="sqrt",
        class_weight=None,
        bootstrap=True,
        max_samples=0.8,
        random_state=615,
        n_jobs=-1,
    )
    probe.fit(x_probe, y_probe)
    p_valid = probe.predict_proba(x_valid)[:, 1]
    p_test = probe.predict_proba(x_test)[:, 1]
    probe_imp = pd.DataFrame({"feature": cols, "importance": probe.feature_importances_}).sort_values("importance", ascending=False)
    probe_imp_path = out_dir / "fs14_probe_rf_importance.csv"
    probe_imp.to_csv(probe_imp_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    top_imp = probe_imp.head(20).sort_values("importance")
    ax.barh(top_imp["feature"], top_imp["importance"])
    ax.set_title("fs14 probe RF feature importance top 20")
    fig.tight_layout()
    imp_chart = chart_dir / "fs14_probe_rf_importance_top20.png"
    fig.savefig(imp_chart, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print({"phase": "drift_psi"}, flush=True)
    year_rows = []
    if stream_years:
        train_mask = frame["year"].isin(CFG.split.train_years)
    else:
        train_mask = frame["year"].isin(CFG.split.train_years) & (frame["buy_signal"] == 1) & frame[cols].notna().all(axis=1)
    for feature in top_features:
        train_vals = frame.loc[train_mask, feature].to_numpy(float)
        for year in CFG.split.valid_years + CFG.split.test_years:
            if stream_years:
                mask = frame["year"] == year
            else:
                mask = (frame["year"] == year) & (frame["buy_signal"] == 1) & frame[cols].notna().all(axis=1)
            year_rows.append({"feature": feature, "year": year, "psi_vs_train": _psi(train_vals, frame.loc[mask, feature].to_numpy(float))})
    psi_df = pd.DataFrame(year_rows)
    psi_path = out_dir / "fs14_top15_psi_vs_train.csv"
    psi_df.to_csv(psi_path, index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "feature_set": "fs14_bb_overlay_norm_behavior",
        "n_features": len(cols),
        "train_samples": int(len(y_train)),
        "valid_samples": int(len(y_valid)),
        "test_samples": int(len(y_test)),
        "valid_auc_probe": float(roc_auc_score(y_valid, p_valid)),
        "test_auc_probe": float(roc_auc_score(y_test, p_test)),
        "top_mi": mi_df.head(15).to_dict(orient="records"),
        "top_probe_importance": probe_imp.head(15).to_dict(orient="records"),
        "max_top15_psi": float(psi_df["psi_vs_train"].max()) if len(psi_df) else np.nan,
        "elapsed_sec": round(time.time() - started, 2),
        "stream_years": stream_years,
        "max_rows_per_year": max_rows_per_year,
        "outputs": {
            "mutual_info": str(mi_path),
            "decile_lift": str(lift_path),
            "probe_importance": str(probe_imp_path),
            "psi": str(psi_path),
            "importance_chart": str(imp_chart),
            "chart_dir": str(chart_dir),
        },
    }
    summary_path = out_dir / "fs14_behavior_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--stream-years", action="store_true")
    parser.add_argument("--max-rows-per-year", type=int, default=50000)
    parser.add_argument("--max-mi-rows", type=int, default=120000)
    parser.add_argument("--max-probe-rows", type=int, default=180000)
    args = parser.parse_args()
    run(
        force_dataset=args.force_dataset,
        max_mi_rows=args.max_mi_rows,
        max_probe_rows=args.max_probe_rows,
        stream_years=args.stream_years,
        max_rows_per_year=args.max_rows_per_year,
    )
