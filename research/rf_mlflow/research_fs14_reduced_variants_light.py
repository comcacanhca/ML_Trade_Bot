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
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from research_fs14_overlay_norm_light import FS14_NAME, RobustClipScalerLite, _build_year_candidates, _psi


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _variant_columns(cols: list[str]) -> dict[str, list[str]]:
    base = [c for c in cols if c.startswith("tod_") or c.startswith("dow_") or c.startswith("sig_") or c == "buy_signal"]
    fs03_lags = [c for c in cols if c in {
        "close_diff_lag_5",
        "close_diff_lag_8",
        "close_diff_lag_13",
        "close_diff_lag_21",
        "body_lag_5",
        "body_lag_8",
        "body_lag_13",
        "body_lag_21",
    }]
    cross = [
        c for c in cols
        if c in {
            "bb20_bb100_pos_diff",
            "bb20_bb200_pos_diff",
            "bb100_bb200_pos_diff",
            "bb20_width_over_bb100",
            "bb20_width_over_bb200",
            "bb100_width_over_bb200",
            "bb_lower_agree_20_100",
            "bb_lower_agree_20_200",
            "bb_lower_agree_all",
            "bb_trend_context_buy",
        }
    ]
    cycle_rank = [c for c in cols if "_cycle_" in c or c.endswith("_rank_500") or "_width_atr_rank_500" in c]
    velocity = [c for c in cols if "_ov_velocity" in c or "_ov_acceleration" in c]
    distance_lag = [c for c in cols if ("_ov_value_lag" in c or "_ov_distance_lag" in c)]
    rolling_atr = [c for c in cols if "_gap_atr_rollmean_" in c or "_gap_atr_rollstd_" in c]
    bb20_roll = [c for c in rolling_atr if c.startswith("bb20_")]
    bb100_200_roll = [c for c in rolling_atr if c.startswith("bb100_") or c.startswith("bb200_")]

    def uniq(items: list[str]) -> list[str]:
        return list(dict.fromkeys([c for c in items if c in cols and c != "atr14"]))

    return {
        "fs14r00_fs03_like_light": uniq(base + fs03_lags),
        "fs14r01_cycle_rank_cross": uniq(base + cycle_rank + cross),
        "fs14r02_distance_lags_cross": uniq(base + distance_lag + cross),
        "fs14r03_velocity_accel_cross": uniq(base + velocity + cross),
        "fs14r04_bb20_rolling_atr": uniq(base + bb20_roll + cross),
        "fs14r05_bb100_200_rolling_atr": uniq(base + bb100_200_roll + cross),
        "fs14r06_all_no_raw_atr": uniq(base + cycle_rank + velocity + distance_lag + rolling_atr + cross),
        "fs14r07_fs03_plus_cycle_rank_cross": uniq(base + fs03_lags + cycle_rank + cross),
        "fs14r08_fs03_plus_velocity_cross": uniq(base + fs03_lags + velocity + cross),
        "fs14r09_fs03_plus_bb100_200_roll": uniq(base + fs03_lags + bb100_200_roll + cross),
    }


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
    }


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
                    "threshold": threshold,
                    "year": int(year),
                    "resolved_sample": resolved,
                    "wins": wins,
                    "losses": losses,
                    "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    s = summary.sort_values("valid_auc", ascending=False)
    x = np.arange(len(s))
    ax1.bar(x, s["valid_auc"], label="valid AUC")
    ax1.set_ylim(0.48, max(0.54, float(s["valid_auc"].max()) + 0.01))
    ax1.set_xticks(x)
    ax1.set_xticklabels(s["variant"], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, s["test_auc"], marker="o", color="tab:red", label="test AUC")
    ax1.set_title("fs14 reduced variants: probe AUC")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_importance(imp: pd.DataFrame, path: Path, title: str, top_n: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    top = imp.head(top_n).sort_values("importance")
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("RF Gini importance")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_threshold_curve(curve: pd.DataFrame, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = curve.groupby("threshold", as_index=False).agg(
        resolved_sample=("resolved_sample", "sum"),
        wins=("wins", "sum"),
    )
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved_sample"].replace(0, np.nan)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(grouped["threshold"], grouped["resolved_sample"], marker="o", label="sample selected/resolved")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Sample selected")
    ax2 = ax1.twinx()
    ax2.plot(grouped["threshold"], grouped["winrate"], marker="o", color="tab:red", label="sample WR")
    ax2.axhline(58, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Winrate %")
    ax1.set_title(title)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_score_distribution(y_true: np.ndarray, prob: np.ndarray, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prob[y_true == 0], bins=40, alpha=0.55, label="Lose")
    ax.hist(prob[y_true == 1], bins=40, alpha=0.55, label="Win")
    ax.set_xlabel("Predicted P(Win)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_calibration(y_true: np.ndarray, prob: np.ndarray, path: Path, title: str, bins: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y": y_true, "p": prob}).dropna()
    try:
        df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
        cal = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), obs_wr=("y", "mean"), count=("y", "size")).reset_index()
    except ValueError:
        cal = pd.DataFrame({"mean_pred": [], "obs_wr": [], "count": []})
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(cal):
        ax.plot(cal["mean_pred"], cal["obs_wr"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed winrate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_yearly_threshold(curve: pd.DataFrame, path: Path, title: str, threshold: float = 0.56) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = curve[curve["threshold"] == threshold].copy()
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rows))
    ax1.bar(x, rows["resolved_sample"], label="sample selected")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rows["year"].astype(str))
    ax1.set_ylabel("Sample selected")
    ax2 = ax1.twinx()
    ax2.plot(x, rows["winrate"], marker="o", color="tab:red", label="sample WR")
    ax2.axhline(58, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Winrate %")
    ax1.set_title(title)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run(max_rows_per_year: int, n_estimators: int, only_variants: tuple[str, ...] = ()) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"fs14_reduced_variants_light_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    feature_cols: list[str] | None = None
    for year in CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years:
        print({"phase": "build_year", "year": year}, flush=True)
        df_year, cols = _build_year_candidates(year, max_rows=max_rows_per_year, seed=12000 + year)
        feature_cols = feature_cols or cols
        frames.append(df_year)
        print({"phase": "build_year_done", "year": year, "rows": len(df_year)}, flush=True)
        gc.collect()

    assert feature_cols is not None
    data = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    train = data[data["year"].isin(CFG.split.train_years)].reset_index(drop=True)
    valid = data[data["year"].isin(CFG.split.valid_years)].reset_index(drop=True)
    test = data[data["year"].isin(CFG.split.test_years)].reset_index(drop=True)
    variants = _variant_columns(feature_cols)
    if only_variants:
        variants = {name: cols for name, cols in variants.items() if name in set(only_variants)}
        if not variants:
            raise ValueError(f"No requested variants found: {only_variants}")

    mlflow = _mlflow()
    parent = mlflow.start_run(run_name=f"fs14_reduced_variants_light_{run_id}") if mlflow is not None else None
    summary_rows = []

    try:
        for idx, (name, cols) in enumerate(variants.items(), start=1):
            print({"phase": "variant", "name": name, "n_features": len(cols)}, flush=True)
            x_train_raw = train[cols].to_numpy("float32")
            y_train = train["label"].to_numpy("int8")
            x_valid_raw = valid[cols].to_numpy("float32")
            y_valid = valid["label"].to_numpy("int8")
            x_test_raw = test[cols].to_numpy("float32")
            y_test = test["label"].to_numpy("int8")

            scaler = RobustClipScalerLite().fit(x_train_raw)
            x_train = scaler.transform(x_train_raw)
            x_valid = scaler.transform(x_valid_raw)
            x_test = scaler.transform(x_test_raw)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_leaf=200,
                min_samples_split=600,
                max_features="sqrt",
                max_samples=0.8,
                class_weight=None,
                bootstrap=True,
                random_state=13000 + idx,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            p_valid = model.predict_proba(x_valid)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            valid_metrics = _metrics(y_valid, p_valid)
            test_metrics = _metrics(y_test, p_test)

            imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            imp_path = out_dir / f"{name}_importance.csv"
            imp.to_csv(imp_path, index=False, encoding="utf-8-sig")

            valid_curve = _threshold_table(valid, p_valid, CFG.threshold_grid)
            test_curve = _threshold_table(test, p_test, CFG.threshold_grid)
            valid_curve_path = out_dir / f"{name}_threshold_valid_sample.csv"
            test_curve_path = out_dir / f"{name}_threshold_test_sample.csv"
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")

            variant_chart_dir = chart_dir / name
            chart_paths = [
                _plot_importance(imp, variant_chart_dir / "importance_top20.png", f"{name} importance top 20"),
                _plot_threshold_curve(valid_curve, variant_chart_dir / "threshold_valid_sample.png", f"{name} valid sample threshold curve"),
                _plot_threshold_curve(test_curve, variant_chart_dir / "threshold_test_sample.png", f"{name} test sample threshold curve"),
                _plot_score_distribution(y_valid, p_valid, variant_chart_dir / "score_distribution_valid.png", f"{name} valid score distribution"),
                _plot_score_distribution(y_test, p_test, variant_chart_dir / "score_distribution_test.png", f"{name} test score distribution"),
                _plot_calibration(y_valid, p_valid, variant_chart_dir / "calibration_valid.png", f"{name} valid calibration"),
                _plot_calibration(y_test, p_test, variant_chart_dir / "calibration_test.png", f"{name} test calibration"),
                _plot_yearly_threshold(test_curve, variant_chart_dir / "yearly_test_sample_t056.png", f"{name} test yearly sample @0.56", threshold=0.56),
            ]

            top_features = imp.head(12)["feature"].tolist()
            psi_values = []
            for feature in top_features:
                train_vals = train[feature].to_numpy(float)
                for year in CFG.split.valid_years + CFG.split.test_years:
                    psi_values.append(_psi(train_vals, data.loc[data["year"] == year, feature].to_numpy(float)))
            max_psi = float(np.nanmax(psi_values)) if psi_values else np.nan

            model_file = CFG.model_dir / f"{name}_{run_id}_light_bundle.joblib"
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "feature_columns": cols,
                    "feature_set": name,
                    "base_feature_family": FS14_NAME,
                    "valid_metrics": valid_metrics,
                    "test_metrics": test_metrics,
                    "max_top12_psi": max_psi,
                },
                model_file,
                compress=3,
            )

            row = {
                "variant": name,
                "n_features": len(cols),
                "valid_auc": valid_metrics["auc"],
                "valid_brier": valid_metrics["brier"],
                "test_auc": test_metrics["auc"],
                "test_brier": test_metrics["brier"],
                "max_top12_psi": max_psi,
                "model_file": str(model_file),
            }
            summary_rows.append(row)
            if mlflow is not None:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_params({"variant": name, "n_features": len(cols), "n_estimators": n_estimators})
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_{k}", v)
                    mlflow.log_metric("max_top12_psi", max_psi)
                    for path in [imp_path, valid_curve_path, test_curve_path, model_file]:
                        mlflow.log_artifact(str(path))
                    for path in chart_paths:
                        mlflow.log_artifact(str(path), artifact_path=f"charts/{name}")

            del x_train_raw, x_valid_raw, x_test_raw, x_train, x_valid, x_test, model
            gc.collect()

        summary = pd.DataFrame(summary_rows).sort_values("valid_auc", ascending=False)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        chart = _plot_summary(summary, chart_dir / "variant_auc_summary.png")
        meta = {
            "run_id": run_id,
            "max_rows_per_year": max_rows_per_year,
            "n_estimators": n_estimators,
            "only_variants": ",".join(only_variants),
            "rows_train": int(len(train)),
            "rows_valid": int(len(valid)),
            "rows_test": int(len(test)),
            "elapsed_sec": round(time.time() - started, 2),
            "summary_path": str(summary_path),
            "chart": str(chart),
            "top_summary": summary.to_dict(orient="records"),
        }
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if mlflow is not None:
            mlflow.log_params({"max_rows_per_year": max_rows_per_year, "n_estimators": n_estimators})
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(meta_path))
            mlflow.log_artifact(str(chart), artifact_path="charts")
        print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
        return meta
    finally:
        if mlflow is not None and parent is not None:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows-per-year", type=int, default=15000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--only-variants", default="")
    args = parser.parse_args()
    only_variants = tuple(item.strip() for item in args.only_variants.split(",") if item.strip())
    run(max_rows_per_year=args.max_rows_per_year, n_estimators=args.n_estimators, only_variants=only_variants)
