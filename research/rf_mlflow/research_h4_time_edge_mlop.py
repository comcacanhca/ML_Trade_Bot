from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from plots import plot_model_diagnostics, plot_threshold_curve
from research_random50_initial_features import (
    CACHE_FAMILY,
    META_COLS,
    TIME_FIXED_FEATURES,
    _mlflow,
    _plot_importance,
    _plot_shap,
    _sample_train,
    _select_threshold,
    _threshold_table,
    _total_at_threshold,
    build_cache,
)
from research_rolling_zscore_50_h4h1 import (
    PASSTHROUGH_FEATURES,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    _rolling_zscore_frame,
)


FEATURE_FAMILY = "h4_time_edge_mlop_lgbm_rz500"
EDGE_MIN_VALID_RESOLVED = 800
DEFAULT_THRESHOLD_GRID = tuple(np.round(np.arange(0.50, 0.701, 0.01), 2))

SIGNAL_FEATURES = ["sig_bb_reversion_buy", "sig_pullback_trend_buy", "sig_momentum_buy"]
FS03_CORE = [
    "close_diff_lag_5",
    "body_lag_5",
    "close_diff_lag_8",
    "body_lag_8",
    "close_diff_lag_13",
    "body_lag_13",
    "close_diff_lag_21",
    "body_lag_21",
    "z_close_20",
    "ret_sum_10",
    "ema_dist_8",
    "ema_dist_55",
    "rsi_14",
]
H4_CORE = [
    "h4_ret",
    "h4_range",
    "h4_ret_sum_3",
    "h4_ret_sum_6",
    "h4_ret_sum_12",
    "h4_volatility_3",
    "h4_volatility_6",
    "h4_volatility_12",
    "h4_ema_dist_8",
    "h4_ema_dist_21",
    "h4_ema_dist_55",
    "h4_ema_slope_8",
    "h4_ema_slope_21",
    "h4_ema_slope_55",
    "h4_rsi_14",
]
H1_CONFIRM = [
    "h1_ret",
    "h1_range",
    "h1_ret_sum_3",
    "h1_ret_sum_6",
    "h1_volatility_3",
    "h1_volatility_6",
    "h1_ema_dist_8",
    "h1_ema_dist_21",
    "h1_ema_slope_21",
    "h1_rsi_14",
]
BB20_CORE = [
    "bb20_lower_ov_value",
    "bb20_lower_ov_distance",
    "bb20_lower_ov_distance_lag1",
    "bb20_lower_ov_distance_lag3",
    "bb20_lower_ov_std",
    "bb20_lower_ov_velocity1",
    "bb20_lower_ov_velocity3",
    "bb20_lower_ov_acceleration1",
    "bb20_mid_ov_distance",
    "bb20_mid_ov_velocity1",
    "bb20_upper_ov_distance",
]
DONCHIAN_CORE = [
    "donchian20_lower_ov_distance",
    "donchian20_lower_ov_distance_lag1",
    "donchian20_lower_ov_velocity1",
    "donchian20_mid_ov_distance",
    "donchian20_upper_ov_distance",
]
KELTNER_CORE = [
    "keltner20_lower_ov_distance",
    "keltner20_mid_ov_distance",
    "keltner20_upper_ov_distance",
    "keltner20_lower_ov_velocity1",
]
MA_CONTEXT = [
    "sma20_ov_distance",
    "sma50_ov_distance",
    "ema21_ov_distance",
    "ema55_ov_distance",
    "ema89_ov_distance",
]

GATE_COLS = sorted(
    set(
        [
            "label",
            "year",
            "session_london_ny",
            "session_ny",
            "session_asia",
            "sig_bb_reversion_buy",
            "sig_pullback_trend_buy",
            "sig_momentum_buy",
            "buy_signal",
            "h4_ret_sum_3",
            "h4_ret_sum_6",
            "h4_ret_sum_12",
            "h4_volatility_3",
            "h4_volatility_6",
            "h4_ema_slope_21",
            "h4_ema_dist_55",
            "h4_rsi_14",
            "h1_ret_sum_3",
            "h1_volatility_3",
            "bb20_lower_ov_distance",
            "bb20_lower_ov_velocity1",
            "donchian20_lower_ov_distance",
        ]
    )
)


def _cache_dir() -> Path:
    return CFG.cache_dir / CACHE_FAMILY


def _year_path(year: int) -> Path:
    return _cache_dir() / f"candidates_{year}.parquet"


def _safe_cols(cols: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return [c for c in dict.fromkeys(cols) if c in available_set and c not in META_COLS]


def _read_year(year: int, columns: list[str]) -> pd.DataFrame:
    path = _year_path(year)
    missing = [c for c in columns if c not in pd.read_parquet(path, columns=[]).columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing[:20]}")
    return pd.read_parquet(path, columns=columns)


def _read_years(years: tuple[int, ...], columns: list[str]) -> pd.DataFrame:
    parts = []
    for year in years:
        parts.append(pd.read_parquet(_year_path(year), columns=columns))
    return pd.concat(parts, ignore_index=True)


def _metric_row(split: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "split": split,
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else float("nan"),
    }


def _wr_summary(frame: pd.DataFrame, mask: np.ndarray, split: str, gate: str) -> dict:
    y = frame["label"].to_numpy(np.int8)
    rows = int(mask.sum())
    wins = int(np.count_nonzero(mask & (y == 1)))
    losses = int(np.count_nonzero(mask & (y == 0)))
    resolved = wins + losses
    return {
        "split": split,
        "gate": gate,
        "rows": rows,
        "wins": wins,
        "losses": losses,
        "resolved": resolved,
        "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
        "coverage": round(rows * 100 / len(frame), 4) if len(frame) else 0.0,
    }


def _quantiles(train: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in cols:
        if col not in train.columns:
            continue
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        out[col] = {
            "q20": float(s.quantile(0.20)),
            "q35": float(s.quantile(0.35)),
            "q50": float(s.quantile(0.50)),
            "q65": float(s.quantile(0.65)),
            "q80": float(s.quantile(0.80)),
        }
    return out


def _gate_mask(frame: pd.DataFrame, gate: str, q: dict[str, dict[str, float]]) -> np.ndarray:
    n = len(frame)
    mask = np.ones(n, dtype=bool)

    def has(col: str) -> bool:
        return col in frame.columns and col in q

    def gt(col: str, key: str) -> np.ndarray:
        return pd.to_numeric(frame[col], errors="coerce").to_numpy(float) > q[col][key]

    def lt(col: str, key: str) -> np.ndarray:
        return pd.to_numeric(frame[col], errors="coerce").to_numpy(float) < q[col][key]

    def eq1(col: str) -> np.ndarray:
        return frame[col].fillna(0).to_numpy(np.int8) == 1

    if gate == "all_buy_signal":
        return mask
    if gate == "london_ny":
        return eq1("session_london_ny") if "session_london_ny" in frame.columns else mask
    if gate == "ny":
        return eq1("session_ny") if "session_ny" in frame.columns else mask
    if gate == "bb_reversion_signal":
        return eq1("sig_bb_reversion_buy") if "sig_bb_reversion_buy" in frame.columns else np.zeros(n, dtype=bool)
    if gate == "pullback_signal":
        return eq1("sig_pullback_trend_buy") if "sig_pullback_trend_buy" in frame.columns else np.zeros(n, dtype=bool)
    if gate == "h4_momentum_up":
        return gt("h4_ret_sum_6", "q65") if has("h4_ret_sum_6") else mask
    if gate == "h4_pullback_uptrend":
        out = mask
        if has("h4_ema_dist_55"):
            out &= gt("h4_ema_dist_55", "q50")
        if has("h4_ret_sum_3"):
            out &= lt("h4_ret_sum_3", "q35")
        return out
    if gate == "h4_reversion_from_weakness":
        out = mask
        if has("h4_ret_sum_12"):
            out &= lt("h4_ret_sum_12", "q35")
        if has("h4_rsi_14"):
            out &= lt("h4_rsi_14", "q50")
        return out
    if gate == "h4_low_vol":
        return lt("h4_volatility_6", "q35") if has("h4_volatility_6") else mask
    if gate == "h4_midlow_vol_london":
        out = eq1("session_london_ny") if "session_london_ny" in frame.columns else mask
        if has("h4_volatility_6"):
            out &= lt("h4_volatility_6", "q65")
        return out
    if gate == "bb_h4_lowvol":
        out = eq1("sig_bb_reversion_buy") if "sig_bb_reversion_buy" in frame.columns else np.zeros(n, dtype=bool)
        if has("h4_volatility_6"):
            out &= lt("h4_volatility_6", "q65")
        return out
    if gate == "bb_h4_weak_london":
        out = eq1("sig_bb_reversion_buy") if "sig_bb_reversion_buy" in frame.columns else np.zeros(n, dtype=bool)
        if "session_london_ny" in frame.columns:
            out &= eq1("session_london_ny")
        if has("h4_ret_sum_12"):
            out &= lt("h4_ret_sum_12", "q50")
        return out
    if gate == "pullback_h4_up_london":
        out = eq1("sig_pullback_trend_buy") if "sig_pullback_trend_buy" in frame.columns else np.zeros(n, dtype=bool)
        if "session_london_ny" in frame.columns:
            out &= eq1("session_london_ny")
        if has("h4_ema_dist_55"):
            out &= gt("h4_ema_dist_55", "q50")
        return out
    return mask


def _gate_names() -> list[str]:
    return [
        "all_buy_signal",
        "london_ny",
        "ny",
        "bb_reversion_signal",
        "pullback_signal",
        "h4_momentum_up",
        "h4_pullback_uptrend",
        "h4_reversion_from_weakness",
        "h4_low_vol",
        "h4_midlow_vol_london",
        "bb_h4_lowvol",
        "bb_h4_weak_london",
        "pullback_h4_up_london",
    ]


def _load_edge_frames(available_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = set(available_cols)
    cols = [c for c in GATE_COLS if c in available or c in {"label", "year"}]
    return (
        _read_years(CFG.split.train_years, cols),
        _read_years(CFG.split.valid_years, cols),
        _read_years(CFG.split.test_years, cols),
    )


def edge_scan(out_dir: Path, available_cols: list[str]) -> pd.DataFrame:
    train, valid, test = _load_edge_frames(available_cols)
    q = _quantiles(train, [c for c in GATE_COLS if c not in {"label", "year"}])
    rows = []
    for gate in _gate_names():
        for split, frame in [("train", train), ("valid", valid), ("test", test)]:
            rows.append(_wr_summary(frame, _gate_mask(frame, gate, q), split, gate))
    scan = pd.DataFrame(rows)
    scan.to_csv(out_dir / "edge_scan_gates.csv", index=False, encoding="utf-8-sig")
    Path(out_dir / "edge_gate_quantiles.json").write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")

    pivot = scan.pivot(index="gate", columns="split", values="winrate").reset_index()
    counts = scan.pivot(index="gate", columns="split", values="resolved").reset_index()
    merged = pivot.merge(counts, on="gate", suffixes=("_wr", "_resolved"))
    merged.to_csv(out_dir / "edge_scan_gates_pivot.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 6))
    valid_rows = scan[scan["split"] == "valid"].sort_values("winrate", ascending=False)
    ax.bar(valid_rows["gate"], valid_rows["winrate"])
    ax.axhline(float(train["label"].mean() * 100), color="gray", linestyle="--", linewidth=1, label="train base WR")
    ax.set_title("Gate edge scan - valid winrate")
    ax.set_ylabel("Winrate %")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "edge_scan_valid_winrate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    del train, valid, test
    gc.collect()
    return scan


def _feature_sets(available_cols: list[str]) -> dict[str, list[str]]:
    base = list(TIME_FIXED_FEATURES) + SIGNAL_FEATURES
    specs = {
        "fs03_core": base + FS03_CORE,
        "h4_time_core": base + H4_CORE,
        "h4_h1_context": base + H4_CORE + H1_CONFIRM,
        "bb20_h4": base + FS03_CORE + H4_CORE + BB20_CORE,
        "bb20_h4_h1": base + FS03_CORE + H4_CORE + H1_CONFIRM + BB20_CORE,
        "bb20_donchian_h4": base + FS03_CORE + H4_CORE + BB20_CORE + DONCHIAN_CORE,
        "bb20_channel_h4": base + FS03_CORE + H4_CORE + BB20_CORE + DONCHIAN_CORE + KELTNER_CORE,
        "bb20_ma_context_h4": base + FS03_CORE + H4_CORE + BB20_CORE + MA_CONTEXT,
    }
    return {name: _safe_cols(cols, available_cols) for name, cols in specs.items()}


def _make_model_specs(available_cols: list[str], n_models: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    feature_sets = _feature_sets(available_cols)
    gates = _gate_names()
    # Prefer gates that directly match the known historical edge.
    weighted_gates = [
        "bb_h4_weak_london",
        "bb_h4_lowvol",
        "h4_midlow_vol_london",
        "pullback_h4_up_london",
        "bb_reversion_signal",
        "h4_reversion_from_weakness",
        *gates,
    ]
    param_grid = [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3, "min_child_samples": 250, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 1.5},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4, "min_child_samples": 350, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 0.10, "reg_lambda": 2.0},
        {"n_estimators": 450, "learning_rate": 0.025, "num_leaves": 11, "max_depth": 3, "min_child_samples": 500, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"n_estimators": 700, "learning_rate": 0.015, "num_leaves": 15, "max_depth": 4, "min_child_samples": 700, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.15, "reg_lambda": 3.0},
    ]
    specs = []
    fs_names = list(feature_sets)
    for i in range(n_models):
        fs_name = fs_names[i % len(fs_names)] if i < len(fs_names) else str(rng.choice(fs_names))
        gate = weighted_gates[i % len(weighted_gates)] if i < len(weighted_gates) else str(rng.choice(weighted_gates))
        params = dict(param_grid[i % len(param_grid)])
        params["subsample_freq"] = 1
        specs.append(
            {
                "model_name": f"h4edge_lgbm_{i + 1:02d}",
                "feature_set": fs_name,
                "gate": gate,
                "features": feature_sets[fs_name],
                "params": params,
                "seed": seed + i + 1,
            }
        )
    return specs


def _load_scaled_split(
    years: tuple[int, ...],
    model_cols: list[str],
    gate_cols: list[str],
    gate: str,
    gate_quantiles: dict[str, dict[str, float]],
) -> pd.DataFrame:
    read_cols = list(dict.fromkeys(model_cols + gate_cols + ["label", "year", "session_london_ny"]))
    parts = []
    for year in years:
        frame = pd.read_parquet(_year_path(year), columns=read_cols)
        gate_mask = _gate_mask(frame, gate, gate_quantiles)
        frame = _rolling_zscore_frame(frame, model_cols)
        frame = frame.loc[gate_mask].reset_index(drop=True)
        parts.append(frame)
        del frame
        gc.collect()
    if not parts:
        return pd.DataFrame(columns=read_cols)
    return pd.concat(parts, ignore_index=True)


def _score_objective(test_total: dict, test_curve: pd.DataFrame, threshold: float, test_auc: float) -> float:
    rows = test_curve[test_curve["threshold"] == threshold]
    wr_std = float(rows["winrate"].std()) if len(rows) > 1 else 10.0
    min_year = int(rows["resolved"].min()) if len(rows) else 0
    deal_score = min(1.0, float(test_total["resolved"]) / 7000.0)
    sparse_penalty = max(0, 1000 - min_year) / 300.0
    return float(test_total["winrate"] + 2.0 * deal_score + 2.0 * test_auc - 0.45 * wr_std - sparse_penalty)


def _yearly_threshold_table(frame: pd.DataFrame, prob: np.ndarray, thresholds: tuple[float, ...]) -> pd.DataFrame:
    return _threshold_table(frame[["label", "year", "session_london_ny"]], prob, thresholds)


def run(
    n_models: int,
    max_train_rows: int,
    seed: int,
    shap_rows: int,
    force_cache: bool,
    run_edge_scan: bool,
    enable_mlflow: bool,
    start_index: int,
) -> dict:
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_dir = out_dir / "models"
    model_artifact_dir.mkdir(parents=True, exist_ok=True)

    available_cols = build_cache(force_cache)
    missing_paths = [str(_year_path(y)) for y in CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years if not _year_path(y).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing candidate cache files: {missing_paths[:5]}")

    train_edge, _, _ = _load_edge_frames(available_cols)
    gate_quantiles = _quantiles(train_edge, [c for c in GATE_COLS if c not in {"label", "year"}])
    del train_edge
    gc.collect()

    scan = edge_scan(out_dir, available_cols) if run_edge_scan else pd.DataFrame()
    specs = _make_model_specs(available_cols, n_models, seed)
    specs_path = out_dir / "model_specs.json"
    specs_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    scaling_contract = {
        "scaler": "rolling_zscore",
        "window": ROLLING_WINDOW,
        "min_periods": ROLLING_MIN_PERIODS,
        "use_past_only": True,
        "passthrough_features": sorted(PASSTHROUGH_FEATURES),
        "note": "time/session/signal features are passthrough; numeric market features are rolling_zscore per year before gate filtering.",
    }
    (out_dir / "feature_scaling.json").write_text(json.dumps(scaling_contract, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow() if enable_mlflow else None
    parent = mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}") if mlflow else None
    summary_rows = []
    try:
        if mlflow:
            mlflow.log_params(
                {
                    "feature_family": FEATURE_FAMILY,
                    "model_type": "LightGBM",
                    "n_models": n_models,
                    "max_train_rows": max_train_rows,
                    "seed": seed,
                    "scaler": "rolling_zscore",
                    "rolling_window": ROLLING_WINDOW,
                    "rolling_min_periods": ROLLING_MIN_PERIODS,
                    "cache_family": CACHE_FAMILY,
                }
            )
            for path in [specs_path, out_dir / "feature_scaling.json", out_dir / "edge_scan_gates.csv", out_dir / "edge_scan_gates_pivot.csv", out_dir / "edge_scan_valid_winrate.png"]:
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="setup")

        for idx, spec in enumerate(specs, start=1):
            if idx < start_index:
                continue
            name = spec["model_name"]
            cols = spec["features"]
            gate = spec["gate"]
            params = spec["params"]
            print({"phase": "model_start", "idx": idx, "n_models": n_models, "name": name, "feature_set": spec["feature_set"], "gate": gate, "n_features": len(cols)}, flush=True)

            train = _load_scaled_split(CFG.split.train_years, cols, GATE_COLS, gate, gate_quantiles)
            train = _sample_train(train, max_train_rows, int(spec["seed"]))
            valid = _load_scaled_split(CFG.split.valid_years, cols, GATE_COLS, gate, gate_quantiles)
            test = _load_scaled_split(CFG.split.test_years, cols, GATE_COLS, gate, gate_quantiles)

            if len(train) < 1000 or len(valid) < 300 or len(test) < 300:
                row = {
                    "model_name": name,
                    "feature_set": spec["feature_set"],
                    "gate": gate,
                    "n_features": len(cols),
                    "skip_reason": "too_few_rows",
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "test_rows": len(test),
                }
                summary_rows.append(row)
                pd.DataFrame(summary_rows).to_csv(out_dir / "summary_checkpoint.csv", index=False, encoding="utf-8-sig")
                del train, valid, test
                gc.collect()
                continue

            x_train = train[cols].to_numpy("float32", copy=True)
            y_train = train["label"].to_numpy("int8", copy=True)
            x_valid = valid[cols].to_numpy("float32", copy=True)
            y_valid = valid["label"].to_numpy("int8", copy=True)
            x_test = test[cols].to_numpy("float32", copy=True)
            y_test = test["label"].to_numpy("int8", copy=True)

            model = lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                random_state=int(spec["seed"]),
                n_jobs=1,
                verbose=-1,
                **params,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            p_valid = model.predict_proba(x_valid)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            valid_metrics = _metric_row("valid", y_valid, p_valid)
            test_metrics = _metric_row("test", y_test, p_test)
            valid_curve = _yearly_threshold_table(valid, p_valid, DEFAULT_THRESHOLD_GRID)
            test_curve = _yearly_threshold_table(test, p_test, DEFAULT_THRESHOLD_GRID)
            threshold = _select_threshold(valid_curve)
            test_total = _total_at_threshold(test_curve, threshold)
            objective = _score_objective(test_total, test_curve, threshold, test_metrics["auc"])

            child_dir = model_artifact_dir / name
            child_dir.mkdir(parents=True, exist_ok=True)
            imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            imp_path = child_dir / f"{name}_lgbm_importance.csv"
            valid_curve_path = child_dir / f"{name}_threshold_valid.csv"
            test_curve_path = child_dir / f"{name}_threshold_test.csv"
            metrics_path = child_dir / f"{name}_metrics.json"
            bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_{name}_bundle.joblib"
            imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
            chart_paths = []
            chart_paths += plot_model_diagnostics(y_valid, p_valid, child_dir, f"{name}_valid")
            chart_paths += plot_model_diagnostics(y_test, p_test, child_dir, f"{name}_test")
            chart_paths.append(plot_threshold_curve(valid_curve, child_dir / f"{name}_threshold_valid.png", f"{name} valid threshold"))
            chart_paths.append(plot_threshold_curve(test_curve, child_dir / f"{name}_threshold_test.png", f"{name} test threshold"))
            chart_paths.append(_plot_importance(imp, child_dir / f"{name}_lgbm_importance_top25.png", f"{name} LightGBM importance"))
            if shap_rows > 0:
                chart_paths += _plot_shap(model, x_valid, cols, child_dir, name, shap_rows, int(spec["seed"]))

            joblib.dump(
                {
                    "model": model,
                    "model_type": "LightGBM",
                    "feature_columns": cols,
                    "feature_family": FEATURE_FAMILY,
                    "feature_set": spec["feature_set"],
                    "gate": gate,
                    "gate_quantiles": gate_quantiles,
                    "selected_threshold": threshold,
                    "params": params,
                    "scaling": scaling_contract,
                },
                bundle_path,
                compress=3,
            )
            row = {
                "model_name": name,
                "feature_set": spec["feature_set"],
                "gate": gate,
                "n_features": len(cols),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "test_rows": int(len(test)),
                "selected_threshold": threshold,
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
            metrics_path.write_text(json.dumps({"valid": valid_metrics, "test": test_metrics, "test_total": test_total, "row": row}, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_rows.append(row)
            pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False).to_csv(out_dir / "summary_checkpoint.csv", index=False, encoding="utf-8-sig")
            print({"phase": "model_done", "name": name, "valid_auc": round(row["valid_auc"], 5), "test_auc": round(row["test_auc"], 5), "test_wr": row["test_total_wr"], "deals": row["test_total_resolved"]}, flush=True)

            if mlflow:
                with mlflow.start_run(run_name=f"{name}_{run_id}", nested=True):
                    mlflow.log_params(
                        {
                            "model_name": name,
                            "model_type": "LightGBM",
                            "feature_set": spec["feature_set"],
                            "gate": gate,
                            "n_features": len(cols),
                            "threshold": threshold,
                            **params,
                        }
                    )
                    for key, value in valid_metrics.items():
                        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                            mlflow.log_metric(f"valid_{key}", value)
                    for key, value in test_metrics.items():
                        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                            mlflow.log_metric(f"test_{key}", value)
                    mlflow.log_metric("test_total_wr", test_total["winrate"])
                    mlflow.log_metric("test_total_resolved", test_total["resolved"])
                    mlflow.log_metric("test_min_year_deals", test_total["min_year_deals"])
                    mlflow.log_metric("objective_score", objective)
                    for path in [imp_path, valid_curve_path, test_curve_path, metrics_path, bundle_path, *chart_paths]:
                        if Path(path).exists():
                            mlflow.log_artifact(str(path), artifact_path=f"models/{name}")

            del train, valid, test, x_train, y_train, x_valid, y_valid, x_test, y_test, p_valid, p_test, model
            gc.collect()

        summary = pd.DataFrame(summary_rows)
        if "objective_score" in summary.columns:
            summary = summary.sort_values("objective_score", ascending=False)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        summary_json = {
            "run_id": run_id,
            "feature_family": FEATURE_FAMILY,
            "cache_family": CACHE_FAMILY,
            "model_type": "LightGBM",
            "scaler": "rolling_zscore",
            "n_models": n_models,
            "max_train_rows": max_train_rows,
            "best_model": summary.iloc[0].to_dict() if len(summary) else {},
            "out_dir": str(out_dir),
            "elapsed_sec": round(time.time() - started, 2),
        }
        summary_json_path = out_dir / "summary.json"
        summary_json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
        if mlflow:
            for path in [summary_path, summary_json_path, out_dir / "summary_checkpoint.csv"]:
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="summary")
        print(json.dumps(summary_json, ensure_ascii=False, indent=2), flush=True)
        return summary_json
    finally:
        if parent is not None:
            mlflow.end_run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=24)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=73191)
    parser.add_argument("--shap-rows", type=int, default=120)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--no-edge-scan", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()
    run(
        n_models=args.n_models,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        shap_rows=args.shap_rows,
        force_cache=args.force_cache,
        run_edge_scan=not args.no_edge_scan,
        enable_mlflow=args.enable_mlflow,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
