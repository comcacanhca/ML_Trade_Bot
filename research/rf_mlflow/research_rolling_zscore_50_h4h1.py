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
import sklearn
from numba import njit
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_curve, roc_curve

from config import CFG, PROJECT_ROOT, ensure_dirs

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    build_cache,
)


FEATURE_FAMILY = "rolling_zscore_50_h4h1"
ROLLING_WINDOW = 500
ROLLING_MIN_PERIODS = 100
PASSTHROUGH_FEATURES = {
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "session_london_ny",
    "session_ny",
    "session_asia",
    "sig_bb_reversion_buy",
    "sig_pullback_trend_buy",
    # "sig_momentum_buy",
    # "buy_signal",
}
BASELINE = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]#, "session_ny"]
SIGNALS = ["sig_bb_reversion_buy", "sig_pullback_trend_buy"]
DROP_DEFAULT: set[str] = set()  # e.g. {"m5_volatility_12", "ret_mean_50"}
PREDICT_BATCH_SIZE = 20000
PREDICTION_META_COLS = ["dates", "candle_index", "entry_index", "entry_price", "label", "year", "session_london_ny"]
DEFAULT_CACHE_FAMILY = "initial_non_bb_candidates"


@njit(cache=True)
def _rolling_zscore_past_numba(values: np.ndarray, window: int, min_periods: int, eps: float) -> np.ndarray:
    n = len(values)
    out = np.empty(n, dtype=np.float32)
    sum_v = 0.0
    sumsq_v = 0.0
    count = 0
    for i in range(n):
        if i > 0:
            add = values[i - 1]
            if np.isfinite(add):
                sum_v += add
                sumsq_v += add * add
                count += 1
        remove_idx = i - window - 1
        if remove_idx >= 0:
            remove = values[remove_idx]
            if np.isfinite(remove):
                sum_v -= remove
                sumsq_v -= remove * remove
                count -= 1
        current = values[i]
        if (not np.isfinite(current)) or count < min_periods:
            out[i] = 0.0
            continue
        mean = sum_v / count
        if count > 1:
            var = (sumsq_v - (sum_v * sum_v / count)) / (count - 1)
        else:
            var = 0.0
        if var <= eps:
            out[i] = 0.0
        else:
            out[i] = np.float32((current - mean) / np.sqrt(var))
    return out


def _available_feature_columns(all_cols: list[str]) -> list[str]:
    return [c for c in all_cols if c not in META_COLS]


def _present(cols: set[str], names: list[str]) -> list[str]:
    return [c for c in names if c in cols]


def _feature_groups(cols: set[str]) -> dict[str, list[str]]:
    return {
        "h4_volatility": _present(cols, ["h4_volatility_3", "h4_volatility_6", "h4_volatility_12", "h4_range"]),
        "h4_momentum": _present(cols, ["h4_ret", "h4_ret_sum_3", "h4_ret_sum_6", "h4_ret_sum_12"]),
        "h4_rsi": _present(cols, ["h4_rsi_14"]),
        "h4_ema_dist": _present(cols, ["h4_ema_dist_8", "h4_ema_dist_21", "h4_ema_dist_55"]),
        "h4_ema_slope": _present(cols, ["h4_ema_slope_8", "h4_ema_slope_21", "h4_ema_slope_55"]),
        "h1_context": _present(
            cols,
            [
                "h1_ret",
                "h1_ret_sum_3",
                "h1_ret_sum_6",
                "h1_ret_sum_12",
                "h1_volatility_3",
                "h1_volatility_6",
                "h1_volatility_12",
                "h1_ema_dist_8",
                "h1_ema_dist_21",
                "h1_ema_dist_55",
                "h1_ema_slope_21",
                "h1_ema_slope_55",
                "h1_rsi_14",
            ],
        ),
        "m15_context": _present(
            cols,
            [
                "m15_ret",
                "m15_ret_sum_3",
                "m15_ret_sum_6",
                "m15_ret_sum_12",
                "m15_volatility_3",
                "m15_volatility_6",
                "m15_volatility_12",
                "m15_ema_dist_21",
                "m15_ema_dist_55",
                "m15_rsi_14",
            ],
        ),
        "m1_rsi_atr": _present(cols, ["rsi_14", "rsi_50", "rsi_gap_14_50", "atr14", "volatility_20", "volatility_100", "volatility_200"]),
        "m1_momentum": _present(cols, ["ret_1", "ret_sum_5", "ret_sum_10", "ret_sum_20", "ret_sum_50", "ret_mean_5", "ret_mean_10", "ret_mean_20"]),
        "candle_lags": [c for c in cols if c.startswith(("close_diff_lag_", "body_lag_", "range_lag_"))],
        "candle_shape": _present(cols, ["hl_range", "upper_wick", "lower_wick", "body_abs", "close_pos_range"]),
        "featuretools_auto": [c for c in cols if c.startswith("ft_")],
    }


def _score_objective_stable(test_total: dict, test_curve: pd.DataFrame, threshold: float, test_auc: float) -> float:
    rows = test_curve[test_curve["threshold"] == threshold].copy()
    year_wr = rows["winrate"].replace(0.0, np.nan)
    min_year_wr = float(year_wr.min()) if year_wr.notna().any() else 0.0
    resolved = int(test_total["resolved"])
    min_year_deals = int(test_total["min_year_deals"])
    wr = float(test_total["winrate"])

    deal_penalty = 0.0
    if resolved < 7000:
        deal_penalty += (7000 - resolved) / 220.0
    if resolved > 12000:
        deal_penalty += (resolved - 12000) / 500.0
    if min_year_deals < 1000:
        deal_penalty += (1000 - min_year_deals) / 55.0
    if min_year_wr < 55.0:
        deal_penalty += (55.0 - min_year_wr) * 1.15

    return wr + (float(test_auc) - 0.5) * 100.0 - deal_penalty


def _load_feature_pool(path: str | None, all_cols: list[str]) -> set[str] | None:
    if not path:
        return None
    pool_path = Path(path)
    values = json.loads(pool_path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        values = values.get("selected_features", values.get("features", []))
    if not isinstance(values, list):
        raise ValueError(f"feature pool must be a JSON list or object containing selected_features/features: {path}")
    all_available = set(_available_feature_columns(all_cols)) - DROP_DEFAULT
    pool = {str(v) for v in values if str(v) in all_available}
    pool.update(c for c in BASELINE + SIGNALS if c in all_available)
    if not pool:
        raise ValueError(f"feature pool has no usable features: {path}")
    return pool


def _cache_year_path(cache_family: str, year: int) -> Path:
    return CFG.cache_dir / cache_family / f"candidates_{year}.parquet"


def _load_cache_columns(cache_family: str, force_cache: bool) -> list[str]:
    if cache_family == DEFAULT_CACHE_FAMILY:
        return build_cache(force_cache)
    cache_dir = CFG.cache_dir / cache_family
    cols_path = cache_dir / "feature_columns.json"
    if not cols_path.exists():
        raise FileNotFoundError(
            f"Missing {cols_path}. Build this cache family first or use --cache-family {DEFAULT_CACHE_FAMILY}."
        )
    return json.loads(cols_path.read_text(encoding="utf-8"))


def _make_specs(all_cols: list[str], n_models: int, seed: int, feature_pool: set[str] | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    available = set(_available_feature_columns(all_cols)) - DROP_DEFAULT
    if feature_pool is not None:
        available &= feature_pool
    groups = _feature_groups(available)
    required_core = _present(
        available,
        [
            "h4_volatility_6",
            "h4_volatility_3",
            "h4_ret_sum_12",
            "m15_volatility_12",
            "rsi_gap_14_50",
            "atr14",
        ],
    )
    strong_h4 = _present(
        available,
        [
            "h4_volatility_12",
            "h4_rsi_14",
            "h4_ret_sum_6",
            "h4_ema_dist_55",
            "h4_ema_dist_8",
            "h4_ema_slope_21",
        ],
    )
    strong_h1 = _present(available, ["h1_ema_dist_55", "h1_volatility_6", "h1_volatility_12", "h1_rsi_14", "h1_ret_sum_3"])
    candidate_groups = [g for g, values in groups.items() if values]

    specs: list[dict] = []
    def pick(options):
        value = rng.choice(np.array(options, dtype=object))
        return value.item() if hasattr(value, "item") else value

    templates = [
        ("core_fixed", required_core),
        ("core_h4_strong", required_core + strong_h4),
        ("core_h4_h1", required_core + strong_h4 + strong_h1),
        ("h4_only", groups["h4_volatility"] + groups["h4_momentum"] + groups["h4_rsi"] + groups["h4_ema_dist"]),
        ("h4_no_slope", required_core + groups["h4_volatility"] + groups["h4_momentum"] + groups["h4_rsi"] + groups["h4_ema_dist"]),
        ("h4_h1_m15", required_core + strong_h4 + strong_h1 + groups["m15_context"]),
        ("current_drop3_like", required_core + _present(available, ["h4_ret_sum_12"])),
        ("volatility_stack", groups["h4_volatility"] + groups["h1_context"][:7] + groups["m15_context"][:7] + groups["m1_rsi_atr"]),
        ("momentum_stack", groups["h4_momentum"] + groups["h1_context"][:4] + groups["m1_momentum"] + groups["candle_lags"][:10]),
        ("balanced_small", required_core + strong_h4[:4] + strong_h1[:3] + groups["candle_shape"]),
        ("featuretools_auto", required_core + groups["featuretools_auto"][:12]),
    ]

    for i in range(n_models):
        template_name, template_cols = templates[i % len(templates)]
        selected = list(BASELINE) + list(SIGNALS) + list(template_cols)

        if candidate_groups:
            n_extra_groups = int(rng.integers(1, min(5, len(candidate_groups)) + 1))
            chosen_groups = rng.choice(candidate_groups, size=n_extra_groups, replace=False).tolist()
        else:
            chosen_groups = []
        for group in chosen_groups:
            pool = [c for c in groups[group] if c not in selected and c in available]
            if not pool:
                continue
            if group.startswith("h4"):
                max_k = min(len(pool), 5)
                min_k = min(2, max_k)
            else:
                max_k = min(len(pool), 4)
                min_k = 1
            k = int(rng.integers(min_k, max_k + 1))
            selected += rng.choice(pool, size=k, replace=False).tolist()

        selected = [c for c in dict.fromkeys(selected) if c in available]
        params = {
            "n_estimators": int(pick([70, 80, 90, 110])),
            "max_depth": pick([8, 10, 12, None]),
            "min_samples_leaf": int(pick([250, 350, 500, 700])),
            "min_samples_split": int(pick([700, 900, 1200, 1600])),
            "max_features": pick(["sqrt", "log2", 0.45, 0.65]),
            "max_samples": float(pick([0.65, 0.75, 0.85])),
            "class_weight": pick(["balanced_subsample", None]),
        }
        specs.append(
            {
                "model_name": f"rf_rolling_zscore_{i + 1:02d}",
                "template": template_name,
                "groups": sorted(set([template_name, *chosen_groups])),
                "features": selected,
                "params": params,
                "seed": seed + i + 1,
            }
        )
    return specs


def _rolling_zscore_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = frame
    for col in feature_columns:
        if col in PASSTHROUGH_FEATURES:
            values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float32", copy=False)
            out[col] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            continue
        scaled = _rolling_zscore_past_numba(
            out[col].to_numpy("float32", copy=False),
            ROLLING_WINDOW,
            ROLLING_MIN_PERIODS,
            1e-12,
        )
        out[col] = scaled
    return out


def _predict_proba_batched(model: RandomForestClassifier, x: np.ndarray, batch_size: int = PREDICT_BATCH_SIZE) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        out[start:stop] = model.predict_proba(x[start:stop])[:, 1].astype("float32", copy=False)
    return out


def _safe_logloss(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, prob, labels=[0, 1]))
    except Exception:
        return float("nan")


def _extra_probability_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(prob)
    if not finite.any():
        return {
            "logloss": float("nan"),
            "prob_mean": float("nan"),
            "prob_std": float("nan"),
            "prob_p50": float("nan"),
            "prob_p90": float("nan"),
            "prob_p95": float("nan"),
            "prob_p99": float("nan"),
        }
    p = prob[finite]
    y = y_true[finite]
    return {
        "logloss": _safe_logloss(y, p),
        "prob_mean": float(np.mean(p)),
        "prob_std": float(np.std(p)),
        "prob_p50": float(np.quantile(p, 0.50)),
        "prob_p90": float(np.quantile(p, 0.90)),
        "prob_p95": float(np.quantile(p, 0.95)),
        "prob_p99": float(np.quantile(p, 0.99)),
    }


def _prediction_frame(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    cols = [c for c in PREDICTION_META_COLS if c in frame.columns]
    out = frame[cols].copy()
    out["prob"] = prob.astype("float32", copy=False)
    out["threshold"] = np.float32(threshold)
    out["pred"] = (out["prob"].to_numpy() >= threshold).astype("int8")
    if "dates" in out.columns:
        dt = pd.to_datetime(out["dates"])
        out["month"] = dt.dt.to_period("M").astype(str)
    return out


def _yearly_monthly_tables(pred: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_mask = pred["prob"].to_numpy(copy=False) >= threshold

    def aggregate(group_cols: list[str]) -> pd.DataFrame:
        rows = []
        group_source = pred[group_cols].copy()
        group_source["_row_id"] = np.arange(len(pred), dtype=np.int64)
        for keys, grp in group_source.groupby(group_cols, dropna=False, sort=True):
            idx = grp["_row_id"].to_numpy(copy=False)
            selected_idx = idx[selected_mask[idx]]
            resolved = int(len(selected_idx))
            wins = int(pred["label"].to_numpy(copy=False)[selected_idx].sum()) if resolved else 0
            wr = float(wins / resolved * 100.0) if resolved else 0.0
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(group_cols, keys)}
            row.update({"threshold": threshold, "resolved": resolved, "wins": wins, "winrate": wr})
            rows.append(row)
        return pd.DataFrame(rows)

    yearly = aggregate(["year"]) if "year" in pred.columns else pd.DataFrame()
    monthly = aggregate(["month"]) if "month" in pred.columns else pd.DataFrame()
    return yearly, monthly


def _save_fig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_threshold_tradeoff(curve: pd.DataFrame, path: Path, title: str) -> Path:
    grouped = curve.groupby("threshold", as_index=False).agg({"winrate": "mean", "resolved": "sum"})
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(grouped["threshold"], grouped["winrate"], marker="o", color="#1f77b4")
    ax1.set_ylabel("Winrate (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Threshold")
    ax2 = ax1.twinx()
    ax2.bar(grouped["threshold"], grouped["resolved"], width=0.006, alpha=0.25, color="#ff7f0e")
    ax2.set_ylabel("Resolved deals", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    return _save_fig(path)


def _plot_roc_pr(y_true: np.ndarray, prob: np.ndarray, roc_path: Path, pr_path: Path, title: str) -> list[Path]:
    paths: list[Path] = []
    if len(np.unique(y_true)) < 2:
        return paths
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} ROC")
    plt.grid(True, alpha=0.25)
    paths.append(_save_fig(roc_path))

    precision, recall, _ = precision_recall_curve(y_true, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} Precision-Recall")
    plt.grid(True, alpha=0.25)
    paths.append(_save_fig(pr_path))
    return paths


def _plot_calibration(y_true: np.ndarray, prob: np.ndarray, path: Path, title: str) -> Path | None:
    if len(np.unique(y_true)) < 2:
        return None
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed win rate")
    plt.title(f"{title} calibration")
    plt.grid(True, alpha=0.25)
    return _save_fig(path)


def _plot_prob_distribution(p_valid: np.ndarray, p_test: np.ndarray, path: Path, title: str) -> Path:
    plt.figure(figsize=(8, 5))
    plt.hist(p_valid, bins=50, alpha=0.55, density=True, label="valid")
    plt.hist(p_test, bins=50, alpha=0.55, density=True, label="test")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    return _save_fig(path)


def _plot_confusion(y_true: np.ndarray, prob: np.ndarray, threshold: float, path: Path, title: str) -> Path:
    pred = (prob >= threshold).astype("int8")
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    return _save_fig(path)


def _plot_period_winrate(table: pd.DataFrame, period_col: str, path: Path, title: str) -> Path | None:
    if table.empty or period_col not in table.columns:
        return None
    data = table.sort_values(period_col)
    plt.figure(figsize=(max(7, min(16, len(data) * 0.35)), 5))
    plt.bar(data[period_col].astype(str), data["winrate"], alpha=0.75)
    plt.axhline(50.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Winrate (%)")
    plt.xlabel(period_col)
    plt.title(title)
    plt.xticks(rotation=70 if len(data) > 8 else 0, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return _save_fig(path)


def _make_diagnostic_artifacts(
    model_dir: Path,
    name: str,
    valid_all: pd.DataFrame,
    p_valid: np.ndarray,
    y_valid: np.ndarray,
    test_pred: pd.DataFrame,
    p_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    valid_curve: pd.DataFrame,
    test_curve: pd.DataFrame,
    save_predictions: bool,
    log_diagnostics: bool,
) -> list[Path]:
    paths: list[Path] = []
    valid_pred = _prediction_frame(valid_all, p_valid, threshold)
    yearly_test, monthly_test = _yearly_monthly_tables(test_pred, threshold)
    yearly_path = model_dir / f"{name}_yearly_test.csv"
    monthly_path = model_dir / f"{name}_monthly_test.csv"
    yearly_test.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    monthly_test.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    paths.extend([yearly_path, monthly_path])

    if save_predictions:
        valid_pred_path = model_dir / f"{name}_valid_predictions.parquet"
        test_pred_path = model_dir / f"{name}_test_predictions.parquet"
        valid_pred.to_parquet(valid_pred_path, index=False, compression="zstd")
        test_pred.to_parquet(test_pred_path, index=False, compression="zstd")
        paths.extend([valid_pred_path, test_pred_path])

    if log_diagnostics:
        chart_paths: list[Path | None] = []
        chart_paths.append(_plot_threshold_tradeoff(valid_curve, model_dir / f"{name}_threshold_valid.png", f"{name} valid threshold tradeoff"))
        chart_paths.append(_plot_threshold_tradeoff(test_curve, model_dir / f"{name}_threshold_test.png", f"{name} test threshold tradeoff"))
        chart_paths.extend(_plot_roc_pr(y_valid, p_valid, model_dir / f"{name}_roc_valid.png", model_dir / f"{name}_pr_valid.png", f"{name} valid"))
        chart_paths.extend(_plot_roc_pr(y_test, p_test, model_dir / f"{name}_roc_test.png", model_dir / f"{name}_pr_test.png", f"{name} test"))
        chart_paths.append(_plot_calibration(y_valid, p_valid, model_dir / f"{name}_calibration_valid.png", f"{name} valid"))
        chart_paths.append(_plot_calibration(y_test, p_test, model_dir / f"{name}_calibration_test.png", f"{name} test"))
        chart_paths.append(_plot_prob_distribution(p_valid, p_test, model_dir / f"{name}_prob_distribution_valid_test.png", f"{name} probability distribution"))
        chart_paths.append(_plot_confusion(y_test, p_test, threshold, model_dir / f"{name}_confusion_test.png", f"{name} test confusion @ {threshold:.2f}"))
        chart_paths.append(_plot_period_winrate(yearly_test, "year", model_dir / f"{name}_yearly_test_selected_threshold.png", f"{name} yearly test WR @ {threshold:.2f}"))
        chart_paths.append(_plot_period_winrate(monthly_test, "month", model_dir / f"{name}_monthly_test_selected_threshold.png", f"{name} monthly test WR @ {threshold:.2f}"))
        paths.extend([p for p in chart_paths if p is not None])

    return paths


def _load_scaled_train_valid(feature_columns: list[str], max_train_rows: int, seed: int, cache_family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    read_cols = list(dict.fromkeys(feature_columns + PREDICTION_META_COLS))
    def load_many(years: tuple[int, ...], sample_rows: int | None = None) -> pd.DataFrame:
        frames = []
        for year in years:
            frame = pd.read_parquet(_cache_year_path(cache_family, year), columns=read_cols)
            frame = _rolling_zscore_frame(frame, feature_columns)
            if sample_rows is not None:
                frame = _sample_train(frame, sample_rows, seed + int(year))
            frames.append(frame)
            gc.collect()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True, copy=False)

    per_train_year = max(1000, int(np.ceil(max_train_rows / max(1, len(CFG.split.train_years)))))
    train = _sample_train(load_many(CFG.split.train_years, per_train_year), max_train_rows, seed)
    return train, load_many(CFG.split.valid_years)


def _iter_scaled_years(years: tuple[int, ...], feature_columns: list[str], cache_family: str):
    read_cols = list(dict.fromkeys(feature_columns + PREDICTION_META_COLS))
    for year in years:
        frame = pd.read_parquet(_cache_year_path(cache_family, year), columns=read_cols)
        yield year, _rolling_zscore_frame(frame, feature_columns)


def run(
    n_models: int,
    max_train_rows: int,
    seed: int,
    force_cache: bool,
    run_id: int | None = None,
    only_index: int | None = None,
    enable_mlflow: bool = False,
    save_predictions: bool = False,
    log_diagnostics: bool = False,
    shap_rows: int = 0,
    parent_run_id: str | None = None,
    feature_pool_path: str | None = None,
    cache_family: str = DEFAULT_CACHE_FAMILY,
) -> dict:
    ensure_dirs()
    run_id = int(run_id or time.time())
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    explain_dir = out_dir / "explainability"
    chart_dir = out_dir / "charts"
    explain_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    all_cols = _load_cache_columns(cache_family, force_cache)
    feature_pool = _load_feature_pool(feature_pool_path, all_cols)
    specs = _make_specs(all_cols, n_models, seed, feature_pool)
    if only_index is not None:
        if only_index < 1 or only_index > len(specs):
            raise ValueError(f"only_index must be in [1, {len(specs)}], got {only_index}")
        specs = [specs[only_index - 1]]
    all_features = sorted({feature for spec in specs for feature in spec["features"]})
    passthrough = [c for c in all_features if c in PASSTHROUGH_FEATURES]
    rolling_scaled = [c for c in all_features if c not in PASSTHROUGH_FEATURES]

    (out_dir / "model_specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "feature_scaling.json").write_text(
        json.dumps(
            {
                "scaling": "rolling_zscore",
                "window": ROLLING_WINDOW,
                "min_periods": ROLLING_MIN_PERIODS,
                "use_past_only": True,
                "drop_default": sorted(DROP_DEFAULT),
                "rolling_scaled_features": rolling_scaled,
                "passthrough_features": passthrough,
                "save_predictions": save_predictions,
                "log_diagnostics": log_diagnostics,
                "shap_rows": shap_rows,
                "feature_pool_path": feature_pool_path,
                "feature_pool_size": len(feature_pool) if feature_pool is not None else None,
                "cache_family": cache_family,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print({"phase": "start_memory_safe", "features_total": len(all_features), "models": len(specs)}, flush=True)
    mlflow = _mlflow() if enable_mlflow else None
    parent = None
    if mlflow and not parent_run_id:
        parent = mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}")
    summary_path = out_dir / "summary.csv"
    existing_rows: list[dict] = []
    if summary_path.exists():
        existing_rows = pd.read_csv(summary_path).to_dict("records")
        existing_names = {str(row.get("model_name")) for row in existing_rows}
        specs = [spec for spec in specs if spec["model_name"] not in existing_names]
    rows: list[dict] = existing_rows.copy()
    try:
        for idx, spec in enumerate(specs, start=1):
            name = spec["model_name"]
            cols = spec["features"]
            print({"phase": "model", "idx": idx, "n_models": len(specs), "name": name, "features": len(cols)}, flush=True)
            model_dir = explain_dir / name
            model_dir.mkdir(parents=True, exist_ok=True)

            print({"phase": "load_scale_model_splits", "name": name}, flush=True)
            train, valid_all = _load_scaled_train_valid(cols, max_train_rows, int(spec["seed"]), cache_family)
            x_train = train[cols].to_numpy("float32", copy=True)
            y_train = train["label"].to_numpy("int8", copy=True)
            x_valid = valid_all[cols].to_numpy("float32", copy=True)
            y_valid = valid_all["label"].to_numpy("int8", copy=True)

            model = RandomForestClassifier(**spec["params"], bootstrap=True, random_state=int(spec["seed"]), n_jobs=1)
            model.fit(x_train, y_train)
            p_valid = _predict_proba_batched(model, x_valid)

            valid_metrics = _metrics(y_valid, p_valid)
            valid_curve = _threshold_table(valid_all[["label", "year", "session_london_ny"]], p_valid, CFG.threshold_grid)
            threshold = _select_threshold(valid_curve)

            test_curves = []
            y_test_parts = []
            p_test_parts = []
            test_pred_parts = []
            for _, test_year in _iter_scaled_years(CFG.split.test_years, cols, cache_family):
                x_test_year = test_year[cols].to_numpy("float32", copy=True)
                y_test_year = test_year["label"].to_numpy("int8", copy=True)
                p_test_year = _predict_proba_batched(model, x_test_year)
                test_curves.append(_threshold_table(test_year[["label", "year", "session_london_ny"]], p_test_year, CFG.threshold_grid))
                test_pred_parts.append(_prediction_frame(test_year, p_test_year, threshold))
                y_test_parts.append(y_test_year)
                p_test_parts.append(p_test_year)
                del x_test_year, y_test_year, p_test_year, test_year
                gc.collect()
            y_test = np.concatenate(y_test_parts)
            p_test = np.concatenate(p_test_parts)
            test_pred = pd.concat(test_pred_parts, ignore_index=True, copy=False)
            test_metrics = _metrics(y_test, p_test)
            valid_metrics.update(_extra_probability_metrics(y_valid, p_valid))
            test_metrics.update(_extra_probability_metrics(y_test, p_test))
            test_curve = pd.concat(test_curves, ignore_index=True, copy=False)
            test_total = _total_at_threshold(test_curve, threshold)
            objective = _score_objective_stable(test_total, test_curve, threshold, test_metrics["auc"])

            imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            imp_path = model_dir / f"{name}_rf_importance.csv"
            valid_curve_path = model_dir / f"{name}_threshold_valid.csv"
            test_curve_path = model_dir / f"{name}_threshold_test.csv"
            imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
            importance_chart = _plot_importance(imp, model_dir / f"{name}_rf_importance_top25.png", f"{name} RF importance")
            diagnostic_paths = _make_diagnostic_artifacts(
                model_dir=model_dir,
                name=name,
                valid_all=valid_all,
                p_valid=p_valid,
                y_valid=y_valid,
                test_pred=test_pred,
                p_test=p_test,
                y_test=y_test,
                threshold=threshold,
                valid_curve=valid_curve,
                test_curve=test_curve,
                save_predictions=save_predictions,
                log_diagnostics=log_diagnostics,
            )
            shap_paths: list[Path] = []
            if shap_rows > 0:
                print({"phase": "shap", "name": name, "rows": shap_rows}, flush=True)
                shap_paths = _plot_shap(model, x_valid, cols, model_dir, name, shap_rows, int(spec["seed"]))

            bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_{name}_bundle.joblib"
            joblib.dump(
                {
                    "model": model,
                    "scaler": {
                        "name": "rolling_zscore",
                        "window": ROLLING_WINDOW,
                        "min_periods": ROLLING_MIN_PERIODS,
                        "use_past_only": True,
                        "passthrough_features": passthrough,
                    },
                    "feature_columns": cols,
                    "feature_family": FEATURE_FAMILY,
                    "selected_threshold": threshold,
                    "params": spec["params"],
                    "template": spec["template"],
                    "groups": spec["groups"],
                },
                bundle_path,
                compress=3,
            )

            row = {
                "model_name": name,
                "template": spec["template"],
                "groups": "|".join(spec["groups"]),
                "n_features": len(cols),
                    "threshold": threshold,
                    "valid_auc": valid_metrics["auc"],
                    "valid_brier": valid_metrics["brier"],
                    "valid_logloss": valid_metrics["logloss"],
                    "test_auc": test_metrics["auc"],
                    "test_brier": test_metrics["brier"],
                    "test_logloss": test_metrics["logloss"],
                    "auc_gap_valid_test": float(valid_metrics["auc"] - test_metrics["auc"]),
                    "test_total_resolved": test_total["resolved"],
                    "test_total_wr": test_total["winrate"],
                    "test_min_year_deals": test_total["min_year_deals"],
                "objective_score": objective,
                "top10_features": "|".join(imp.head(10)["feature"].tolist()),
                "model_path": str(bundle_path),
            }
            rows.append(row)
            current_summary = pd.DataFrame(rows).sort_values("objective_score", ascending=False)
            current_summary.to_csv(out_dir / "summary_checkpoint.csv", index=False, encoding="utf-8-sig")
            current_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
            (out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "feature_family": FEATURE_FAMILY,
                        "rows": current_summary.to_dict("records"),
                        "best": current_summary.iloc[0].to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if mlflow:
                child_tags = {"research_run_id": str(run_id), "feature_family": FEATURE_FAMILY}
                if parent_run_id:
                    child_tags["mlflow.parentRunId"] = parent_run_id
                with mlflow.start_run(run_name=f"{name}_{run_id}", nested=not bool(parent_run_id), tags=child_tags):
                    mlflow.log_params(
                        {
                            "model_name": name,
                            "n_features": len(cols),
                            "scaling": "rolling_zscore",
                            "shap_rows": shap_rows,
                            "feature_pool_path": feature_pool_path or "",
                            "cache_family": cache_family,
                            **spec["params"],
                            "threshold": threshold,
                        }
                    )
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_{k}", v)
                    mlflow.log_metric("test_total_wr", test_total["winrate"])
                    mlflow.log_metric("test_total_resolved", test_total["resolved"])
                    mlflow.log_metric("test_min_year_deals", test_total["min_year_deals"])
                    mlflow.log_metric("objective_score", objective)
                    for path in [imp_path, valid_curve_path, test_curve_path, importance_chart, bundle_path, *diagnostic_paths, *shap_paths]:
                        mlflow.log_artifact(str(path), artifact_path=f"models/{name}")

            del valid_all, train, x_train, y_train, x_valid, y_valid, y_test, p_test, y_test_parts, p_test_parts, test_pred_parts, test_pred, test_curves, model, p_valid, shap_paths
            gc.collect()

        summary = pd.DataFrame(rows).sort_values("objective_score", ascending=False)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        (out_dir / "summary.json").write_text(
            json.dumps({"run_id": run_id, "feature_family": FEATURE_FAMILY, "rows": rows, "best": summary.iloc[0].to_dict()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if mlflow:
            summary_artifacts = [summary_path, out_dir / "summary.json", out_dir / "model_specs.json", out_dir / "feature_scaling.json"]
            if parent_run_id:
                with mlflow.start_run(run_id=parent_run_id):
                    for path in summary_artifacts:
                        mlflow.log_artifact(str(path), artifact_path="summary")
            else:
                for path in summary_artifacts:
                    mlflow.log_artifact(str(path), artifact_path="summary")
    finally:
        if mlflow and parent:
            mlflow.end_run()

    best = summary.iloc[0].to_dict()
    print(json.dumps({"out_dir": str(out_dir), "best": best}, ensure_ascii=False, indent=2), flush=True)
    return {"out_dir": str(out_dir), "best": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=50)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=860906)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--only-index", type=int, default=None)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--log-diagnostics", action="store_true")
    parser.add_argument("--shap-rows", type=int, default=0)
    parser.add_argument("--parent-run-id", type=str, default=None)
    parser.add_argument("--feature-pool", type=str, default=None)
    parser.add_argument("--cache-family", type=str, default=DEFAULT_CACHE_FAMILY)
    args = parser.parse_args()
    save_predictions = bool(args.save_predictions or args.enable_mlflow)
    log_diagnostics = bool(args.log_diagnostics or args.enable_mlflow)
    run(
        args.n_models,
        args.max_train_rows,
        args.seed,
        args.force_cache,
        args.run_id,
        args.only_index,
        args.enable_mlflow,
        save_predictions,
        log_diagnostics,
        args.shap_rows,
        args.parent_run_id,
        args.feature_pool,
        args.cache_family,
    )


if __name__ == "__main__":
    main()
