from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from data_io import CLOSE, DATE, HIGH, LOW, OPEN, VOLUME, load_year
from labels import buy_label_arrays
from plots import plot_model_diagnostics, plot_threshold_curve
from research_fs14_overlay_norm_light import RobustClipScalerLite
from method.OverlayIndicatorsNormalization import (
    OverlayNormalizationConfig,
    normalize_overlay_indicator,
    overlay_pct_distance,
)


FEATURE_FAMILY = "random50_initial_features_lgbm_time_fixed"
CACHE_FAMILY = "initial_non_bb_candidates"
POINT = 100.0
TIME_FIXED_FEATURES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "session_london_ny", "session_ny", "session_asia"]
BASELINE = TIME_FIXED_FEATURES
META_COLS = {"dates", "candle_index", "entry_index", "entry_price", "year", "label"}


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _cache_dir() -> Path:
    return CFG.cache_dir / CACHE_FAMILY


def _year_path(year: int) -> Path:
    return _cache_dir() / f"candidates_{year}.parquet"


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _completed_htf_features(base: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    ohlc = (
        base[[DATE, OPEN, HIGH, LOW, CLOSE]]
        .copy()
        .set_index(DATE)
        .resample(rule, label="right", closed="right")
        .agg({OPEN: "first", HIGH: "max", LOW: "min", CLOSE: "last"})
        .dropna()
        .reset_index()
    )
    ohlc[f"{prefix}_ret"] = (ohlc[CLOSE] - ohlc[OPEN]) * POINT
    ohlc[f"{prefix}_range"] = (ohlc[HIGH] - ohlc[LOW]) * POINT
    returns = ohlc[CLOSE].diff() * POINT
    for win in [3, 6, 12]:
        ohlc[f"{prefix}_ret_sum_{win}"] = returns.rolling(win, min_periods=win).sum()
        ohlc[f"{prefix}_volatility_{win}"] = returns.rolling(win, min_periods=win).std()
    for span in [8, 21, 55]:
        ema = ohlc[CLOSE].ewm(span=span, min_periods=span, adjust=False).mean()
        ohlc[f"{prefix}_ema_dist_{span}"] = (ohlc[CLOSE] - ema) * POINT
        ohlc[f"{prefix}_ema_slope_{span}"] = ema.diff() * POINT
    ohlc[f"{prefix}_rsi_14"] = _rsi(ohlc[CLOSE], 14)
    keep = [DATE] + [c for c in ohlc.columns if c.startswith(prefix)]
    return pd.merge_asof(base[[DATE]].sort_values(DATE), ohlc[keep].sort_values(DATE), on=DATE, direction="backward")


def _parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    if len(high) == 0:
        return pd.Series(dtype="float64", index=high.index)
    sar = np.full(len(high), np.nan, dtype="float64")
    high_arr = high.to_numpy(dtype="float64", copy=False)
    low_arr = low.to_numpy(dtype="float64", copy=False)
    trend_up = True
    af = step
    ep = high_arr[0]
    sar[0] = low_arr[0]
    for i in range(1, len(high_arr)):
        prev_sar = sar[i - 1]
        sar[i] = prev_sar + af * (ep - prev_sar)
        if trend_up:
            if i >= 2:
                sar[i] = min(sar[i], low_arr[i - 1], low_arr[i - 2])
            if low_arr[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                ep = low_arr[i]
                af = step
            elif high_arr[i] > ep:
                ep = high_arr[i]
                af = min(af + step, max_step)
        else:
            if i >= 2:
                sar[i] = max(sar[i], high_arr[i - 1], high_arr[i - 2])
            if high_arr[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = high_arr[i]
                af = step
            elif low_arr[i] < ep:
                ep = low_arr[i]
                af = min(af + step, max_step)
    return pd.Series(sar, index=high.index)


def _supertrend_line(high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series, multiplier: float = 3.0) -> pd.Series:
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    line = pd.Series(np.nan, index=close.index, dtype="float64")
    trend_up = True
    for i in range(len(close)):
        if i == 0 or pd.isna(upper.iloc[i - 1]) or pd.isna(lower.iloc[i - 1]):
            line.iloc[i] = lower.iloc[i] if trend_up else upper.iloc[i]
            continue
        if close.iloc[i] > upper.iloc[i - 1]:
            trend_up = True
        elif close.iloc[i] < lower.iloc[i - 1]:
            trend_up = False
        line.iloc[i] = lower.iloc[i] if trend_up else upper.iloc[i]
    return line


def _session_vwap(base: pd.DataFrame) -> pd.Series | None:
    if VOLUME not in base.columns:
        return None
    typical = (base[HIGH] + base[LOW] + base[CLOSE]) / 3
    volume = base[VOLUME].replace(0, np.nan)
    session = pd.to_datetime(base[DATE]).dt.date
    pv_sum = (typical * volume).groupby(session).cumsum()
    vol_sum = volume.groupby(session).cumsum()
    return pv_sum / vol_sum.replace(0, np.nan)


def _normalize_overlay_line(
    feature: dict[str, pd.Series | np.ndarray],
    name: str,
    indicator: pd.Series,
    close: pd.Series,
    atr14: pd.Series,
    norm_cfg: OverlayNormalizationConfig,
    std_window: int,
) -> None:
    norm = normalize_overlay_indicator(
        indicator=indicator,
        close=close,
        atr=atr14,
        name=name,
        config=norm_cfg,
        value_method="atr_gap",
        value_lag_periods=(1, 3, 5, 8, 13),
        distance_lag_periods=(1, 3, 5, 8, 13),
        raw_lag_periods=(),
        velocity_periods=(1, 3, 5, 8),
        velocity_lag_periods=(1, 3),
        acceleration_periods=(1, 3, 5),
        std_window=std_window,
        std_mode="relative_distance_rolling_rank",
    )
    for col in norm.columns:
        feature[col] = norm[col]

    pct_distance = overlay_pct_distance(indicator, close)
    feature[f"{name}_pct_distance"] = pct_distance
    for lag in [1, 3, 5, 8, 13]:
        feature[f"{name}_pct_distance_lag{lag}"] = pct_distance.shift(lag)


def _add_overlay_features(feature: dict[str, pd.Series | np.ndarray], base: pd.DataFrame, atr14: pd.Series) -> None:
    """Add normalized overlay features for MA/band/channel/trend-line indicators.

    These are causal features. Rolling baselines in OverlayNormalizationConfig use
    previous rows only, and velocity/acceleration are backward-looking diffs.
    Ichimoku lines are not forward-shifted, and Chikou is intentionally skipped.
    """

    close = base[CLOSE]
    high = base[HIGH]
    low = base[LOW]
    norm_cfg = OverlayNormalizationConfig(window=500, min_periods=100, use_past_only=True)

    overlay_lines: list[tuple[str, pd.Series, int]] = []

    for span in [8, 13, 21, 34, 55, 89, 144, 200]:
        overlay_lines.append((f"ema{span}_ov", close.ewm(span=span, min_periods=span, adjust=False).mean(), min(span, 50)))

    for period in [20, 50, 100, 200]:
        overlay_lines.append((f"sma{period}_ov", close.rolling(period, min_periods=period).mean(), min(period, 50)))

    for period in [20, 100, 200]:
        prefix = f"bb{period}"
        mid = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        overlay_lines.extend(
            [
                (f"{prefix}_mid_ov", mid, min(period, 50)),
                (f"{prefix}_upper_ov", upper, min(period, 50)),
                (f"{prefix}_lower_ov", lower, min(period, 50)),
            ]
        )

    atr20 = _atr(high, low, close, 20)
    kc_mid = close.ewm(span=20, min_periods=20, adjust=False).mean()
    overlay_lines.extend(
        [
            ("keltner20_mid_ov", kc_mid, 20),
            ("keltner20_upper_ov", kc_mid + 1.5 * atr20, 20),
            ("keltner20_lower_ov", kc_mid - 1.5 * atr20, 20),
        ]
    )

    for period in [20, 55]:
        dc_upper = high.rolling(period, min_periods=period).max()
        dc_lower = low.rolling(period, min_periods=period).min()
        dc_mid = (dc_upper + dc_lower) / 2
        overlay_lines.extend(
            [
                (f"donchian{period}_upper_ov", dc_upper, min(period, 50)),
                (f"donchian{period}_lower_ov", dc_lower, min(period, 50)),
                (f"donchian{period}_mid_ov", dc_mid, min(period, 50)),
            ]
        )

    rolling_vwap = (close * base[VOLUME]).rolling(100, min_periods=100).sum() / base[VOLUME].rolling(100, min_periods=100).sum().replace(0, np.nan) if VOLUME in base.columns else None
    if rolling_vwap is not None:
        overlay_lines.append(("vwap100_ov", rolling_vwap, 50))
    session_vwap = _session_vwap(base)
    if session_vwap is not None:
        overlay_lines.append(("vwap_session_ov", session_vwap, 50))

    tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2
    kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2
    overlay_lines.extend(
        [
            ("ichimoku_tenkan_ov", tenkan, 9),
            ("ichimoku_kijun_ov", kijun, 26),
            ("ichimoku_span_a_raw_ov", span_a_raw, 26),
            ("ichimoku_span_b_raw_ov", span_b_raw, 50),
        ]
    )

    overlay_lines.append(("parabolic_sar_ov", _parabolic_sar(high, low), 50))
    overlay_lines.append(("supertrend20x3_ov", _supertrend_line(high, low, close, atr20, multiplier=3.0), 50))

    for name, indicator, std_window in overlay_lines:
        _normalize_overlay_line(feature, name, indicator, close, atr14, norm_cfg, std_window=std_window)


def _build_year_cache(year: int) -> tuple[dict[str, np.ndarray], list[str]]:
    raw = load_year(year)
    base = raw[[DATE, OPEN, HIGH, LOW, CLOSE] + ([VOLUME] if VOLUME in raw.columns else [])].copy()
    close = base[CLOSE]
    high = base[HIGH]
    low = base[LOW]
    open_ = base[OPEN]

    feature: dict[str, pd.Series | np.ndarray] = {}
    feature["ret_1"] = (close - open_) * POINT
    feature["hl_range"] = (high - low) * POINT
    feature["upper_wick"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) * POINT
    feature["lower_wick"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) * POINT
    feature["body_abs"] = pd.Series(feature["ret_1"]).abs()
    feature["close_pos_range"] = (close - low) / (high - low).replace(0, np.nan)
    feature["atr14"] = _atr(high, low, close, 14)
    returns = close.diff() * POINT

    for lag in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        feature[f"close_diff_lag_{lag}"] = returns.shift(lag)
        feature[f"body_lag_{lag}"] = pd.Series(feature["ret_1"]).shift(lag)
        feature[f"range_lag_{lag}"] = pd.Series(feature["hl_range"]).shift(lag)

    for win in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(win, min_periods=win).mean()
        std = close.rolling(win, min_periods=win).std().replace(0, np.nan)
        feature[f"z_close_{win}"] = (close - ma) / std
        feature[f"range_mean_{win}"] = pd.Series(feature["hl_range"]).rolling(win, min_periods=win).mean()
        feature[f"range_std_{win}"] = pd.Series(feature["hl_range"]).rolling(win, min_periods=win).std()
        feature[f"body_mean_{win}"] = pd.Series(feature["body_abs"]).rolling(win, min_periods=win).mean()
        feature[f"body_std_{win}"] = pd.Series(feature["body_abs"]).rolling(win, min_periods=win).std()
        feature[f"ret_sum_{win}"] = returns.rolling(win, min_periods=win).sum()
        feature[f"ret_mean_{win}"] = returns.rolling(win, min_periods=win).mean()
        feature[f"volatility_{win}"] = returns.rolling(win, min_periods=win).std()

    ema_by_span: dict[int, pd.Series] = {}
    for span in [8, 13, 21, 34, 55, 89, 144]:
        ema = close.ewm(span=span, min_periods=span, adjust=False).mean()
        ema_by_span[span] = ema
        feature[f"ema_dist_{span}"] = (close - ema) * POINT
        feature[f"ema_slope_{span}"] = ema.diff() * POINT
    for fast, slow in [(8, 21), (8, 55), (21, 55), (34, 89), (55, 144)]:
        feature[f"ema_gap_{fast}_{slow}"] = (ema_by_span[fast] - ema_by_span[slow]) * POINT

    feature["rsi_14"] = _rsi(close, 14)
    feature["rsi_50"] = _rsi(close, 50)
    feature["rsi_gap_14_50"] = pd.Series(feature["rsi_14"]) - pd.Series(feature["rsi_50"])
    for lag in [1, 3, 5, 8, 13]:
        feature[f"rsi14_lag_{lag}"] = pd.Series(feature["rsi_14"]).shift(lag)

    _add_overlay_features(feature, base=base, atr14=pd.Series(feature["atr14"]))

    mid20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper20 = mid20 + 2 * std20
    lower20 = mid20 - 2 * std20
    bb20_pos = (close - lower20) / (upper20 - lower20).replace(0, np.nan)
    z_close_20 = feature["z_close_20"]
    ret_sum_10 = feature["ret_sum_10"]

    dt = pd.to_datetime(base[DATE])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    feature["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    feature["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    feature["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 5)
    feature["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 5)
    feature["session_london_ny"] = ((dt.dt.hour >= 7) & (dt.dt.hour <= 21)).astype(np.int8)
    feature["session_ny"] = ((dt.dt.hour >= 12) & (dt.dt.hour <= 21)).astype(np.int8)
    feature["session_asia"] = ((dt.dt.hour >= 0) & (dt.dt.hour < 7)).astype(np.int8)

    if VOLUME in base.columns:
        vol = base[VOLUME]
        feature["volume_z_100"] = (vol - vol.rolling(100, min_periods=100).mean()) / vol.rolling(100, min_periods=100).std().replace(0, np.nan)

    for rule, prefix in [("5min", "m5"), ("15min", "m15"), ("1h", "h1"), ("4h", "h4")]:
        htf = _completed_htf_features(base, rule, prefix)
        for col in htf.columns:
            if col != DATE:
                feature[col] = htf[col]

    sig_bb_reversion_buy = ((bb20_pos < 0.18) & (pd.Series(feature["rsi_14"]) < 45)).astype(np.int8)
    sig_pullback_trend_buy = ((pd.Series(feature["ema_dist_55"]) > 0) & (pd.Series(feature["ema_dist_8"]) < 0) & (pd.Series(feature["ret_1"]) > 0)).astype(np.int8)
    sig_momentum_buy = ((pd.Series(ret_sum_10) > 0) & (pd.Series(z_close_20) > -0.5) & (pd.Series(z_close_20) < 1.5)).astype(np.int8)
    buy_signal = (
        (sig_bb_reversion_buy == 1)
        | (sig_pullback_trend_buy == 1)
        | ((sig_momentum_buy == 1) & (pd.Series(feature["session_london_ny"]) == 1))
    ).astype(np.int8)
    feature["sig_bb_reversion_buy"] = sig_bb_reversion_buy
    feature["sig_pullback_trend_buy"] = sig_pullback_trend_buy
    feature["sig_momentum_buy"] = sig_momentum_buy
    feature["buy_signal"] = buy_signal

    labels, entry_index, entry_price = buy_label_arrays(base)
    pre_mask = (np.asarray(buy_signal, dtype=np.int8) == 1) & ~np.isnan(labels)
    idx = np.flatnonzero(pre_mask)

    out: dict[str, np.ndarray] = {}
    feature_cols = list(feature.keys())
    finite = np.ones(len(idx), dtype=bool)
    for col in feature_cols:
        src = feature[col].to_numpy(copy=False) if isinstance(feature[col], pd.Series) else np.asarray(feature[col])
        values = src[idx]
        if values.dtype.kind == "f":
            values = values.astype("float32", copy=False)
            finite &= np.isfinite(values)
        elif values.dtype.kind in {"i", "u", "b"}:
            values = values.astype("int8" if values.max(initial=0) <= 1 and values.min(initial=0) >= 0 else "float32", copy=False)
        out[col] = values
    if not finite.all():
        for col in list(out.keys()):
            out[col] = out[col][finite]
        idx = idx[finite]

    out[DATE] = base[DATE].iloc[idx].to_numpy()
    out["candle_index"] = idx.astype("int32")
    out["entry_index"] = entry_index[idx].astype("int32")
    out["entry_price"] = entry_price[idx].astype("float32")
    out["year"] = np.full(len(idx), year, dtype="int16")
    out["label"] = labels[idx].astype("int8")
    del raw, base, feature, labels, entry_index, entry_price
    gc.collect()
    return out, feature_cols


def build_cache(force: bool) -> list[str]:
    _cache_dir().mkdir(parents=True, exist_ok=True)
    feature_cols: list[str] | None = None
    for year in CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years:
        path = _year_path(year)
        if path.exists() and not force:
            print({"phase": "cache_exists", "year": year, "path": str(path)}, flush=True)
            continue
        print({"phase": "build_year_cache", "year": year}, flush=True)
        data, cols = _build_year_cache(year)
        feature_cols = feature_cols or cols
        table = pa.Table.from_pydict(data)
        pq.write_table(table, path, compression="zstd")
        print({"phase": "build_year_cache_done", "year": year, "rows": table.num_rows}, flush=True)
        del data, table
        gc.collect()

    cols_path = _cache_dir() / "feature_columns.json"
    if feature_cols is not None:
        cols_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(cols_path.read_text(encoding="utf-8"))


def _load_years(years: tuple[int, ...], columns: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(_year_path(y), columns=columns) for y in years], ignore_index=True)


def _sample_train(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    y = df["label"].to_numpy(np.int8)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(len(pos), max_rows // 2)
    n_neg = min(len(neg), max_rows - n_pos)
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    return df.iloc[idx].reset_index(drop=True)


def _feature_groups(cols: list[str]) -> dict[str, list[str]]:
    return {
        "signal": [c for c in cols if c.startswith("sig_") or c == "buy_signal"],
        "candle": [c for c in cols if c in {"ret_1", "hl_range", "upper_wick", "lower_wick", "body_abs", "close_pos_range", "atr14"}],
        "lags": [c for c in cols if "_lag_" in c or c.startswith("close_diff_lag_") or c.startswith("body_lag_") or c.startswith("range_lag_")],
        "rolling": [c for c in cols if c.startswith(("z_close_", "range_mean_", "range_std_", "body_mean_", "body_std_", "ret_sum_", "ret_mean_", "volatility_"))],
        "ema": [c for c in cols if c.startswith(("ema_dist_", "ema_slope_", "ema_gap_"))],
        "rsi": [c for c in cols if c.startswith("rsi")],
        "mtf_m5": [c for c in cols if c.startswith("m5_")],
        "mtf_m15": [c for c in cols if c.startswith("m15_")],
        "mtf_h1": [c for c in cols if c.startswith("h1_")],
        "mtf_h4": [c for c in cols if c.startswith("h4_")],
        "ma_overlay": [c for c in cols if c.startswith(("ema", "sma")) and "_ov_" in c],
        "bb_overlay": [c for c in cols if c.startswith(("bb20_", "bb100_", "bb200_"))],
        "channel_overlay": [c for c in cols if c.startswith(("keltner", "donchian"))],
        "vwap_overlay": [c for c in cols if c.startswith("vwap")],
        "trendline_overlay": [c for c in cols if c.startswith(("ichimoku", "parabolic_sar", "supertrend"))],
        "volume": [c for c in cols if c.startswith("volume_")],
    }


def _random_feature_sets(cols: list[str], n_models: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    groups = _feature_groups(cols)
    group_names = [g for g, values in groups.items() if values]
    specs = []
    def pick(options):
        value = rng.choice(np.array(options, dtype=object))
        return value.item() if hasattr(value, "item") else value

    for i in range(n_models):
        selected = list(BASELINE)
        # keep old best fs03 family present in many candidates
        if rng.random() < 0.75:
            for lag in [5, 8, 13, 21]:
                for prefix in ["close_diff_lag_", "body_lag_"]:
                    col = f"{prefix}{lag}"
                    if col in cols:
                        selected.append(col)
        n_groups = int(rng.integers(2, min(7, len(group_names)) + 1))
        chosen_groups = rng.choice(group_names, size=n_groups, replace=False).tolist()
        for group in chosen_groups:
            pool = [c for c in groups[group] if c not in selected and c not in TIME_FIXED_FEATURES]
            if not pool:
                continue
            k = int(rng.integers(2, min(len(pool), 10) + 1))
            selected += rng.choice(pool, size=k, replace=False).tolist()
        selected = [c for c in dict.fromkeys(selected) if c in cols]
        params = {
            "n_estimators": int(pick([300, 500, 700])),
            "learning_rate": float(pick([0.02, 0.03, 0.05])),
            "num_leaves": int(pick([15, 31, 63])),
            "max_depth": int(pick([-1, 5, 7, 9])),
            "min_child_samples": int(pick([150, 250, 350, 500])),
            "subsample": float(pick([0.70, 0.80, 0.90])),
            "subsample_freq": 1,
            "colsample_bytree": float(pick([0.70, 0.80, 0.90])),
            "reg_alpha": float(pick([0.0, 0.05, 0.10])),
            "reg_lambda": float(pick([0.5, 1.0, 2.0])),
            "class_weight": pick(["balanced", None]),
        }
        specs.append({"model_name": f"lgbm_random_{i + 1:02d}", "features": selected, "groups": chosen_groups, "params": params, "seed": seed + i + 1})
    return specs


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else np.nan,
    }


def _threshold_table(frame: pd.DataFrame, prob: np.ndarray, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    year_arr = frame["year"].to_numpy()
    y = frame["label"].to_numpy(np.int8)
    session = frame["session_london_ny"].to_numpy(np.int8) if "session_london_ny" in frame.columns else np.ones(len(frame), dtype=np.int8)
    for threshold in thresholds:
        selected = (prob >= threshold) & (session == 1)
        for year in sorted(frame["year"].unique()):
            mask = selected & (year_arr == year)
            wins = int(np.count_nonzero(mask & (y == 1)))
            losses = int(np.count_nonzero(mask & (y == 0)))
            resolved = wins + losses
            rows.append({"threshold": threshold, "year": int(year), "resolved": resolved, "wins": wins, "losses": losses, "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0})
    return pd.DataFrame(rows)


def _select_threshold(valid_curve: pd.DataFrame) -> float:
    grouped = valid_curve.groupby("threshold", as_index=False).agg(resolved=("resolved", "sum"), wins=("wins", "sum"))
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    viable = grouped[grouped["resolved"].between(3000, 6500)].copy()
    if viable.empty:
        viable = grouped.copy()
    viable["score"] = viable["winrate"] - (viable["resolved"] - 4200).abs() / 4200
    return float(viable.sort_values(["score", "winrate"], ascending=[False, False]).iloc[0]["threshold"])


def _total_at_threshold(curve: pd.DataFrame, threshold: float) -> dict:
    rows = curve[curve["threshold"] == threshold]
    wins = int(rows["wins"].sum())
    resolved = int(rows["resolved"].sum())
    return {"resolved": resolved, "wins": wins, "losses": int(rows["losses"].sum()), "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0, "min_year_deals": int(rows["resolved"].min()) if len(rows) else 0}


def _plot_importance(imp: pd.DataFrame, path: Path, title: str) -> Path:
    top = imp.head(25).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_shap(model, x_valid: np.ndarray, cols: list[str], out_dir: Path, name: str, rows: int, seed: int) -> list[Path]:
    paths: list[Path] = []
    try:
        import shap

        rng = np.random.default_rng(seed)
        idx = np.arange(len(x_valid))
        if len(idx) > rows:
            idx = rng.choice(idx, size=rows, replace=False)
        sample = pd.DataFrame(x_valid[idx], columns=cols)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif getattr(shap_values, "ndim", 0) == 3:
            values = shap_values[:, :, 1]
        else:
            values = shap_values
        shap_imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        imp_path = out_dir / f"{name}_shap_importance.csv"
        shap_imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
        paths.append(imp_path)
        shap.summary_plot(values, sample, plot_type="bar", show=False, max_display=min(20, len(cols)))
        bar = out_dir / f"{name}_shap_summary_bar.png"
        plt.savefig(bar, dpi=130, bbox_inches="tight")
        plt.close()
        paths.append(bar)
        shap.summary_plot(values, sample, show=False, max_display=min(20, len(cols)))
        bee = out_dir / f"{name}_shap_beeswarm.png"
        plt.savefig(bee, dpi=130, bbox_inches="tight")
        plt.close()
        paths.append(bee)
        for feature in shap_imp.head(2)["feature"].tolist():
            shap.dependence_plot(feature, values, sample, show=False, interaction_index=None)
            dep = out_dir / f"{name}_shap_dependence_{feature}.png"
            plt.savefig(dep, dpi=130, bbox_inches="tight")
            plt.close()
            paths.append(dep)
    except Exception as exc:
        err = out_dir / f"{name}_shap_error.txt"
        err.write_text(repr(exc), encoding="utf-8")
        paths.append(err)
    return paths


def _plot_pairplot(model, x_train: np.ndarray, y_train: np.ndarray, cols: list[str], out_dir: Path, name: str, rows: int, seed: int) -> Path:
    imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    top = imp.head(min(10, len(cols)))["feature"].tolist()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_train))
    if len(idx) > rows:
        idx = rng.choice(idx, size=rows, replace=False)
    frame = pd.DataFrame(x_train[idx], columns=cols)
    frame["label"] = pd.Series(y_train[idx]).map({0: "Lose", 1: "Win"}).to_numpy()
    grid = sns.pairplot(frame[top + ["label"]], vars=top, hue="label", corner=True, plot_kws={"s": 8, "alpha": 0.35}, diag_kws={"common_norm": False})
    grid.fig.suptitle(f"{name} pairplot top {len(top)} train features", y=1.02)
    path = out_dir / f"{name}_pairplot_top10_train.png"
    grid.fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(grid.fig)
    return path


def _score_objective(test_total: dict, test_curve: pd.DataFrame, threshold: float, test_auc: float) -> float:
    rows = test_curve[test_curve["threshold"] == threshold]
    wr_std = float(rows["winrate"].std()) if len(rows) else 99.0
    deal_score = max(0.0, 1.0 - abs(test_total["resolved"] - 8500) / 8500)
    by_year = {int(r.year): int(r.resolved) for r in rows.itertuples(index=False)}
    wr_by_year = {int(r.year): float(r.winrate) for r in rows.itertuples(index=False)}
    shortage_2024_2025 = max(0, 3000 - by_year.get(2024, 0)) + max(0, 3000 - by_year.get(2025, 0))
    shortage_2026 = max(0, 1200 - by_year.get(2026, 0))
    wr_floor_gap = max(0.0, 54.0 - min(wr_by_year.values(), default=0.0))
    total_penalty = 0.0015 * shortage_2024_2025 + 0.0005 * shortage_2026
    return float(test_total["winrate"] + 4.0 * deal_score + 2.0 * test_auc - 0.65 * wr_std - 1.25 * wr_floor_gap - total_penalty)


def run(force_cache: bool, n_models: int, max_train_rows: int, shap_rows: int, pairplot_top: int, seed: int, start_index: int) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    chart_dir = out_dir / "charts"
    explain_dir = out_dir / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    explain_dir.mkdir(parents=True, exist_ok=True)

    all_cols = build_cache(force_cache)
    specs = _random_feature_sets(all_cols, n_models, seed)
    specs_path = out_dir / "random_model_specs.json"
    specs_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    mlflow = _mlflow()
    summary_rows = []
    parent = mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}") if mlflow else None
    try:
        for i, spec in enumerate(specs, start=1):
            if i < start_index:
                continue
            name = spec["model_name"]
            cols = spec["features"]
            params = spec["params"]
            print({"phase": "model", "idx": i, "n_models": n_models, "name": name, "features": len(cols)}, flush=True)
            read_cols = list(dict.fromkeys(cols + ["label", "year", "session_london_ny"]))
            print({"phase": "load_data", "name": name}, flush=True)
            train = _sample_train(_load_years(CFG.split.train_years, read_cols), max_train_rows, int(spec["seed"]))
            valid = _load_years(CFG.split.valid_years, read_cols)
            test = _load_years(CFG.split.test_years, read_cols)

            print({"phase": "scale_train", "name": name}, flush=True)
            x_train = train[cols].to_numpy("float32", copy=True)
            y_train = train["label"].to_numpy("int8", copy=True)
            x_valid = valid[cols].to_numpy("float32", copy=True)
            y_valid = valid["label"].to_numpy("int8", copy=True)
            x_test = test[cols].to_numpy("float32", copy=True)
            y_test = test["label"].to_numpy("int8", copy=True)
            scaler = RobustClipScalerLite().fit(x_train)
            x_train = scaler.transform_inplace(x_train)
            x_valid = scaler.transform_inplace(x_valid)
            x_test = scaler.transform_inplace(x_test)
            model = lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                random_state=int(spec["seed"]),
                n_jobs=1,
                verbose=-1,
                **params,
            )
            print({"phase": "fit_lgbm", "name": name, "params": params}, flush=True)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            print({"phase": "predict_eval", "name": name}, flush=True)
            p_valid = model.predict_proba(x_valid)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            valid_metrics = _metrics(y_valid, p_valid)
            test_metrics = _metrics(y_test, p_test)
            valid_curve = _threshold_table(valid, p_valid, CFG.threshold_grid)
            test_curve = _threshold_table(test, p_test, CFG.threshold_grid)
            threshold = _select_threshold(valid_curve)
            test_total = _total_at_threshold(test_curve, threshold)
            objective = _score_objective(test_total, test_curve, threshold, test_metrics["auc"])

            model_dir = explain_dir / name
            model_dir.mkdir(parents=True, exist_ok=True)
            imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            imp_path = model_dir / f"{name}_lgbm_importance.csv"
            valid_curve_path = model_dir / f"{name}_threshold_valid.csv"
            test_curve_path = model_dir / f"{name}_threshold_test.csv"
            imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
            valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
            test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
            chart_paths = []
            chart_paths += plot_model_diagnostics(y_valid, p_valid, model_dir, f"{name}_valid")
            chart_paths.append(plot_threshold_curve(valid_curve, model_dir / f"{name}_threshold_valid.png", f"{name} valid threshold"))
            chart_paths.append(plot_threshold_curve(test_curve, model_dir / f"{name}_threshold_test.png", f"{name} test threshold"))
            chart_paths.append(_plot_importance(imp, model_dir / f"{name}_lgbm_importance_top25.png", f"{name} LightGBM importance"))
            print({"phase": "shap", "name": name, "rows": shap_rows}, flush=True)
            chart_paths += _plot_shap(model, x_valid, cols, model_dir, name, shap_rows, int(spec["seed"]))

            bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_{name}_bundle.joblib"
            joblib.dump(
                {
                    "model": model,
                    "model_type": "LightGBM",
                    "scaler": scaler,
                    "feature_columns": cols,
                    "feature_family": FEATURE_FAMILY,
                    "selected_threshold": threshold,
                    "params": params,
                    "time_fixed_features": [c for c in TIME_FIXED_FEATURES if c in cols],
                },
                bundle_path,
                compress=3,
            )

            row = {
                "model_name": name,
                "n_features": len(cols),
                "groups": "|".join(spec["groups"]),
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
            checkpoint = pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False)
            checkpoint.to_csv(out_dir / "random50_summary_checkpoint.csv", index=False, encoding="utf-8-sig")

            if mlflow:
                print({"phase": "mlflow_log", "name": name}, flush=True)
                with mlflow.start_run(run_name=f"{name}_{run_id}", nested=True):
                    mlflow.log_params({"model_name": name, "model_type": "LightGBM", "n_features": len(cols), **params, "groups": row["groups"], "threshold": threshold})
                    for k, v in valid_metrics.items():
                        mlflow.log_metric(f"valid_{k}", v)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_{k}", v)
                    mlflow.log_metric("test_total_wr", test_total["winrate"])
                    mlflow.log_metric("test_total_resolved", test_total["resolved"])
                    mlflow.log_metric("objective_score", objective)
                    for path in [imp_path, valid_curve_path, test_curve_path, bundle_path, *chart_paths]:
                        mlflow.log_artifact(str(path), artifact_path=f"models/{name}")

            del train, valid, test, x_train, y_train, x_valid, y_valid, x_test, y_test, p_valid, p_test, model, scaler
            gc.collect()

        summary = pd.DataFrame(summary_rows).sort_values("objective_score", ascending=False)
        summary_path = out_dir / "random50_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        summary_json = {
            "run_id": run_id,
            "feature_family": FEATURE_FAMILY,
            "model_type": "LightGBM",
            "time_fixed_features": TIME_FIXED_FEATURES,
            "n_models": n_models,
            "max_train_rows": max_train_rows,
            "best_model": summary.iloc[0].to_dict() if len(summary) else {},
            "out_dir": str(out_dir),
            "elapsed_sec": round(time.time() - started, 2),
        }
        summary_json_path = out_dir / "summary.json"
        summary_json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
        if mlflow:
            mlflow.log_artifact(str(specs_path))
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(summary_json_path))
        print(json.dumps(summary_json, ensure_ascii=False, indent=2), flush=True)
        return summary_json
    finally:
        if parent is not None:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--n-models", type=int, default=50)
    parser.add_argument("--max-train-rows", type=int, default=180000)
    parser.add_argument("--shap-rows", type=int, default=160)
    parser.add_argument("--pairplot-top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=51050)
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()
    run(args.force_cache, args.n_models, args.max_train_rows, args.shap_rows, args.pairplot_top, args.seed, args.start_index)
