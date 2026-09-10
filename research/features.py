from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import CFG
from data_io import CLOSE, DATE, HIGH, LOW, OPEN, VOLUME
from method.OverlayIndicatorsNormalization import (
    OverlayNormalizationConfig,
    normalize_overlay_indicator,
    rolling_rank_scale,
)


class RobustClipScaler(BaseEstimator, TransformerMixin):
    """Clip train-fitted tails, then median/IQR-scale.

    This is intentionally stable for live data_handler: future outliers are clipped to
    train quantile boundaries before scaling, reducing distribution shocks.
    """

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper_q, axis=0)
        clipped = np.clip(arr, self.lower_, self.upper_)
        self.median_ = np.nanmedian(clipped, axis=0)
        q25 = np.nanquantile(clipped, 0.25, axis=0)
        q75 = np.nanquantile(clipped, 0.75, axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, self.lower_, self.upper_)
        arr = (arr - self.median_) / self.scale_
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    point = 100.0
    out["ret_1"] = (out[CLOSE] - out[OPEN]) * point
    out["hl_range"] = (out[HIGH] - out[LOW]) * point
    out["upper_wick"] = (out[HIGH] - out[[OPEN, CLOSE]].max(axis=1)) * point
    out["lower_wick"] = (out[[OPEN, CLOSE]].min(axis=1) - out[LOW]) * point
    out["body_abs"] = out["ret_1"].abs()
    out["close_pos_range"] = (out[CLOSE] - out[LOW]) / (out[HIGH] - out[LOW]).replace(0, np.nan)
    out["atr14"] = _atr(out[HIGH], out[LOW], out[CLOSE], period=14)

    returns = out[CLOSE].diff() * point
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        out[f"close_diff_lag_{lag}"] = returns.shift(lag)
        out[f"body_lag_{lag}"] = out["ret_1"].shift(lag)

    for win in [5, 10, 20, 50, 100, 200]:
        roll_close = out[CLOSE].rolling(win, min_periods=win)
        ma = roll_close.mean()
        std = roll_close.std().replace(0, np.nan)
        out[f"z_close_{win}"] = (out[CLOSE] - ma) / std
        out[f"range_mean_{win}"] = out["hl_range"].rolling(win, min_periods=win).mean()
        out[f"body_mean_{win}"] = out["body_abs"].rolling(win, min_periods=win).mean()
        out[f"ret_sum_{win}"] = returns.rolling(win, min_periods=win).sum()
        out[f"volatility_{win}"] = returns.rolling(win, min_periods=win).std()

    for span in [8, 21, 55, 144]:
        ema = out[CLOSE].ewm(span=span, min_periods=span, adjust=False).mean()
        out[f"ema_dist_{span}"] = (out[CLOSE] - ema) * point
        out[f"ema_slope_{span}"] = ema.diff() * point

    mid = out[CLOSE].rolling(20, min_periods=20).mean()
    std = out[CLOSE].rolling(20, min_periods=20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    out["bb20_pos"] = (out[CLOSE] - lower) / (upper - lower).replace(0, np.nan)
    out["bb20_width"] = (upper - lower) * point
    out["bb20_lower_dist"] = (out[CLOSE] - lower) * point
    out["rsi_14"] = _rsi(out[CLOSE], 14)
    out["rsi_50"] = _rsi(out[CLOSE], 50)

    out = _add_bb_family_features(out, point=point)

    dt = pd.to_datetime(out[DATE])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 5)
    out["session_london_ny"] = ((dt.dt.hour >= 7) & (dt.dt.hour <= 21)).astype(int)
    out["session_ny"] = ((dt.dt.hour >= 12) & (dt.dt.hour <= 21)).astype(int)

    if VOLUME in out.columns:
        out["volume_z_100"] = (out[VOLUME] - out[VOLUME].rolling(100, min_periods=100).mean()) / out[VOLUME].rolling(100, min_periods=100).std().replace(0, np.nan)

    return out


def _add_bb_family_features(out: pd.DataFrame, point: float = 100.0) -> pd.DataFrame:
    """Add BB20/BB100/BB200 features known at current candle close.

    No future data_handler is used. Width ranks compare current width to previous rows
    only via shifted rolling reference.
    """

    close = out[CLOSE]
    atr14 = out["atr14"]
    norm_cfg = OverlayNormalizationConfig(window=500, min_periods=100, use_past_only=True)
    extra: dict[str, pd.Series] = {}
    for period in [20, 100, 200]:
        p = f"bb{period}"
        mid = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        width = upper - lower
        denom = width.replace(0, np.nan)
        pos = (close - lower) / denom
        z = (close - mid) / std.replace(0, np.nan)
        width_points = width * point
        mid_slope_5 = (mid - mid.shift(5)) * point
        extra[f"{p}_mid"] = mid
        extra[f"{p}_upper"] = upper
        extra[f"{p}_lower"] = lower
        extra[f"{p}_pos"] = pos
        extra[f"{p}_z"] = z
        extra[f"{p}_width"] = width_points
        extra[f"{p}_mid_dist"] = (close - mid) * point
        extra[f"{p}_upper_dist"] = (upper - close) * point
        extra[f"{p}_lower_dist"] = (close - lower) * point
        extra[f"{p}_mid_slope_1"] = mid.diff() * point
        extra[f"{p}_mid_slope_5"] = mid_slope_5
        extra[f"{p}_width_slope_1"] = width.diff() * point
        width_slope_5 = (width - width.shift(5)) * point
        extra[f"{p}_width_slope_5"] = width_slope_5
        extra[f"{p}_squeeze"] = (width_slope_5 < 0).astype(int)
        extra[f"{p}_expansion"] = (width_slope_5 > 0).astype(int)
        extra[f"{p}_lower_reversion_raw"] = (pos <= 0.20).astype(int)
        extra[f"{p}_outer_low"] = (pos < 0.0).astype(int)

        for line_name, indicator in [("mid", mid), ("upper", upper), ("lower", lower)]:
            norm = normalize_overlay_indicator(
                indicator=indicator,
                close=close,
                atr=atr14,
                name=f"{p}_{line_name}_ov",
                config=norm_cfg,
                value_method="atr_gap",
                value_lag_periods=(1, 3, 5, 8, 13),
                distance_lag_periods=(1, 3, 5, 8, 13),
                raw_lag_periods=(),
                velocity_periods=(1, 3, 5, 8),
                velocity_lag_periods=(1, 3),
                acceleration_periods=(1, 3, 5),
                std_window=min(period, 50),
                std_mode="relative_distance_rolling_rank",
            )
            for col in norm.columns:
                extra[col] = norm[col].set_axis(out.index)

        pos_clip = pos.clip(-0.5, 1.5)
        extra[f"{p}_pos_cycle_sin"] = np.sin(2 * np.pi * pos_clip)
        extra[f"{p}_pos_cycle_cos"] = np.cos(2 * np.pi * pos_clip)
        extra[f"{p}_pos_rank_500"] = rolling_rank_scale(pos, window=500, min_periods=100, use_past_only=True).set_axis(out.index)
        extra[f"{p}_z_rank_500"] = rolling_rank_scale(z, window=500, min_periods=100, use_past_only=True).set_axis(out.index)
        width_atr = width_points / (atr14 * point).replace(0, np.nan)
        extra[f"{p}_width_atr"] = width_atr
        extra[f"{p}_width_atr_rank_500"] = rolling_rank_scale(width_atr, window=500, min_periods=100, use_past_only=True).set_axis(out.index)
        mid_gap_atr = (close - mid) / atr14.replace(0, np.nan)
        lower_gap_atr = (close - lower) / atr14.replace(0, np.nan)
        for win in [20, 100, 200]:
            extra[f"{p}_mid_gap_atr_rollmean_{win}"] = mid_gap_atr.rolling(win, min_periods=win).mean()
            extra[f"{p}_mid_gap_atr_rollstd_{win}"] = mid_gap_atr.rolling(win, min_periods=win).std()
            extra[f"{p}_lower_gap_atr_rollmean_{win}"] = lower_gap_atr.rolling(win, min_periods=win).mean()
            extra[f"{p}_lower_gap_atr_rollstd_{win}"] = lower_gap_atr.rolling(win, min_periods=win).std()

        width_ref = width_points.shift(1)
        for rank_win in [500, 2000]:
            roll_min = width_ref.rolling(rank_win, min_periods=min(100, rank_win)).min()
            roll_max = width_ref.rolling(rank_win, min_periods=min(100, rank_win)).max()
            extra[f"{p}_width_minmax_{rank_win}"] = (width_points - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    extra["bb20_bb100_pos_diff"] = extra["bb20_pos"] - extra["bb100_pos"]
    extra["bb20_bb200_pos_diff"] = extra["bb20_pos"] - extra["bb200_pos"]
    extra["bb100_bb200_pos_diff"] = extra["bb100_pos"] - extra["bb200_pos"]
    extra["bb20_width_over_bb100"] = extra["bb20_width"] / extra["bb100_width"].replace(0, np.nan)
    extra["bb20_width_over_bb200"] = extra["bb20_width"] / extra["bb200_width"].replace(0, np.nan)
    extra["bb100_width_over_bb200"] = extra["bb100_width"] / extra["bb200_width"].replace(0, np.nan)
    extra["bb_lower_agree_20_100"] = ((extra["bb20_pos"] <= 0.20) & (extra["bb100_pos"] <= 0.35)).astype(int)
    extra["bb_lower_agree_20_200"] = ((extra["bb20_pos"] <= 0.20) & (extra["bb200_pos"] <= 0.40)).astype(int)
    extra["bb_lower_agree_all"] = (
        (extra["bb20_pos"] <= 0.20) & (extra["bb100_pos"] <= 0.35) & (extra["bb200_pos"] <= 0.40)
    ).astype(int)
    extra["bb_trend_context_buy"] = ((extra["bb100_mid_slope_5"] > 0) | (extra["bb200_mid_slope_5"] > 0)).astype(int)

    duplicated = [col for col in extra if col in out.columns]
    if duplicated:
        out = out.drop(columns=duplicated)
    extra_df = pd.DataFrame(extra, index=out.index)
    float_cols = extra_df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        extra_df[float_cols] = extra_df[float_cols].astype("float32")
    return pd.concat([out, extra_df], axis=1)


def _completed_htf_features(df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    base = df[[DATE, OPEN, HIGH, LOW, CLOSE]].copy().set_index(DATE)
    ohlc = base.resample(rule, label="right", closed="right").agg(
        {OPEN: "first", HIGH: "max", LOW: "min", CLOSE: "last"}
    ).dropna().reset_index()
    point = 100.0
    ohlc[f"{prefix}_ret"] = (ohlc[CLOSE] - ohlc[OPEN]) * point
    ohlc[f"{prefix}_range"] = (ohlc[HIGH] - ohlc[LOW]) * point
    for span in [8, 21, 55]:
        ema = ohlc[CLOSE].ewm(span=span, min_periods=span, adjust=False).mean()
        ohlc[f"{prefix}_ema_dist_{span}"] = (ohlc[CLOSE] - ema) * point
        ohlc[f"{prefix}_ema_slope_{span}"] = ema.diff() * point
    ohlc[f"{prefix}_rsi_14"] = _rsi(ohlc[CLOSE], 14)
    keep = [DATE] + [col for col in ohlc.columns if col.startswith(prefix)]
    return pd.merge_asof(df[[DATE]].sort_values(DATE), ohlc[keep].sort_values(DATE), on=DATE, direction="backward")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    out = _add_base_features(data)
    for rule, prefix in [("5min", "m5"), ("15min", "m15"), ("1h", "h1"), ("4h", "h4")]:
        htf = _completed_htf_features(out, rule, prefix)
        for col in htf.columns:
            if col != DATE:
                out[col] = htf[col].to_numpy()

    # Candidate Buy signals known at close of current M1 candle. Entry is next open.
    out["sig_bb_reversion_buy"] = ((out["bb20_pos"] < 0.18) & (out["rsi_14"] < 45)).astype(int)
    out["sig_pullback_trend_buy"] = ((out["ema_dist_55"] > 0) & (out["ema_dist_8"] < 0) & (out["ret_1"] > 0)).astype(int)
    out["sig_momentum_buy"] = ((out["ret_sum_10"] > 0) & (out["z_close_20"] > -0.5) & (out["z_close_20"] < 1.5)).astype(int)
    out["buy_signal"] = (
        (out["sig_bb_reversion_buy"] == 1)
        | (out["sig_pullback_trend_buy"] == 1)
        | ((out["sig_momentum_buy"] == 1) & (out["session_london_ny"] == 1))
    ).astype(int)

    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, "label", "label_res", "entry_index", "entry_price", "year"}
    return [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]


def named_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    cols = feature_columns(frame)

    def pick(*tokens: str) -> list[str]:
        return [col for col in cols if any(token in col for token in tokens)]

    def pick_exact(*names: str) -> list[str]:
        wanted = set(names)
        return [col for col in cols if col in wanted]

    bb_family_relative = []
    for period in [20, 100, 200]:
        p = f"bb{period}"
        bb_family_relative.extend(
            [
                f"{p}_pos",
                f"{p}_z",
                f"{p}_width",
                f"{p}_mid_dist",
                f"{p}_upper_dist",
                f"{p}_lower_dist",
                f"{p}_mid_slope_1",
                f"{p}_mid_slope_5",
                f"{p}_width_slope_1",
                f"{p}_width_slope_5",
                f"{p}_squeeze",
                f"{p}_expansion",
                f"{p}_lower_reversion_raw",
                f"{p}_outer_low",
                f"{p}_width_minmax_500",
                f"{p}_width_minmax_2000",
            ]
        )
    bb_cross_relative = [
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
    ]

    sets = {
        "fs01_price_action": pick("ret_1", "hl_range", "wick", "body_abs", "close_pos_range"),
        "fs02_lags_short": pick("lag_1", "lag_2", "lag_3", "lag_5", "ret_1", "hl_range"),
        "fs03_lags_cycle": pick("lag_5", "lag_8", "lag_13", "lag_21", "tod_", "dow_"),
        "fs04_rolling_short": pick("_5", "_10", "_20", "bb20", "rsi_14"),
        "fs05_rolling_long": pick("_50", "_100", "_200", "rsi_50", "volume_z"),
        "fs06_ema_trend": pick("ema_dist", "ema_slope"),
        "fs07_bb_rsi_reversion": pick("bb20", "rsi_", "lower_wick", "close_pos_range"),
        "fs08_time_session": pick("tod_", "dow_", "session_"),
        "fs09_mtf_fast": pick("m5_", "m15_", "ret_1", "bb20", "rsi_14"),
        "fs10_mtf_full_all": cols,
        "fs11_fs03_bb_core": pick("lag_5", "lag_8", "lag_13", "lag_21", "tod_", "dow_")
        + pick_exact(*(bb_family_relative + bb_cross_relative)),
        "fs12_bb_reversion_multiband": pick("tod_", "dow_", "bb20_pos", "bb100_pos", "bb200_pos", "bb20_z", "bb100_z", "bb200_z", "lower_reversion", "outer_low", "lower_agree", "trend_context"),
        "fs13_bb_width_regime": pick("tod_", "dow_", "width", "squeeze", "expansion", "pos_diff", "width_over", "lower_agree"),
        "fs14_bb_overlay_norm_behavior": pick("tod_", "dow_", "_ov_", "_cycle_", "_rank_500", "_width_atr", "_gap_atr_roll")
        + pick_exact(*(bb_cross_relative + ["atr14"])),
    }

    # Keep signal columns in every variant because the model predicts P(Win)
    # conditional on a candidate Buy event and these flags define the event family.
    signal_cols = [col for col in cols if col.startswith("sig_") or col == "buy_signal"]
    cleaned: dict[str, list[str]] = {}
    for name, selected in sets.items():
        merged = list(dict.fromkeys(selected + signal_cols))
        if not merged:
            raise ValueError(f"Empty feature set: {name}")
        cleaned[name] = merged
    return cleaned
