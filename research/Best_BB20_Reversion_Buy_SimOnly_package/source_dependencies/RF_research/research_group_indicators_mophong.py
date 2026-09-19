import argparse
import json
import logging
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


CURRENT_DIR = os.path.dirname(__file__)

from data import CLOSE, DATE, HIGH, LOW, MAU_NEN, MAX_VOL, OPEN, RAU_DUOI, RAU_TREN, THAN_NEN, VOLUME
from data.download_data_mt5 import get_data_from_csv
from method.MoPhongDeals import MoPhongDeals


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)

STRATEGY_NAME = "RF_Group_indicators_MoPhongDeals"
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
PREPARED_FILE = os.path.join(MODEL_DIR, "prepared_years.joblib")
MODEL_FILE = os.path.join(MODEL_DIR, "group_models.joblib")
SCORE_FILE = os.path.join(MODEL_DIR, "group_scores.joblib")
PROXY_FILE = os.path.join(CURRENT_DIR, "PROXY_RF_Group_indicators.csv")
GRID_FILE = os.path.join(CURRENT_DIR, "GRID_RF_Group_indicators_MoPhongDeals.csv")
SUMMARY_FILE = os.path.join(CURRENT_DIR, "SUMMARY_RF_Group_indicators_MoPhongDeals.csv")
BEST_FILE = os.path.join(CURRENT_DIR, "BEST_RF_Group_indicators_MoPhongDeals.json")

RR = 1
R1 = 6.0
MAX_FORWARD = 1440
MAX_RUNNING_DEALS = 10
EXPIRED = 5
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEARS = [2024, 2025, 2026]
ALL_YEARS = TRAIN_YEARS + TEST_YEARS


BASE_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "range_r",
    "body_r",
    "body_abs_r",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "wick_balance",
]

FEATURE_GROUPS = {
    "ma_family": BASE_FEATURES + [
        "ema5_dist_r",
        "ema9_dist_r",
        "ema20_dist_r",
        "ema21_dist_r",
        "ema50_dist_r",
        "ema100_dist_r",
        "ema200_dist_r",
        "ema5_slope_r",
        "ema9_slope_r",
        "ema20_slope_r",
        "ema21_slope_r",
        "ema50_slope_r",
        "ema100_slope_r",
        "ema200_slope_r",
        "ema_stack_bull",
        "ema_stack_bear",
    ],
    "bb_rsi_stoch": BASE_FEATURES + [
        "bb20_pos",
        "bb20_width_r",
        "bb20_z",
        "RSI14",
        "rsi_center",
        "rsi_slope3",
        "rsi_slope10",
        "STO14",
        "STO14_D3",
        "stoch_spread",
        "stoch_slope3",
    ],
    "adx_di_regime": BASE_FEATURES + [
        "ADX14",
        "adx_slope3",
        "adx_slope10",
        "PLUS_DI14",
        "MINUS_DI14",
        "di_spread",
        "di_abs",
        "atr14_r",
        "atr_rank",
    ],
    "psar_ma_cycle": BASE_FEATURES + [
        "psar_dist_r",
        "psar_side",
        "ema9_dist_r",
        "ema21_dist_r",
        "signal_dir",
        "cycle_age_clip",
        "cycle_phase",
        "phase_start",
        "phase_middle",
        "phase_end",
        "signal_spread_r",
        "signal_spread_slope_r",
        "bars_since_mid",
    ],
    "signal_cycle_only": [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "signal_dir",
        "cycle_age_clip",
        "cycle_phase",
        "phase_start",
        "phase_middle",
        "phase_end",
        "signal_spread_r",
        "signal_spread_slope_r",
        "bars_since_mid",
        "first_touch_ma20",
        "close_cross_ma20",
        "strong_body_confirm",
        "price_zone_ma20_r",
    ],
    "entry_timing_zone": BASE_FEATURES + [
        "signal_dir",
        "cycle_phase",
        "bars_since_mid",
        "first_touch_ma20",
        "close_cross_ma20",
        "strong_body_confirm",
        "price_zone_ma20_r",
        "bb20_pos",
        "RSI14",
        "ema20_dist_r",
        "psar_side",
    ],
    "ma5_ma20_cycle_order": BASE_FEATURES + [
        "ma5_ma20_signal_dir",
        "ma5_ma20_cycle_bar",
        "ma5_ma20_cycle_bar_clip",
        "ma5_ma20_cycle_bar_log",
        "ma5_ma20_cycle_start",
        "ma5_ma20_cycle_age_0_3",
        "ma5_ma20_cycle_age_4_10",
        "ma5_ma20_cycle_age_11_30",
        "ma5_ma20_cycle_age_31_plus",
        "ma5_ma20_spread_r",
        "ma5_ma20_spread_slope_r",
        "ma5_ma20_price_extension_r",
        "ma5_ma20_first_touch_ma20",
        "ma5_ma20_touch_ma20_count_clip",
        "strong_body_confirm",
        "price_zone_ma20_r",
    ],
    "momentum_volatility": BASE_FEATURES + [
        "ret1_r",
        "ret3_r",
        "ret5_r",
        "ret10_r",
        "ret15_r",
        "ret30_r",
        "ret60_r",
        "ret120_r",
        "vol30",
        "vol60",
        "vol120",
        "range_rank",
        "atr_rank",
        "ADX14",
        "di_spread",
        "signal_dir",
        "cycle_phase",
    ],
}


def _maybe_njit(fn):
    return njit(fn) if njit is not None else fn


@_maybe_njit
def _tp_sl_next_open(open_, high, low, close, r1, max_fwd):
    n = len(open_)
    buy = np.zeros(n, np.int8)
    sell = np.zeros(n, np.int8)
    for i in range(250, n - 2):
        entry = open_[i + 1]
        buy_tp = entry + r1
        buy_sl = entry - r1
        sell_tp = entry - r1
        sell_sl = entry + r1
        end = i + 1 + max_fwd
        if end > n:
            end = n
        for j in range(i + 1, end):
            buy_hit_tp = high[j] >= buy_tp
            buy_hit_sl = low[j] <= buy_sl
            sell_hit_tp = low[j] <= sell_tp
            sell_hit_sl = high[j] >= sell_sl
            if buy_hit_tp and buy_hit_sl:
                if close[j] >= open_[j]:
                    buy[i] = 1
                    sell[i] = -1
                else:
                    buy[i] = -1
                    sell[i] = 1
                break
            if buy_hit_tp:
                buy[i] = 1
                sell[i] = -1
                break
            if buy_hit_sl:
                buy[i] = -1
                sell[i] = 1
                break
            if sell_hit_tp:
                sell[i] = 1
                buy[i] = -1
                break
            if sell_hit_sl:
                sell[i] = -1
                buy[i] = 1
                break
    return buy, sell


def _load_year_data(year):
    path = rf"D:\SCJ999\data\1M\XAUUSDm_{year}.csv"
    if not os.path.exists(path):
        raise RuntimeError(f"Missing history file: {path}")
    data = get_data_from_csv(path)
    if year == 2026:
        data = data[data[DATE] < "2026-06-06"].copy()
    if VOLUME not in data.columns:
        data[VOLUME] = 1
    return data[[DATE, OPEN, HIGH, LOW, CLOSE, VOLUME]].dropna().reset_index(drop=True)


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def _calc_psar(high, low, start_af=0.02, step_af=0.02, max_af=0.2):
    high_np = high.to_numpy(float)
    low_np = low.to_numpy(float)
    sar = np.zeros(len(high_np), dtype=float)
    if len(sar) == 0:
        return pd.Series(sar)
    sar[0] = low_np[0]
    af = start_af
    ep = high_np[0]
    uptrend = True
    for i in range(1, len(sar)):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if uptrend:
            sar[i] = min(sar[i], low_np[i - 1], low_np[i - 2] if i > 1 else low_np[i - 1])
            if low_np[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low_np[i]
                af = start_af
            elif high_np[i] > ep:
                ep = high_np[i]
                af = min(af + step_af, max_af)
        else:
            sar[i] = max(sar[i], high_np[i - 1], high_np[i - 2] if i > 1 else high_np[i - 1])
            if high_np[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high_np[i]
                af = start_af
            elif low_np[i] < ep:
                ep = low_np[i]
                af = min(af + step_af, max_af)
    return pd.Series(sar)


def _add_basic_candle_columns(data):
    data[MAU_NEN] = (data[CLOSE] - data[OPEN]) * 100
    data[THAN_NEN] = (data[CLOSE] - data[OPEN]).abs() * 100
    data[RAU_TREN] = (data[HIGH] - np.maximum(data[OPEN], data[CLOSE])).abs() * 100
    data[RAU_DUOI] = (np.minimum(data[OPEN], data[CLOSE]) - data[LOW]).abs() * 100
    data[MAX_VOL] = (data[HIGH] - data[LOW]) * 100
    return data


def _add_signal_cycle_features(data):
    signal_dir = np.sign((data["EMA9"] - data["EMA21"]).fillna(0).to_numpy(float)).astype(np.int8)
    signal_dir[signal_dir == 0] = 1
    changed = np.r_[True, signal_dir[1:] != signal_dir[:-1]]
    cycle_id = np.cumsum(changed)
    age = pd.Series(np.arange(len(data))).groupby(cycle_id).cumcount().to_numpy()

    spread = (data["EMA9"] - data["EMA21"]).abs()
    spread_slope = spread - spread.shift(3)
    mid_condition = (age > 3) & (spread_slope.to_numpy(float) >= 0)
    end_condition = (age > 8) & (spread_slope.rolling(3).sum().to_numpy(float) < 0)
    phase = np.zeros(len(data), dtype=np.int8)
    phase[mid_condition] = 1
    phase[end_condition] = 2

    bars_since_mid = np.zeros(len(data), dtype=np.int16)
    last_mid = -1
    for i in range(len(data)):
        if phase[i] == 1:
            last_mid = i
        bars_since_mid[i] = 0 if last_mid < 0 else i - last_mid

    prev_close_side = np.sign((data[CLOSE].shift(1) - data["EMA20"].shift(1)).fillna(0))
    close_side = np.sign((data[CLOSE] - data["EMA20"]).fillna(0))
    close_cross = ((prev_close_side != 0) & (close_side != 0) & (prev_close_side != close_side)).astype(float)
    touch_ma20 = ((data[LOW] <= data["EMA20"]) & (data[HIGH] >= data["EMA20"])).astype(float)
    first_touch_ma20 = np.zeros(len(data), dtype=float)
    seen_touch = set()
    for i, cid in enumerate(cycle_id):
        if touch_ma20.iat[i] and cid not in seen_touch:
            first_touch_ma20[i] = 1.0
            seen_touch.add(cid)

    body_abs = (data[CLOSE] - data[OPEN]).abs()
    strong_body = (body_abs > body_abs.rolling(200, min_periods=50).quantile(0.70)).astype(float)
    candle_dir = np.sign(data[CLOSE] - data[OPEN])
    strong_confirm = ((strong_body == 1) & (candle_dir == signal_dir)).astype(float)

    ma5_ma20_dir = np.sign((data["EMA5"] - data["EMA20"]).fillna(0).to_numpy(float)).astype(np.int8)
    ma5_ma20_dir[ma5_ma20_dir == 0] = 1
    ma5_ma20_changed = np.r_[True, ma5_ma20_dir[1:] != ma5_ma20_dir[:-1]]
    ma5_ma20_cycle_id = np.cumsum(ma5_ma20_changed)
    ma5_ma20_bar = pd.Series(np.arange(len(data))).groupby(ma5_ma20_cycle_id).cumcount().to_numpy()
    ma5_ma20_spread = data["EMA5"] - data["EMA20"]
    ma5_ma20_spread_r = _safe_div(ma5_ma20_spread, data["norm_scale"])
    ma5_ma20_spread_slope_r = ma5_ma20_spread_r - ma5_ma20_spread_r.shift(3)
    ma5_ma20_price_extension_r = _safe_div(data[CLOSE] - data["EMA20"], data["norm_scale"]) * ma5_ma20_dir

    ma20_touch = ((data[LOW] <= data["EMA20"]) & (data[HIGH] >= data["EMA20"])).astype(int).to_numpy()
    ma5_ma20_first_touch = np.zeros(len(data), dtype=float)
    ma5_ma20_touch_count = np.zeros(len(data), dtype=np.int16)
    last_cycle = -1
    touches = 0
    for i, cid in enumerate(ma5_ma20_cycle_id):
        if cid != last_cycle:
            last_cycle = cid
            touches = 0
        if ma20_touch[i]:
            touches += 1
            if touches == 1:
                ma5_ma20_first_touch[i] = 1.0
        ma5_ma20_touch_count[i] = touches

    data["signal_dir"] = signal_dir
    data["cycle_age_clip"] = np.minimum(age, 60)
    data["cycle_phase"] = phase
    data["phase_start"] = (phase == 0).astype(float)
    data["phase_middle"] = (phase == 1).astype(float)
    data["phase_end"] = (phase == 2).astype(float)
    data["bars_since_mid"] = bars_since_mid
    data["first_touch_ma20"] = first_touch_ma20
    data["close_cross_ma20"] = close_cross
    data["strong_body_confirm"] = strong_confirm
    data["ma5_ma20_signal_dir"] = ma5_ma20_dir
    data["ma5_ma20_cycle_id"] = ma5_ma20_cycle_id
    data["ma5_ma20_cycle_bar"] = ma5_ma20_bar
    data["ma5_ma20_cycle_bar_clip"] = np.minimum(ma5_ma20_bar, 120)
    data["ma5_ma20_cycle_bar_log"] = np.log1p(ma5_ma20_bar)
    data["ma5_ma20_cycle_start"] = (ma5_ma20_bar == 0).astype(float)
    data["ma5_ma20_cycle_age_0_3"] = (ma5_ma20_bar <= 3).astype(float)
    data["ma5_ma20_cycle_age_4_10"] = ((ma5_ma20_bar >= 4) & (ma5_ma20_bar <= 10)).astype(float)
    data["ma5_ma20_cycle_age_11_30"] = ((ma5_ma20_bar >= 11) & (ma5_ma20_bar <= 30)).astype(float)
    data["ma5_ma20_cycle_age_31_plus"] = (ma5_ma20_bar >= 31).astype(float)
    data["ma5_ma20_spread_r"] = ma5_ma20_spread_r
    data["ma5_ma20_spread_slope_r"] = ma5_ma20_spread_slope_r
    data["ma5_ma20_price_extension_r"] = ma5_ma20_price_extension_r
    data["ma5_ma20_first_touch_ma20"] = ma5_ma20_first_touch
    data["ma5_ma20_touch_ma20_count_clip"] = np.minimum(ma5_ma20_touch_count, 10)
    return data


def _prepare_features(raw):
    data = _add_basic_candle_columns(raw.copy().reset_index(drop=True))
    dates = pd.to_datetime(data[DATE], errors="coerce")
    close = data[CLOSE].astype(float)
    open_ = data[OPEN].astype(float)
    high = data[HIGH].astype(float)
    low = data[LOW].astype(float)

    body_abs_price = (close - open_).abs()
    body_range = body_abs_price.rolling(200, min_periods=50).max() - body_abs_price.rolling(200, min_periods=50).min()
    fallback_scale = (high - low).rolling(200, min_periods=50).mean()
    data["norm_scale"] = body_range.replace(0, np.nan).fillna(fallback_scale).replace(0, np.nan)

    data["hour_sin"] = np.sin(2 * np.pi * dates.dt.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * dates.dt.hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    data["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)

    data["range_r"] = _safe_div(high - low, data["norm_scale"])
    data["body_r"] = _safe_div(close - open_, data["norm_scale"])
    data["body_abs_r"] = data["body_r"].abs()
    for n in [1, 3, 5, 10, 15, 30, 60, 120]:
        data[f"ret{n}_r"] = _safe_div(close - close.shift(n), data["norm_scale"])
    close_ret = close.pct_change()
    for n in [30, 60, 120]:
        data[f"vol{n}"] = close_ret.rolling(n).std() * 100
    candle_range = (high - low).replace(0, np.nan)
    data["upper_wick_ratio"] = (high - np.maximum(open_, close)) / candle_range
    data["lower_wick_ratio"] = (np.minimum(open_, close) - low) / candle_range
    data["wick_balance"] = data["lower_wick_ratio"] - data["upper_wick_ratio"]

    for ma in [5, 9, 20, 21, 50, 100, 200]:
        data[f"EMA{ma}"] = close.ewm(span=ma, min_periods=ma, adjust=False).mean()
        data[f"ema{ma}_dist_r"] = _safe_div(close - data[f"EMA{ma}"], data["norm_scale"])
        data[f"ema{ma}_slope_r"] = _safe_div(data[f"EMA{ma}"] - data[f"EMA{ma}"].shift(5), data["norm_scale"])
    data["ema_stack_bull"] = ((data["EMA5"] > data["EMA20"]) & (data["EMA20"] > data["EMA50"]) & (data["EMA50"] > data["EMA200"])).astype(float)
    data["ema_stack_bear"] = ((data["EMA5"] < data["EMA20"]) & (data["EMA20"] < data["EMA50"]) & (data["EMA50"] < data["EMA200"])).astype(float)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data["bb20_pos"] = _safe_div(close - bb_lower, bb_upper - bb_lower)
    data["bb20_width_r"] = _safe_div(bb_upper - bb_lower, data["norm_scale"])
    data["bb20_z"] = _safe_div(close - bb_mid, bb_std)

    data["RSI14"] = _calc_rsi(close)
    data["rsi_center"] = data["RSI14"] - 50
    data["rsi_slope3"] = data["RSI14"] - data["RSI14"].shift(3)
    data["rsi_slope10"] = data["RSI14"] - data["RSI14"].shift(10)
    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    data["STO14"] = 100 * _safe_div(close - lowest14, highest14 - lowest14)
    data["STO14_D3"] = data["STO14"].rolling(3).mean()
    data["stoch_spread"] = data["STO14"] - data["STO14_D3"]
    data["stoch_slope3"] = data["STO14"] - data["STO14"].shift(3)

    true_range = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    data["ATR14"] = true_range.rolling(14).mean()
    data["atr14_r"] = _safe_div(data["ATR14"], data["norm_scale"])
    data["atr_rank"] = data["ATR14"].rolling(2000, min_periods=200).rank(pct=True)
    data["range_rank"] = data["range_r"].rolling(2000, min_periods=200).rank(pct=True)
    up_move = high.diff()
    down_move = -low.diff()
    data["PLUS_DI14"] = 100 * up_move.where((up_move > down_move) & (up_move > 0), 0).rolling(14).mean() / data["ATR14"]
    data["MINUS_DI14"] = 100 * down_move.where((down_move > up_move) & (down_move > 0), 0).rolling(14).mean() / data["ATR14"]
    data["di_spread"] = data["PLUS_DI14"] - data["MINUS_DI14"]
    data["di_abs"] = data["di_spread"].abs()
    data["ADX14"] = (100 * data["di_abs"] / (data["PLUS_DI14"] + data["MINUS_DI14"])).rolling(14).mean()
    data["adx_slope3"] = data["ADX14"] - data["ADX14"].shift(3)
    data["adx_slope10"] = data["ADX14"] - data["ADX14"].shift(10)

    data["PSAR"] = _calc_psar(high, low)
    data["psar_dist_r"] = _safe_div(close - data["PSAR"], data["norm_scale"])
    data["psar_side"] = np.sign(close - data["PSAR"])

    data = _add_signal_cycle_features(data)
    data["signal_spread_r"] = _safe_div(data["EMA9"] - data["EMA21"], data["norm_scale"])
    data["signal_spread_slope_r"] = data["signal_spread_r"] - data["signal_spread_r"].shift(3)
    data["price_zone_ma20_r"] = _safe_div(close - data["EMA20"], data["norm_scale"])
    return data.replace([np.inf, -np.inf], np.nan)


def _load_prepared_years(force=False):
    if os.path.exists(PREPARED_FILE) and not force:
        print(f"Loaded prepared years: {PREPARED_FILE}")
        return joblib.load(PREPARED_FILE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    years = {}
    for year in ALL_YEARS:
        data = _prepare_features(_load_year_data(year))
        labels = _tp_sl_next_open(
            data[OPEN].to_numpy(np.float64),
            data[HIGH].to_numpy(np.float64),
            data[LOW].to_numpy(np.float64),
            data[CLOSE].to_numpy(np.float64),
            float(R1),
            MAX_FORWARD,
        )
        years[year] = {
            "data": data,
            "buy_label": labels[0],
            "sell_label": labels[1],
        }
        print(f"Prepared {year}: candles={len(data)}")
    joblib.dump(years, PREPARED_FILE)
    return years


def _rf_model(group_name, side):
    seed = abs(hash((group_name, side))) % 100000
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=700,
        min_samples_split=1400,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.75,
        n_jobs=-1,
        random_state=seed,
    )


def _build_dataset(years, group_name, side):
    label_key = "buy_label" if side == "buy" else "sell_label"
    direction_value = 1 if side == "buy" else -1
    feature_columns = FEATURE_GROUPS[group_name]
    xs = []
    ys = []
    for year in TRAIN_YEARS:
        data = years[year]["data"]
        labels = years[year][label_key]
        features = data[feature_columns]
        valid = (~features.isna().any(axis=1)).to_numpy()
        indexes = np.where((labels != 0) & valid & (data["signal_dir"].to_numpy(np.int8) == direction_value))[0]
        if len(indexes) > 80000:
            rng = np.random.default_rng(year + (17 if side == "buy" else 117))
            indexes = rng.choice(indexes, 80000, replace=False)
        x = features.fillna(0).to_numpy(float)[indexes]
        y = (labels[indexes] == 1).astype(np.int8)
        xs.append(x)
        ys.append(y)
        print(f"Dataset {group_name}/{side}/{year}: {len(y)} base_wr={round(y.mean() * 100, 2) if len(y) else 0}")
    return np.vstack(xs), np.concatenate(ys)


def _train_models(years, force=False):
    if os.path.exists(MODEL_FILE) and not force:
        print(f"Loaded models: {MODEL_FILE}")
        return joblib.load(MODEL_FILE)
    bundle = {}
    for group_name in FEATURE_GROUPS:
        bundle[group_name] = {}
        for side in ["buy", "sell"]:
            x, y = _build_dataset(years, group_name, side)
            model = _rf_model(group_name, side)
            model.fit(x, y)
            sample = min(len(y), 100000)
            auc = roc_auc_score(y[:sample], model.predict_proba(x[:sample])[:, 1]) if len(np.unique(y[:sample])) > 1 else 0.0
            bundle[group_name][side] = {
                "model": model,
                "features": FEATURE_GROUPS[group_name],
                "samples": int(len(y)),
                "base_winrate": round(float(y.mean() * 100), 2),
                "auc_sample": round(float(auc), 4),
            }
            print(f"Trained {group_name}/{side}: wr={bundle[group_name][side]['base_winrate']} auc={bundle[group_name][side]['auc_sample']}")
    joblib.dump(bundle, MODEL_FILE)
    return bundle


def _score_years(years, models, force=False):
    if os.path.exists(SCORE_FILE) and not force:
        print(f"Loaded scores: {SCORE_FILE}")
        return joblib.load(SCORE_FILE)
    scores = {}
    for group_name, group_models in models.items():
        scores[group_name] = {}
        feature_columns = group_models["buy"]["features"]
        for year in ALL_YEARS:
            data = years[year]["data"]
            frame = data[feature_columns]
            valid = (~frame.isna().any(axis=1)).to_numpy()
            x = frame.fillna(0).to_numpy(float)
            scores[group_name][year] = {
                "valid": valid,
                "buy": group_models["buy"]["model"].predict_proba(x)[:, 1],
                "sell": group_models["sell"]["model"].predict_proba(x)[:, 1],
            }
            print(f"Scored {group_name}/{year}")
    joblib.dump(scores, SCORE_FILE)
    return scores


def _session_mask(data, session_mode):
    if session_mode == "all":
        return np.ones(len(data), dtype=bool)
    if "_hour_cache" not in data.columns:
        parsed_dates = pd.to_datetime(data[DATE], errors="coerce")
        data["_hour_cache"] = parsed_dates.dt.hour
        data["_month_cache"] = parsed_dates.dt.month
    hours = data["_hour_cache"].to_numpy()
    months = data["_month_cache"].to_numpy()
    h1 = months <= 6
    if session_mode == "active":
        return (hours >= 7) & (hours <= 22)
    if session_mode == "london":
        return (hours >= 7) & (hours <= 16)
    if session_mode == "ny":
        return (hours >= 13) & (hours <= 22)
    if session_mode == "london_ny":
        return (hours >= 7) & (hours <= 22)
    if session_mode == "h1_all":
        return h1
    if session_mode == "h1_active":
        return h1 & (hours >= 7) & (hours <= 22)
    if session_mode == "h1_ny":
        return h1 & (hours >= 13) & (hours <= 22)
    raise ValueError(f"Unsupported session_mode: {session_mode}")


def _build_orders(data, score, buy_threshold, sell_threshold, entry_mode, min_phase, side_mode="both", session_mode="all"):
    signal = data["signal_dir"].to_numpy(np.int8)
    phase = data["cycle_phase"].to_numpy(np.int8)
    valid_phase = phase >= min_phase
    touch_or_cross = (
        (data["first_touch_ma20"].to_numpy(float) == 1)
        | (data["close_cross_ma20"].to_numpy(float) == 1)
        | (np.abs(data["price_zone_ma20_r"].to_numpy(float)) <= 0.35)
    )
    strong_confirm = data["strong_body_confirm"].to_numpy(float) == 1
    if entry_mode == "signal_any":
        entry_ok = valid_phase
    elif entry_mode == "ma20_zone":
        entry_ok = valid_phase & touch_or_cross
    elif entry_mode == "ma20_confirm":
        entry_ok = valid_phase & touch_or_cross & strong_confirm
    elif entry_mode == "bb_reversion":
        bb = data["bb20_pos"].to_numpy(float)
        rsi = data["RSI14"].to_numpy(float)
        entry_ok = valid_phase & (((signal == 1) & (bb < 0.25) & (rsi < 48)) | ((signal == -1) & (bb > 0.75) & (rsi > 52)))
    elif entry_mode == "trend_pullback":
        zone = np.abs(data["price_zone_ma20_r"].to_numpy(float)) <= 0.55
        entry_ok = valid_phase & zone & strong_confirm
    else:
        raise ValueError(f"Unsupported entry_mode: {entry_mode}")

    orders = np.zeros(len(data), dtype=np.int8)
    mask = score["valid"] & entry_ok
    mask = mask & _session_mask(data, session_mode)
    if side_mode in ["both", "buy_only"]:
        orders[mask & (signal == 1) & (score["buy"] >= buy_threshold)] = 1
    if side_mode in ["both", "sell_only"]:
        orders[mask & (signal == -1) & (score["sell"] >= sell_threshold)] = -1
    orders[:250] = 0
    orders[-2:] = 0
    return orders


def _simulate_mophong(data, orders):
    frame = data.copy().reset_index(drop=True)
    frame[DATE] = frame[DATE].astype(str)
    mophong = MoPhongDeals(frame, EXPIRED, tinh_tien=None)
    for candle in range(250, len(frame) - 2):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=MAX_RUNNING_DEALS)
        side = int(orders[candle])
        if side == 0:
            continue
        order = "buy" if side == 1 else "sell"
        entry = float(frame[OPEN][candle + 1])
        deal = {
            "start_index": candle + 1,
            "entry": round(entry, 5),
            "sl": round(entry - R1 if order == "buy" else entry + R1, 5),
            "tp": round(entry + R1 * RR if order == "buy" else entry - R1 * RR, 5),
            "type": "market",
            "order": order,
        }
        mophong.add_new_deal(deal=deal, min_deal_gap=0)
    for candle in range(max(250, len(frame) - 2), len(frame)):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=MAX_RUNNING_DEALS)
    results = mophong.get_results()
    wins = sum(1 for deal in results if deal["res"] == "Win")
    loses = sum(1 for deal in results if deal["res"] == "Lose")
    nores = sum(1 for deal in results if deal["res"] == "No res")
    resolved = wins + loses
    return {
        "raw_deals": int(np.count_nonzero(orders)),
        "results": len(results),
        "resolved": resolved,
        "wins": wins,
        "loses": loses,
        "nores": nores,
        "winrate": round(wins * 100 / resolved, 2) if resolved else 0.0,
    }


def _weighted_rate(rows):
    wins = sum(row["wins"] for row in rows)
    resolved = sum(row["resolved"] for row in rows)
    return round(wins * 100 / resolved, 2) if resolved else 0.0


def _config_key(row):
    return (
        row["group"],
        row["buy_threshold"],
        row["sell_threshold"],
        row["entry_mode"],
        row["min_phase"],
        row.get("side_mode", "both"),
        row.get("session_mode", "all"),
    )


def _completed_configs():
    if not os.path.exists(GRID_FILE):
        return set()
    try:
        df = pd.read_csv(GRID_FILE)
    except Exception:
        return set()
    return {_config_key(row) for _, row in df.iterrows()}


def _append_grid_row(row):
    exists = os.path.exists(GRID_FILE)
    pd.DataFrame([row]).to_csv(GRID_FILE, mode="a", index=False, header=not exists, encoding="utf-8-sig")


def _candidate_configs():
    thresholds = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
    entry_modes = ["signal_any", "ma20_zone", "ma20_confirm", "bb_reversion", "trend_pullback"]
    side_modes = ["both", "buy_only", "sell_only"]
    session_modes = ["all", "active", "london", "ny", "london_ny", "h1_all", "h1_active", "h1_ny"]
    for group_name in FEATURE_GROUPS:
        for buy_th in thresholds:
            for sell_th in thresholds:
                for entry_mode in entry_modes:
                    for min_phase in [0, 1]:
                        for side_mode in side_modes:
                            for session_mode in session_modes:
                                yield {
                                    "group": group_name,
                                    "buy_threshold": buy_th,
                                    "sell_threshold": sell_th,
                                    "entry_mode": entry_mode,
                                    "min_phase": min_phase,
                                    "side_mode": side_mode,
                                    "session_mode": session_mode,
                                }


def _evaluate_config(years, scores, config, split_years):
    rows = []
    group_scores = scores[config["group"]]
    for year in split_years:
        data = years[year]["data"]
        orders = _build_orders(
            data,
            group_scores[year],
            config["buy_threshold"],
            config["sell_threshold"],
            config["entry_mode"],
            config["min_phase"],
            config.get("side_mode", "both"),
            config.get("session_mode", "all"),
        )
        stats = _simulate_mophong(data, orders)
        stats["year"] = year
        rows.append(stats)
    return rows


def _evaluate_config_proxy(years, scores, config, split_years):
    rows = []
    group_scores = scores[config["group"]]
    for year in split_years:
        data = years[year]["data"]
        orders = _build_orders(
            data,
            group_scores[year],
            config["buy_threshold"],
            config["sell_threshold"],
            config["entry_mode"],
            config["min_phase"],
            config.get("side_mode", "both"),
            config.get("session_mode", "all"),
        )
        buy_mask = orders == 1
        sell_mask = orders == -1
        buy_labels = years[year]["buy_label"]
        sell_labels = years[year]["sell_label"]
        buy_resolved = buy_mask & (buy_labels != 0)
        sell_resolved = sell_mask & (sell_labels != 0)
        wins = int(np.count_nonzero(buy_resolved & (buy_labels == 1)) + np.count_nonzero(sell_resolved & (sell_labels == 1)))
        loses = int(np.count_nonzero(buy_resolved & (buy_labels == -1)) + np.count_nonzero(sell_resolved & (sell_labels == -1)))
        resolved = wins + loses
        rows.append(
            {
                "year": year,
                "raw_deals": int(np.count_nonzero(orders)),
                "results": resolved,
                "resolved": resolved,
                "wins": wins,
                "loses": loses,
                "nores": int(np.count_nonzero(orders) - resolved),
                "winrate": round(wins * 100 / resolved, 2) if resolved else 0.0,
            }
        )
    return rows


def _summarize_rows(rows):
    return {
        "min_winrate": min(row["winrate"] for row in rows),
        "weighted_winrate": _weighted_rate(rows),
        "min_resolved": min(row["resolved"] for row in rows),
        "total_resolved": sum(row["resolved"] for row in rows),
        "total_wins": sum(row["wins"] for row in rows),
        "total_loses": sum(row["loses"] for row in rows),
    }


def _search_train_configs(years, scores, max_configs=None):
    done = _completed_configs()
    count = 0
    for config in _candidate_configs():
        if _config_key(config) in done:
            continue
        started = time.time()
        train_rows = _evaluate_config(years, scores, config, TRAIN_YEARS)
        summary = _summarize_rows(train_rows)
        row = {
            **config,
            "split": "train",
            **{f"train_{key}": value for key, value in summary.items()},
            "seconds": round(time.time() - started, 2),
        }
        for year_row in train_rows:
            row[f"wr_{year_row['year']}"] = year_row["winrate"]
            row[f"deals_{year_row['year']}"] = year_row["resolved"]
        _append_grid_row(row)
        count += 1
        print(f"Train config {count}: {row}")
        if max_configs and count >= max_configs:
            break


def _search_proxy_configs(years, scores, force=False):
    if os.path.exists(PROXY_FILE) and not force:
        print(f"Loaded proxy grid: {PROXY_FILE}")
        return pd.read_csv(PROXY_FILE)
    rows = []
    if force and os.path.exists(PROXY_FILE):
        os.remove(PROXY_FILE)
    started = time.time()
    for idx, config in enumerate(_candidate_configs(), start=1):
        train_rows = _evaluate_config_proxy(years, scores, config, TRAIN_YEARS)
        summary = _summarize_rows(train_rows)
        row = {
            **config,
            **{f"proxy_train_{key}": value for key, value in summary.items()},
        }
        for year_row in train_rows:
            row[f"proxy_wr_{year_row['year']}"] = year_row["winrate"]
            row[f"proxy_deals_{year_row['year']}"] = year_row["resolved"]
        rows.append(row)
        if idx % 100 == 0:
            print(f"Proxy configs {idx}, elapsed={round(time.time() - started, 1)}s")
        if idx % 1000 == 0:
            batch = pd.DataFrame(rows)
            batch.to_csv(PROXY_FILE, index=False, encoding="utf-8-sig")
    proxy = pd.DataFrame(rows)
    proxy.to_csv(PROXY_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved proxy grid: {PROXY_FILE}")
    return proxy


def _best_train_configs(limit=20):
    df = pd.read_csv(GRID_FILE)
    eligible = df[(df["train_min_resolved"] >= 3000) & (df["train_min_winrate"] >= 55)].copy()
    if eligible.empty:
        eligible = df.copy()
    return eligible.sort_values(
        ["train_min_winrate", "train_weighted_winrate", "train_min_resolved", "train_total_resolved"],
        ascending=[False, False, False, False],
    ).head(limit).to_dict("records")


def _best_proxy_configs(proxy, limit=10):
    h1 = proxy[
        proxy["session_mode"].astype(str).str.startswith("h1_")
        & (proxy["proxy_train_min_resolved"] >= 3000)
        & (proxy["proxy_train_min_winrate"] >= 55)
    ].copy()
    if not h1.empty:
        return h1.sort_values(
            [
                "proxy_train_min_winrate",
                "proxy_train_weighted_winrate",
                "proxy_train_min_resolved",
                "proxy_train_total_resolved",
            ],
            ascending=[False, False, False, False],
        ).head(limit).to_dict("records")

    eligible = proxy[
        (proxy["proxy_train_min_resolved"] >= 10000)
        & (proxy["proxy_train_min_winrate"] >= 55)
    ].copy()
    if eligible.empty:
        eligible = proxy[
            (proxy["proxy_train_min_resolved"] >= 30000)
            & (proxy["proxy_train_min_winrate"] >= 54)
        ].copy()
    if eligible.empty:
        eligible = proxy.copy()
    return eligible.sort_values(
        [
            "proxy_train_min_winrate",
            "proxy_train_weighted_winrate",
            "proxy_train_min_resolved",
            "proxy_train_total_resolved",
        ],
        ascending=[False, False, False, False],
    ).head(limit).to_dict("records")


def _evaluate_best_on_test(years, scores, limit=20):
    rows = []
    best_configs = _best_train_configs(limit=limit)
    for config in best_configs:
        clean_config = {
            "group": config["group"],
            "buy_threshold": float(config["buy_threshold"]),
            "sell_threshold": float(config["sell_threshold"]),
            "entry_mode": config["entry_mode"],
            "min_phase": int(config["min_phase"]),
            "side_mode": config.get("side_mode", "both"),
            "session_mode": config.get("session_mode", "all"),
        }
        train_rows = _evaluate_config(years, scores, clean_config, TRAIN_YEARS)
        test_rows = _evaluate_config(years, scores, clean_config, TEST_YEARS)
        row = {
            **clean_config,
            **{f"train_{key}": value for key, value in _summarize_rows(train_rows).items()},
            **{f"test_{key}": value for key, value in _summarize_rows(test_rows).items()},
        }
        for year_row in test_rows:
            row[f"test_wr_{year_row['year']}"] = year_row["winrate"]
            row[f"test_deals_{year_row['year']}"] = year_row["resolved"]
        rows.append(row)
        print(f"Test config: {row}")
        if row["test_min_resolved"] >= 3000 and row["test_min_winrate"] >= 55:
            with open(BEST_FILE, "w", encoding="utf-8") as fh:
                json.dump({"found": True, "config": row}, fh, indent=2, ensure_ascii=False)
            break
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    if rows and not os.path.exists(BEST_FILE):
        with open(BEST_FILE, "w", encoding="utf-8") as fh:
            json.dump({"found": False, "best_checked": rows[0]}, fh, indent=2, ensure_ascii=False)
    return summary


def _evaluate_proxy_top_with_mophong(years, scores, proxy, limit=10):
    rows = []
    done = set()
    if os.path.exists(SUMMARY_FILE):
        old = pd.read_csv(SUMMARY_FILE)
        rows = old.to_dict("records")
        done = {_config_key(row) for row in rows}
    for config in _best_proxy_configs(proxy, limit=limit):
        clean_config = {
            "group": config["group"],
            "buy_threshold": float(config["buy_threshold"]),
            "sell_threshold": float(config["sell_threshold"]),
            "entry_mode": config["entry_mode"],
            "min_phase": int(config["min_phase"]),
            "side_mode": config.get("side_mode", "both"),
            "session_mode": config.get("session_mode", "all"),
        }
        if _config_key(clean_config) in done:
            continue
        started = time.time()
        train_rows = _evaluate_config(years, scores, clean_config, TRAIN_YEARS)
        test_rows = _evaluate_config(years, scores, clean_config, TEST_YEARS)
        row = {
            **clean_config,
            **{f"train_{key}": value for key, value in _summarize_rows(train_rows).items()},
            **{f"test_{key}": value for key, value in _summarize_rows(test_rows).items()},
            "official_seconds": round(time.time() - started, 2),
            "proxy_train_min_winrate": config.get("proxy_train_min_winrate"),
            "proxy_train_min_resolved": config.get("proxy_train_min_resolved"),
        }
        for year_row in test_rows:
            row[f"test_wr_{year_row['year']}"] = year_row["winrate"]
            row[f"test_deals_{year_row['year']}"] = year_row["resolved"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
        print(f"Official MoPhongDeals config: {row}")
        if row["test_min_resolved"] >= 3000 and row["test_min_winrate"] >= 55:
            with open(BEST_FILE, "w", encoding="utf-8") as fh:
                json.dump({"found": True, "config": row}, fh, indent=2, ensure_ascii=False)
            break
    if rows:
        best_row = sorted(rows, key=lambda item: (item["test_min_winrate"], item["test_weighted_winrate"], item["test_min_resolved"]), reverse=True)[0]
        with open(BEST_FILE, "w", encoding="utf-8") as fh:
            json.dump({"found": False, "best_checked": best_row}, fh, indent=2, ensure_ascii=False)
    return pd.DataFrame(rows)


def smoke_test():
    sample = _prepare_features(_load_year_data(2024).head(8000))
    missing = sorted({col for cols in FEATURE_GROUPS.values() for col in cols if col not in sample.columns})
    if missing:
        raise RuntimeError(f"Missing features: {missing}")
    orders = np.zeros(len(sample), dtype=np.int8)
    orders[300:500:3] = 1
    stats = _simulate_mophong(sample, orders)
    print({"rows": len(sample), "feature_groups": len(FEATURE_GROUPS), "mophong_stats": stats})


def run(force=False, max_configs=None, test_limit=20, smoke=False, proxy_top=10, official_grid=False):
    if smoke:
        smoke_test()
        return
    years = _load_prepared_years(force=force)
    models = _train_models(years, force=force)
    scores = _score_years(years, models, force=force)
    proxy = _search_proxy_configs(years, scores, force=force)
    summary = _evaluate_proxy_top_with_mophong(years, scores, proxy, limit=proxy_top)
    if not summary.empty:
        print(summary.to_string(index=False))
    if official_grid:
        _search_train_configs(years, scores, max_configs=max_configs)
    if official_grid and os.path.exists(GRID_FILE):
        summary = _evaluate_best_on_test(years, scores, limit=test_limit)
        print(summary.to_string(index=False))
        print("grid", GRID_FILE)
        print("summary", SUMMARY_FILE)
        print("best", BEST_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=20)
    parser.add_argument("--proxy-top", type=int, default=10)
    parser.add_argument("--official-grid", action="store_true")
    args = parser.parse_args()
    run(
        force=args.force,
        max_configs=args.max_configs,
        test_limit=args.test_limit,
        smoke=args.smoke,
        proxy_top=args.proxy_top,
        official_grid=args.official_grid,
    )
