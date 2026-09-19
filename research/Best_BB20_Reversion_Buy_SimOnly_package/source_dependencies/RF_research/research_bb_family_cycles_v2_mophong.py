import argparse
import json
import logging
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


CURRENT_DIR = os.path.dirname(__file__)

from vendor.scj.data_handler import CLOSE, HIGH, LOW, OPEN
from method.CycleFeatureBuilder import build_cycle_features_from_signal, create_cycle_features
from . import research_group_indicators_mophong as base
from .research_pending_entry_labels_mophong import _tp_sl_pending_entry


warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.CRITICAL)

MODEL_DIR = os.path.join(CURRENT_DIR, "../RF_Group_indicators/models")
PREPARED_FILE = os.path.join(MODEL_DIR, "prepared_years_bb_family_v2.joblib")
MODEL_FILE = os.path.join(MODEL_DIR, "bb_family_v2_models.joblib")
SCORE_FILE = os.path.join(MODEL_DIR, "bb_family_v2_scores.joblib")
PROXY_FILE = os.path.join(CURRENT_DIR, "../RF_Group_indicators/PROXY_RF_BBFamilyV2.csv")
SUMMARY_FILE = os.path.join(CURRENT_DIR, "SUMMARY_RF_BBFamilyV2_MoPhongDeals.csv")
BEST_FILE = os.path.join(CURRENT_DIR, "BEST_RF_BBFamilyV2_MoPhongDeals.json")

BB_PERIODS = [10, 20, 50, 100, 200]
YEAR_WEIGHTS = {2019: 1.0, 2020: 1.0, 2021: 1.0, 2022: 1.0, 2023: 1.0}

BASE_CONTEXT = [
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
    "ATR14",
    "atr_rank",
]


def _cycle_cols(prefix):
    return [
        f"{prefix}_signal_dir",
        f"{prefix}_bar",
        f"{prefix}_bar_clip",
        f"{prefix}_bar_log",
        f"{prefix}_start",
        f"{prefix}_changed",
        f"{prefix}_age_0_3",
        f"{prefix}_age_4_10",
        f"{prefix}_age_11_30",
        f"{prefix}_age_31_60",
        f"{prefix}_age_61_plus",
        f"{prefix}_price_extension",
        f"{prefix}_first_touch_ref",
        f"{prefix}_touch_ref_count_clip",
    ]


def _bb_cols(period):
    p = f"bb{period}"
    return [
        f"{p}_pos",
        f"{p}_width_r",
        f"{p}_width_rank",
        f"{p}_z",
        f"{p}_dist_mid_r",
        f"{p}_dist_upper_r",
        f"{p}_dist_lower_r",
        f"{p}_squeeze",
        f"{p}_break_upper",
        f"{p}_break_lower",
    ]


FEATURE_GROUPS = {}
GROUP_META = {}
for _period in [10, 20, 50, 100]:
    _p = f"bb{_period}"
    FEATURE_GROUPS[f"{_p}_mid_cycles"] = BASE_CONTEXT + _bb_cols(_period) + _cycle_cols(f"{_p}_mid") + _cycle_cols(f"{_p}_width")
    GROUP_META[f"{_p}_mid_cycles"] = {"period": _period, "signal_col": f"{_p}_mid_signal_dir", "bar_col": f"{_p}_mid_bar"}

    FEATURE_GROUPS[f"{_p}_reversion_cycles"] = (
        BASE_CONTEXT
        + _bb_cols(_period)
        + _cycle_cols(f"{_p}_reversion")
        + _cycle_cols(f"{_p}_lower")
        + _cycle_cols(f"{_p}_upper")
        + _cycle_cols(f"{_p}_width")
    )
    GROUP_META[f"{_p}_reversion_cycles"] = {"period": _period, "signal_col": f"{_p}_reversion_signal_dir", "bar_col": f"{_p}_reversion_bar"}

    FEATURE_GROUPS[f"{_p}_squeeze_cycles"] = BASE_CONTEXT + _bb_cols(_period) + _cycle_cols(f"{_p}_squeeze_dir") + _cycle_cols(f"{_p}_width")
    GROUP_META[f"{_p}_squeeze_cycles"] = {"period": _period, "signal_col": f"{_p}_squeeze_dir_signal_dir", "bar_col": f"{_p}_squeeze_dir_bar"}


def _add_bb_family(data):
    close = data[CLOSE].astype(float)
    extra = {}
    for period in BB_PERIODS:
        p = f"bb{period}"
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        width = upper - lower
        extra[f"{p}_mid"] = mid
        extra[f"{p}_upper"] = upper
        extra[f"{p}_lower"] = lower
        extra[f"{p}_pos"] = (close - lower) / width.replace(0, np.nan)
        extra[f"{p}_width_r"] = width / data["norm_scale"].replace(0, np.nan)
        extra[f"{p}_width_rank"] = width.rolling(2000, min_periods=200).rank(pct=True)
        extra[f"{p}_z"] = (close - mid) / std.replace(0, np.nan)
        extra[f"{p}_dist_mid_r"] = (close - mid) / data["norm_scale"].replace(0, np.nan)
        extra[f"{p}_dist_upper_r"] = (close - upper) / data["norm_scale"].replace(0, np.nan)
        extra[f"{p}_dist_lower_r"] = (close - lower) / data["norm_scale"].replace(0, np.nan)
        extra[f"{p}_squeeze"] = (extra[f"{p}_width_rank"] <= 0.20).astype(float)
        extra[f"{p}_break_upper"] = (close > upper).astype(float)
        extra[f"{p}_break_lower"] = (close < lower).astype(float)
        extra[f"{p}_reversion_raw"] = np.where(extra[f"{p}_pos"] <= 0.20, 1, np.where(extra[f"{p}_pos"] >= 0.80, -1, 0))
        extra[f"{p}_squeeze_dir_raw"] = np.where(
            (extra[f"{p}_width_rank"] <= 0.35) & (extra[f"{p}_pos"] >= 0.80),
            1,
            np.where((extra[f"{p}_width_rank"] <= 0.35) & (extra[f"{p}_pos"] <= 0.20), -1, 0),
        )
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    frames = [data.reset_index(drop=True)]

    for period in BB_PERIODS:
        p = f"bb{period}"
        frames.append(
            create_cycle_features(
                data,
                "price_line_cross",
                prefix=f"{p}_mid",
                price_col=CLOSE,
                line_col=f"{p}_mid",
                reference_col=f"{p}_mid",
                norm_col="norm_scale",
            )
        )
        frames.append(
            create_cycle_features(
                data,
                "price_line_cross",
                prefix=f"{p}_upper",
                price_col=CLOSE,
                line_col=f"{p}_upper",
                reference_col=f"{p}_upper",
                norm_col="norm_scale",
            )
        )
        frames.append(
            create_cycle_features(
                data,
                "price_line_cross",
                prefix=f"{p}_lower",
                price_col=CLOSE,
                line_col=f"{p}_lower",
                reference_col=f"{p}_lower",
                norm_col="norm_scale",
            )
        )
        data[f"{p}_width_signal"] = np.where(data[f"{p}_width_rank"] >= 0.50, 1, -1)
        frames.append(
            create_cycle_features(
                data,
                "signal_col",
                prefix=f"{p}_width",
                signal_col=f"{p}_width_signal",
                reference_col=f"{p}_mid",
                norm_col="norm_scale",
            )
        )
        frames.append(
            create_cycle_features(
                data,
                "signal_col",
                prefix=f"{p}_reversion",
                signal_col=f"{p}_reversion_raw",
                reference_col=f"{p}_mid",
                norm_col="norm_scale",
            )
        )
        frames.append(
            create_cycle_features(
                data,
                "signal_col",
                prefix=f"{p}_squeeze_dir",
                signal_col=f"{p}_squeeze_dir_raw",
                reference_col=f"{p}_mid",
                norm_col="norm_scale",
            )
        )
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)


def _prepare_features(raw):
    data = base._prepare_features(raw)
    data = _add_bb_family(data)
    return data


def _load_prepared_years(force=False):
    if os.path.exists(PREPARED_FILE) and not force:
        print(f"Loaded BB prepared years: {PREPARED_FILE}")
        return joblib.load(PREPARED_FILE)
    years = {}
    for year in base.ALL_YEARS:
        data = _prepare_features(base._load_year_data(year))
        labels = _tp_sl_pending_entry(
            data[OPEN].to_numpy(np.float64),
            data[HIGH].to_numpy(np.float64),
            data[LOW].to_numpy(np.float64),
            data[CLOSE].to_numpy(np.float64),
            float(base.R1),
            int(base.EXPIRED),
            int(base.MAX_FORWARD),
        )
        years[year] = {"data": data, "buy_label": labels[0], "sell_label": labels[1]}
        print(f"Prepared BB {year}: candles={len(data)}")
    joblib.dump(years, PREPARED_FILE)
    return years


def _rf_model(group, side):
    seed = abs(hash(("bb_family", group, side))) % 100000
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=500,
        min_samples_split=1000,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.80,
        n_jobs=-1,
        random_state=seed,
    )


def _dataset(years, group, side):
    xs, ys, ws = [], [], []
    label_key = "buy_label" if side == "buy" else "sell_label"
    direction = 1 if side == "buy" else -1
    features = FEATURE_GROUPS[group]
    for year in base.TRAIN_YEARS:
        data = years[year]["data"]
        labels = years[year][label_key]
        frame = data[features]
        valid = (~frame.isna().any(axis=1)).to_numpy()
        signal_values = data[GROUP_META[group]["signal_col"]]
        if isinstance(signal_values, pd.DataFrame):
            signal_values = signal_values.iloc[:, 0]
        indexes = np.where((labels != 0) & valid & (signal_values.to_numpy(np.int8) == direction))[0]
        if len(indexes) > 120000:
            rng = np.random.default_rng(year + (313 if side == "buy" else 1313))
            indexes = rng.choice(indexes, 120000, replace=False)
        x = frame.fillna(0).to_numpy(float)[indexes]
        y = (labels[indexes] == 1).astype(np.int8)
        xs.append(x)
        ys.append(y)
        ws.append(np.full(len(y), YEAR_WEIGHTS[year], dtype=float))
        print(f"BB dataset {group}/{side}/{year}: {len(y)} wr={round(float(y.mean()*100),2) if len(y) else 0}")
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)


def _train_models(years, force=False):
    if os.path.exists(MODEL_FILE) and not force:
        print(f"Loaded BB models: {MODEL_FILE}")
        return joblib.load(MODEL_FILE)
    bundle = {}
    for group in FEATURE_GROUPS:
        bundle[group] = {"features": FEATURE_GROUPS[group]}
        for side in ["buy", "sell"]:
            x, y, w = _dataset(years, group, side)
            model = _rf_model(group, side)
            model.fit(x, y, sample_weight=w)
            bundle[group][side] = model
            print(f"Trained BB {group}/{side}: samples={len(y)} wr={round(float(y.mean()*100),2)}")
    joblib.dump(bundle, MODEL_FILE)
    return bundle


def _score_years(years, models, force=False):
    if os.path.exists(SCORE_FILE) and not force:
        print(f"Loaded BB scores: {SCORE_FILE}")
        return joblib.load(SCORE_FILE)
    scores = {}
    for group, model_bundle in models.items():
        scores[group] = {}
        features = model_bundle["features"]
        for year in base.ALL_YEARS:
            data = years[year]["data"]
            frame = data[features]
            valid = (~frame.isna().any(axis=1)).to_numpy()
            x = frame.fillna(0).to_numpy(float)
            scores[group][year] = {
                "valid": valid,
                "buy": model_bundle["buy"].predict_proba(x)[:, 1],
                "sell": model_bundle["sell"].predict_proba(x)[:, 1],
            }
            print(f"Scored BB {group}/{year}")
    joblib.dump(scores, SCORE_FILE)
    return scores


def _entry_mask(data, config):
    entry_mode = config["entry_mode"]
    meta = GROUP_META[config["group"]]
    period = meta["period"]
    p = f"bb{period}"

    def col(name, dtype=float):
        values = data[name]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        return values.to_numpy(dtype)

    bar = col(meta["bar_col"], np.int32)
    pos = col(f"{p}_pos")
    z = col(f"{p}_z")
    width_rank = col(f"{p}_width_rank")
    signal = col(meta["signal_col"], np.int8)
    first_mid = col(f"{p}_mid_first_touch_ref") == 1
    touch_mid = col(f"{p}_mid_touch_ref_count_clip")

    agree = np.ones(len(data), dtype=bool)
    if period != 200 and "bb200_pos" in data.columns:
        pos200 = col("bb200_pos")
        agree = ((signal == 1) & (pos200 >= 0.50)) | ((signal == -1) & (pos200 <= 0.50))

    if entry_mode == "bb_any":
        return np.ones(len(data), dtype=bool)
    if entry_mode == "mid_cycle_start":
        return bar <= 3
    if entry_mode == "mid_cycle_4_10":
        return (bar >= 4) & (bar <= 10)
    if entry_mode == "mid_cycle_11_30":
        return (bar >= 11) & (bar <= 30)
    if entry_mode == "first_mid_touch":
        return first_mid
    if entry_mode == "pullback_mid":
        return (bar >= 3) & (bar <= 30) & (touch_mid >= 1) & (pos >= 0.35) & (pos <= 0.65)
    if entry_mode == "lower_reversion":
        return (pos <= 0.20) & (z <= -0.8)
    if entry_mode == "upper_reversion":
        return (pos >= 0.80) & (z >= 0.8)
    if entry_mode == "outer_reversion":
        return ((signal == 1) & (pos <= 0.25)) | ((signal == -1) & (pos >= 0.75))
    if entry_mode == "squeeze_breakout":
        return (width_rank <= 0.35) & (((signal == 1) & (pos >= 0.70)) | ((signal == -1) & (pos <= 0.30)))
    if entry_mode == "bb20_bb50_agree":
        return agree
    raise ValueError(f"Unsupported entry_mode: {entry_mode}")


def _build_orders(data, score, config):
    signal_values = data[GROUP_META[config["group"]]["signal_col"]]
    if isinstance(signal_values, pd.DataFrame):
        signal_values = signal_values.iloc[:, 0]
    signal = signal_values.to_numpy(np.int8)
    mask = score["valid"] & _entry_mask(data, config) & base._session_mask(data, config["session_mode"])
    orders = np.zeros(len(data), dtype=np.int8)
    if config["side_mode"] in ["both", "buy_only"]:
        orders[mask & (signal == 1) & (score["buy"] >= config["buy_threshold"])] = 1
    if config["side_mode"] in ["both", "sell_only"]:
        orders[mask & (signal == -1) & (score["sell"] >= config["sell_threshold"])] = -1
    orders[:250] = 0
    orders[-2:] = 0
    return orders


def _candidate_configs():
    thresholds = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
    entry_modes = [
        "mid_cycle_start",
        "mid_cycle_4_10",
        "mid_cycle_11_30",
        "first_mid_touch",
        "pullback_mid",
        "lower_reversion",
        "upper_reversion",
        "outer_reversion",
        "squeeze_breakout",
        "bb20_bb50_agree",
    ]
    side_modes = ["both", "buy_only", "sell_only"]
    session_modes = ["london", "ny", "london_ny"]
    for group in FEATURE_GROUPS:
        for buy_th in thresholds:
            for sell_th in thresholds:
                for entry_mode in entry_modes:
                    for side_mode in side_modes:
                        for session_mode in session_modes:
                            yield {
                                "group": group,
                                "buy_threshold": buy_th,
                                "sell_threshold": sell_th,
                                "entry_mode": entry_mode,
                                "side_mode": side_mode,
                                "session_mode": session_mode,
                            }


def _evaluate_proxy(years, scores, config, split_years):
    rows = []
    for year in split_years:
        data = years[year]["data"]
        orders = _build_orders(data, scores[config["group"]][year], config)
        buy_mask = orders == 1
        sell_mask = orders == -1
        buy_labels = years[year]["buy_label"]
        sell_labels = years[year]["sell_label"]
        buy_resolved = buy_mask & (buy_labels != 0)
        sell_resolved = sell_mask & (sell_labels != 0)
        wins = int(np.count_nonzero(buy_resolved & (buy_labels == 1)) + np.count_nonzero(sell_resolved & (sell_labels == 1)))
        loses = int(np.count_nonzero(buy_resolved & (buy_labels == -1)) + np.count_nonzero(sell_resolved & (sell_labels == -1)))
        resolved = wins + loses
        rows.append({"year": year, "resolved": resolved, "wins": wins, "loses": loses, "winrate": round(wins * 100 / resolved, 2) if resolved else 0.0})
    return rows


def _summarize(rows):
    wins = sum(row["wins"] for row in rows)
    resolved = sum(row["resolved"] for row in rows)
    return {
        "min_winrate": min(row["winrate"] for row in rows),
        "weighted_winrate": round(wins * 100 / resolved, 2) if resolved else 0.0,
        "min_resolved": min(row["resolved"] for row in rows),
        "total_resolved": resolved,
        "total_wins": wins,
        "total_loses": sum(row["loses"] for row in rows),
    }


def _search_proxy(years, scores, force=False):
    if os.path.exists(PROXY_FILE) and not force:
        print(f"Loaded BB proxy: {PROXY_FILE}")
        return pd.read_csv(PROXY_FILE)
    rows = []
    started = time.time()
    for idx, config in enumerate(_candidate_configs(), start=1):
        train_rows = _evaluate_proxy(years, scores, config, base.TRAIN_YEARS)
        summary = _summarize(train_rows)
        row = {**config, **{f"proxy_train_{key}": value for key, value in summary.items()}}
        for year_row in train_rows:
            row[f"proxy_wr_{year_row['year']}"] = year_row["winrate"]
            row[f"proxy_deals_{year_row['year']}"] = year_row["resolved"]
        rows.append(row)
        if idx % 1000 == 0:
            pd.DataFrame(rows).to_csv(PROXY_FILE, index=False, encoding="utf-8-sig")
            print(f"BB proxy {idx}, elapsed={round(time.time() - started, 1)}s")
    proxy = pd.DataFrame(rows)
    proxy.to_csv(PROXY_FILE, index=False, encoding="utf-8-sig")
    return proxy


def _key(row):
    return (row["group"], float(row["buy_threshold"]), float(row["sell_threshold"]), row["entry_mode"], row["side_mode"], row["session_mode"])


def _best_proxy(proxy, limit):
    eligible = proxy[(proxy["proxy_train_min_resolved"] >= 3000) & (proxy["proxy_train_min_winrate"] >= 55)].copy()
    volume_floor = eligible[eligible["proxy_train_min_resolved"] >= 15000].copy()
    if not volume_floor.empty:
        eligible = volume_floor
    if eligible.empty:
        eligible = proxy[proxy["proxy_train_min_resolved"] >= 3000].copy()
    practical = eligible[eligible["proxy_train_min_resolved"] <= 60000].copy()
    if not practical.empty:
        eligible = practical
    if eligible.empty:
        eligible = proxy.copy()
    diversified = []
    used = set()
    for _, row in eligible.sort_values(
        ["proxy_train_min_winrate", "proxy_train_weighted_winrate", "proxy_train_min_resolved"],
        ascending=[False, False, False],
    ).iterrows():
        family_key = (row["group"], row["entry_mode"], row["side_mode"], row["session_mode"])
        if family_key in used and len(diversified) < limit // 2:
            continue
        diversified.append(row.to_dict())
        used.add(family_key)
        if len(diversified) >= limit:
            break
    return diversified


def _evaluate_mophong(years, scores, config, split_years):
    rows = []
    for year in split_years:
        data = years[year]["data"]
        orders = _build_orders(data, scores[config["group"]][year], config)
        stats = base._simulate_mophong(data, orders)
        stats["year"] = year
        rows.append(stats)
    return rows


def _evaluate_top(years, scores, proxy, limit):
    rows = []
    done = set()
    if os.path.exists(SUMMARY_FILE):
        old = pd.read_csv(SUMMARY_FILE)
        rows = old.to_dict("records")
        done = {_key(row) for row in rows}
    for config in _best_proxy(proxy, limit):
        clean = {
            "group": config["group"],
            "buy_threshold": float(config["buy_threshold"]),
            "sell_threshold": float(config["sell_threshold"]),
            "entry_mode": config["entry_mode"],
            "side_mode": config["side_mode"],
            "session_mode": config["session_mode"],
        }
        if _key(clean) in done:
            continue
        started = time.time()
        test_rows = _evaluate_mophong(years, scores, clean, base.TEST_YEARS)
        test_summary = _summarize(test_rows)
        train_rows = []
        train_summary = {
            "min_winrate": 0.0,
            "weighted_winrate": 0.0,
            "min_resolved": 0,
            "total_resolved": 0,
            "total_wins": 0,
            "total_loses": 0,
        }
        if test_summary["min_resolved"] >= 3000 and test_summary["min_winrate"] >= 55:
            train_rows = _evaluate_mophong(years, scores, clean, base.TRAIN_YEARS)
            train_summary = _summarize(train_rows)
        row = {
            **clean,
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"test_{key}": value for key, value in test_summary.items()},
            "official_seconds": round(time.time() - started, 2),
            "proxy_train_min_winrate": config.get("proxy_train_min_winrate"),
            "proxy_train_min_resolved": config.get("proxy_train_min_resolved"),
        }
        for year_row in test_rows:
            row[f"test_wr_{year_row['year']}"] = year_row["winrate"]
            row[f"test_deals_{year_row['year']}"] = year_row["resolved"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
        print(f"BB official: {row}")
        if row["test_min_resolved"] >= 3000 and row["test_min_winrate"] >= 55:
            with open(BEST_FILE, "w", encoding="utf-8") as fh:
                json.dump({"found": True, "config": row}, fh, indent=2, ensure_ascii=False)
            return pd.DataFrame(rows)
    if rows:
        best = sorted(rows, key=lambda row: (row["test_min_winrate"], row["test_weighted_winrate"], row["test_min_resolved"]), reverse=True)[0]
        with open(BEST_FILE, "w", encoding="utf-8") as fh:
            json.dump({"found": False, "best_checked": best}, fh, indent=2, ensure_ascii=False)
    return pd.DataFrame(rows)


def smoke_test():
    data = _prepare_features(base._load_year_data(2025).head(5000))
    missing = sorted({col for cols in FEATURE_GROUPS.values() for col in cols if col not in data.columns})
    if missing:
        raise RuntimeError(f"Missing BB features: {missing}")
    print({"rows": len(data), "groups": list(FEATURE_GROUPS)[:4], "bb20_reversion_bar_max": int(data["bb20_reversion_bar"].max())})


def run(force=False, proxy_top=40, smoke=False):
    if smoke:
        smoke_test()
        return
    years = _load_prepared_years(force=force)
    models = _train_models(years, force=force)
    scores = _score_years(years, models, force=force)
    proxy = _search_proxy(years, scores, force=force)
    summary = _evaluate_top(years, scores, proxy, proxy_top)
    if not summary.empty:
        cols = [col for col in [
            "group",
            "buy_threshold",
            "sell_threshold",
            "entry_mode",
            "side_mode",
            "session_mode",
            "train_min_winrate",
            "train_min_resolved",
            "test_min_winrate",
            "test_weighted_winrate",
            "test_min_resolved",
            "test_wr_2024",
            "test_deals_2024",
            "test_wr_2025",
            "test_deals_2025",
            "test_wr_2026",
            "test_deals_2026",
        ] if col in summary.columns]
        print(summary.sort_values(["test_min_winrate", "test_weighted_winrate"], ascending=[False, False])[cols].head(20).to_string(index=False))
    print("proxy", PROXY_FILE)
    print("summary", SUMMARY_FILE)
    print("best", BEST_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--proxy-top", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(force=args.force, proxy_top=args.proxy_top, smoke=args.smoke)
