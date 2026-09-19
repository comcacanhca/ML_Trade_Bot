import argparse
import json
import logging
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data import CLOSE, HIGH, LOW, OPEN
import research_group_indicators_mophong as base


warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.CRITICAL)

MODEL_DIR = os.path.join(CURRENT_DIR, "models")
PREPARED_FILE = os.path.join(MODEL_DIR, "prepared_years_expired5.joblib")
MODEL_FILE = os.path.join(MODEL_DIR, "expired5_models.joblib")
SCORE_FILE = os.path.join(MODEL_DIR, "expired5_scores.joblib")
PROXY_FILE = os.path.join(CURRENT_DIR, "PROXY_RF_Expired5Labels.csv")
SUMMARY_FILE = os.path.join(CURRENT_DIR, "SUMMARY_RF_Expired5Labels_MoPhongDeals.csv")
BEST_FILE = os.path.join(CURRENT_DIR, "BEST_RF_Expired5Labels_MoPhongDeals.json")

GROUPS = [
    "ma_family",
    "bb_rsi_stoch",
    "adx_di_regime",
    "psar_ma_cycle",
    "entry_timing_zone",
    "ma5_ma20_cycle_order",
    "momentum_volatility",
]
YEAR_WEIGHTS = {2019: 0.50, 2020: 0.75, 2021: 1.00, 2022: 1.25, 2023: 1.50}


def _load_prepared_years(force=False):
    if os.path.exists(PREPARED_FILE) and not force:
        print(f"Loaded expired5 prepared years: {PREPARED_FILE}")
        return joblib.load(PREPARED_FILE)
    source = base._load_prepared_years(force=False)
    years = {}
    max_forward = base.EXPIRED + 2
    for year in base.ALL_YEARS:
        data = source[year]["data"]
        labels = base._tp_sl_next_open(
            data[OPEN].to_numpy(np.float64),
            data[HIGH].to_numpy(np.float64),
            data[LOW].to_numpy(np.float64),
            data[CLOSE].to_numpy(np.float64),
            float(base.R1),
            max_forward,
        )
        years[year] = {"data": data, "buy_label": labels[0], "sell_label": labels[1]}
        buy_resolved = int(np.count_nonzero(labels[0]))
        sell_resolved = int(np.count_nonzero(labels[1]))
        print(f"Expired5 labels {year}: buy_resolved={buy_resolved} sell_resolved={sell_resolved}")
    joblib.dump(years, PREPARED_FILE)
    return years


def _rf_model(group, side):
    seed = abs(hash(("expired5", group, side))) % 100000
    return RandomForestClassifier(
        n_estimators=320,
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
    features = base.FEATURE_GROUPS[group]
    for year in base.TRAIN_YEARS:
        data = years[year]["data"]
        labels = years[year][label_key]
        frame = data[features]
        valid = (~frame.isna().any(axis=1)).to_numpy()
        indexes = np.where((labels != 0) & valid & (data["signal_dir"].to_numpy(np.int8) == direction))[0]
        if len(indexes) > 120000:
            rng = np.random.default_rng(year + (73 if side == "buy" else 173))
            indexes = rng.choice(indexes, 120000, replace=False)
        x = frame.fillna(0).to_numpy(float)[indexes]
        y = (labels[indexes] == 1).astype(np.int8)
        xs.append(x)
        ys.append(y)
        ws.append(np.full(len(y), YEAR_WEIGHTS[year], dtype=float))
        print(f"Expired5 dataset {group}/{side}/{year}: {len(y)} wr={round(float(y.mean()*100),2)}")
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)


def _train_models(years, force=False):
    if os.path.exists(MODEL_FILE) and not force:
        print(f"Loaded expired5 models: {MODEL_FILE}")
        return joblib.load(MODEL_FILE)
    bundle = {}
    for group in GROUPS:
        bundle[group] = {"features": base.FEATURE_GROUPS[group]}
        for side in ["buy", "sell"]:
            x, y, w = _dataset(years, group, side)
            model = _rf_model(group, side)
            model.fit(x, y, sample_weight=w)
            bundle[group][side] = model
            print(f"Trained expired5 {group}/{side}: samples={len(y)} wr={round(float(y.mean()*100),2)}")
    joblib.dump(bundle, MODEL_FILE)
    return bundle


def _score_years(years, models, force=False):
    if os.path.exists(SCORE_FILE) and not force:
        print(f"Loaded expired5 scores: {SCORE_FILE}")
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
            print(f"Scored expired5 {group}/{year}")
    joblib.dump(scores, SCORE_FILE)
    return scores


def _candidate_configs():
    thresholds = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    entry_modes = ["signal_any", "ma20_zone", "ma20_confirm", "trend_pullback", "bb_reversion"]
    side_modes = ["both", "buy_only", "sell_only"]
    session_modes = ["all", "active", "ny", "london_ny", "h1_active", "h1_ny"]
    for group in GROUPS:
        for buy_th in thresholds:
            for sell_th in thresholds:
                for entry_mode in entry_modes:
                    for min_phase in [0, 1]:
                        for side_mode in side_modes:
                            for session_mode in session_modes:
                                yield {
                                    "group": group,
                                    "buy_threshold": buy_th,
                                    "sell_threshold": sell_th,
                                    "entry_mode": entry_mode,
                                    "min_phase": min_phase,
                                    "side_mode": side_mode,
                                    "session_mode": session_mode,
                                }


def _evaluate_proxy(years, scores, config, split_years):
    rows = []
    for year in split_years:
        data = years[year]["data"]
        orders = base._build_orders(
            data,
            scores[config["group"]][year],
            config["buy_threshold"],
            config["sell_threshold"],
            config["entry_mode"],
            config["min_phase"],
            config["side_mode"],
            config["session_mode"],
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
        print(f"Loaded expired5 proxy: {PROXY_FILE}")
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
            print(f"Expired5 proxy {idx}, elapsed={round(time.time() - started, 1)}s")
    proxy = pd.DataFrame(rows)
    proxy.to_csv(PROXY_FILE, index=False, encoding="utf-8-sig")
    return proxy


def _key(row):
    return (
        row["group"],
        float(row["buy_threshold"]),
        float(row["sell_threshold"]),
        row["entry_mode"],
        int(row["min_phase"]),
        row["side_mode"],
        row["session_mode"],
    )


def _best_proxy(proxy, limit):
    eligible = proxy[proxy["proxy_train_min_resolved"] >= 300].copy()
    if eligible.empty:
        eligible = proxy.copy()
    diversified = []
    used = set()
    for _, row in eligible.sort_values(
        ["proxy_train_min_resolved", "proxy_train_min_winrate", "proxy_train_weighted_winrate"],
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
        orders = base._build_orders(
            data,
            scores[config["group"]][year],
            config["buy_threshold"],
            config["sell_threshold"],
            config["entry_mode"],
            config["min_phase"],
            config["side_mode"],
            config["session_mode"],
        )
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
            "min_phase": int(config["min_phase"]),
            "side_mode": config["side_mode"],
            "session_mode": config["session_mode"],
        }
        if _key(clean) in done:
            continue
        started = time.time()
        train_rows = _evaluate_mophong(years, scores, clean, base.TRAIN_YEARS)
        test_rows = _evaluate_mophong(years, scores, clean, base.TEST_YEARS)
        row = {
            **clean,
            **{f"train_{key}": value for key, value in _summarize(train_rows).items()},
            **{f"test_{key}": value for key, value in _summarize(test_rows).items()},
            "official_seconds": round(time.time() - started, 2),
            "proxy_train_min_winrate": config.get("proxy_train_min_winrate"),
            "proxy_train_min_resolved": config.get("proxy_train_min_resolved"),
        }
        for year_row in test_rows:
            row[f"test_wr_{year_row['year']}"] = year_row["winrate"]
            row[f"test_deals_{year_row['year']}"] = year_row["resolved"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
        print(f"Expired5 official: {row}")
        if row["test_min_resolved"] >= 3000 and row["test_min_winrate"] >= 55:
            with open(BEST_FILE, "w", encoding="utf-8") as fh:
                json.dump({"found": True, "config": row}, fh, indent=2, ensure_ascii=False)
            return pd.DataFrame(rows)
    if rows:
        best = sorted(rows, key=lambda row: (row["test_min_winrate"], row["test_weighted_winrate"], row["test_min_resolved"]), reverse=True)[0]
        with open(BEST_FILE, "w", encoding="utf-8") as fh:
            json.dump({"found": False, "best_checked": best}, fh, indent=2, ensure_ascii=False)
    return pd.DataFrame(rows)


def run(force=False, proxy_top=40):
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
    args = parser.parse_args()
    run(force=args.force, proxy_top=args.proxy_top)
