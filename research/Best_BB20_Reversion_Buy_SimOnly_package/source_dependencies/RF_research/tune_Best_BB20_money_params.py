import argparse
import itertools
import logging
import os
import sys
import time

import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
GROUP_DIR = os.path.join(PROJECT_ROOT, "srateries", "RF_Group_indicators")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if GROUP_DIR not in sys.path:
    sys.path.insert(0, GROUP_DIR)

from data import DATE, OPEN
from method.MoPhongDeals import MoPhongDeals
import Best_BB20_Reversion_Buy_SimOnly as best
import research_bb_family_cycles_v2_mophong as bb


logging.disable(logging.CRITICAL)

OUT_DETAIL = os.path.join(CURRENT_DIR, "TUNE_Best_BB20_money_params_detail.csv")
OUT_SUMMARY = os.path.join(CURRENT_DIR, "TUNE_Best_BB20_money_params_summary.csv")

DEFAULT_YEARS = [2024, 2025, 2026]
BUY_THRESHOLDS = [0.50, 0.51, 0.52, 0.53, 0.54]
MAX_RUNNING_GRID = [5, 7, 10, 12, 15]
START_FUND_GRID = [300, 500]
DENSITY_GRID = [0.015, 0.02, 0.025, 0.03]
RUT_GRID = [(20000, 0.05), (10000, 0.05), (20000, 0.10)]
MAX_ADD_FUND_PER_YEAR = 1000.0


def parse_years(value):
    years = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item.strip()) for item in part.split("-", 1)]
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


def key_tuple(row):
    return (
        float(row["buy_threshold"]),
        int(row["max_running_deals"]),
        float(row["start_fund"]),
        float(row["density"]),
        float(row["rut_level"]),
        float(row["rut_pct"]),
        int(row["year"]),
    )


def score_year(data, model, features):
    frame = data[features]
    valid = (~frame.isna().any(axis=1)).to_numpy()
    x = frame.fillna(0).to_numpy(float)
    return {
        "valid": valid,
        "buy": model.predict_proba(x)[:, 1],
        "sell": np.zeros(len(data), dtype=float),
    }


def config_dict(buy_threshold):
    return {
        "group": best.GROUP,
        "buy_threshold": buy_threshold,
        "sell_threshold": best.SELL_THRESHOLD,
        "entry_mode": best.ENTRY_MODE,
        "side_mode": best.SIDE_MODE,
        "session_mode": best.SESSION_MODE,
    }


def build_deal(frame, candle, side):
    order = "buy" if int(side) == 1 else "sell"
    entry = float(frame[OPEN][candle + 1])
    return {
        "start_index": candle + 1,
        "entry": round(entry, 5),
        "sl": round(entry - best.R1 if order == "buy" else entry + best.R1, 5),
        "tp": round(entry + best.R1 * best.RR if order == "buy" else entry - best.R1 * best.RR, 5),
        "type": "market",
        "order": order,
    }


def simulate_year_money(data, orders, max_running_deals, param_tinh_tien):
    frame = data.copy().reset_index(drop=True)
    frame[DATE] = frame[DATE].astype(str)
    mophong = MoPhongDeals(frame, best.EXPIRED, tinh_tien=param_tinh_tien)

    for candle in range(250, len(frame) - 2):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=max_running_deals)
        side = int(orders[candle])
        if side == 0:
            continue
        mophong.add_new_deal(deal=build_deal(frame, candle, side), min_deal_gap=best.MIN_DEAL_GAP)

    for candle in range(max(250, len(frame) - 2), len(frame)):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=max_running_deals)

    results = mophong.get_results()
    wins = sum(1 for deal in results if deal["res"] == "Win")
    loses = sum(1 for deal in results if deal["res"] == "Lose")
    nores = sum(1 for deal in results if deal["res"] == "No res")
    resolved = wins + loses
    fund = mophong.current_fund
    max_fund = mophong.max_fund
    add_fund = mophong.add_fund
    rut_fund = mophong.rut_fund
    return {
        "raw_deals": int(np.count_nonzero(orders)),
        "results": len(results),
        "resolved": resolved,
        "wins": wins,
        "loses": loses,
        "nores": nores,
        "winrate": round(wins * 100 / resolved, 2) if resolved else 0.0,
        "fund": round(float(fund), 2),
        "max_fund": round(float(max_fund), 2),
        "add_fund": round(float(add_fund), 2),
        "rut_fund": round(float(rut_fund), 2),
    }


def summarize(rows):
    df = pd.DataFrame(rows)
    wins = int(df["wins"].sum())
    resolved = int(df["resolved"].sum())
    return {
        "years": ",".join(str(int(year)) for year in sorted(df["year"].unique())),
        "n_years": int(df["year"].nunique()),
        "total_resolved": resolved,
        "total_wins": wins,
        "total_loses": int(df["loses"].sum()),
        "weighted_winrate": round(wins * 100 / resolved, 2) if resolved else 0.0,
        "min_winrate": round(float(df["winrate"].min()), 2),
        "max_winrate": round(float(df["winrate"].max()), 2),
        "sum_max_fund": round(float(df["max_fund"].sum()), 2),
        "best_year_max_fund": round(float(df["max_fund"].max()), 2),
        "min_max_fund": round(float(df["max_fund"].min()), 2),
        "sum_add_fund": round(float(df["add_fund"].sum()), 2),
        "max_add_fund": round(float(df["add_fund"].max()), 2),
        "sum_rut_fund": round(float(df["rut_fund"].sum()), 2),
        "pass_add_fund": bool((df["add_fund"] <= MAX_ADD_FUND_PER_YEAR).all()),
        "seconds": round(float(df["seconds"].sum()), 2),
    }


def run(years_to_run, limit_configs=None, force=False):
    years_cache = best.load_or_prepare_years(force_prepare=False)
    model, features = best.load_model_from_source_or_train(years_cache, retrain=False)

    data_by_year = {}
    score_by_year = {}
    orders_by_key = {}
    for year in years_to_run:
        data = best.load_year_data(years_cache, year)
        data_by_year[year] = data
        score_by_year[year] = score_year(data, model, features)

    old_rows = []
    done = set()
    if os.path.exists(OUT_DETAIL) and not force:
        old = pd.read_csv(OUT_DETAIL)
        old_rows = old.to_dict("records")
        done = {key_tuple(row) for row in old_rows}
        print(f"resume detail rows={len(old_rows)}")

    candidate_params = []
    for buy_threshold, max_running, start_fund, density, rut in itertools.product(
        BUY_THRESHOLDS, MAX_RUNNING_GRID, START_FUND_GRID, DENSITY_GRID, RUT_GRID
    ):
        candidate_params.append((buy_threshold, max_running, start_fund, density, rut))
    if limit_configs:
        candidate_params = candidate_params[: int(limit_configs)]

    detail_rows = old_rows
    started_all = time.time()
    for idx, (buy_threshold, max_running, start_fund, density, rut) in enumerate(candidate_params, start=1):
        param = {
            "start_fund": start_fund,
            "rut": rut,
            "density": density,
            "time_add_fund": [f" {i}:00" for i in range(17, 18, 1)],
            "commision": 16,
        }
        for year in years_to_run:
            row_key = (float(buy_threshold), int(max_running), float(start_fund), float(density), float(rut[0]), float(rut[1]), int(year))
            if row_key in done:
                continue
            started = time.time()
            order_key = (year, buy_threshold)
            if order_key not in orders_by_key:
                orders_by_key[order_key] = bb._build_orders(data_by_year[year], score_by_year[year], config_dict(buy_threshold))
            stats = simulate_year_money(data_by_year[year], orders_by_key[order_key], max_running, param)
            row = {
                "strategy": best.STRATEGY_NAME,
                "group": best.GROUP,
                "side_mode": best.SIDE_MODE,
                "entry_mode": best.ENTRY_MODE,
                "session_mode": best.SESSION_MODE,
                "buy_threshold": buy_threshold,
                "sell_threshold": best.SELL_THRESHOLD,
                "max_running_deals": max_running,
                "start_fund": start_fund,
                "density": density,
                "rut_level": rut[0],
                "rut_pct": rut[1],
                "year": year,
                **stats,
                "seconds": round(time.time() - started, 2),
            }
            detail_rows.append(row)
            done.add(row_key)
            pd.DataFrame(detail_rows).to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
            print(f"{idx}/{len(candidate_params)} {row}", flush=True)

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    group_cols = ["buy_threshold", "max_running_deals", "start_fund", "density", "rut_level", "rut_pct"]
    for keys, part in detail.groupby(group_cols, dropna=False):
        if set(part["year"].astype(int)) >= set(years_to_run):
            item = dict(zip(group_cols, keys))
            item.update(summarize(part[part["year"].astype(int).isin(years_to_run)]))
            summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["pass_add_fund", "sum_max_fund", "best_year_max_fund", "weighted_winrate"],
            ascending=[False, False, False, False],
        )
        summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
        print(summary.head(30).to_string(index=False))
    print(f"detail: {OUT_DETAIL}")
    print(f"summary: {OUT_SUMMARY}")
    print(f"elapsed={round(time.time() - started_all, 2)}s")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(parse_years(args.years), limit_configs=args.limit_configs, force=args.force)
