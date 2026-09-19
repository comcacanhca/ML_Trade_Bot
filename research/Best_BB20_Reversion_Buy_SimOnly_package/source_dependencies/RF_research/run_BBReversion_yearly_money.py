import os
import sys
import time
import argparse

import joblib
import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
GROUP_DIR = os.path.join(PROJECT_ROOT, "srateries", "RF_Group_indicators")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if GROUP_DIR not in sys.path:
    sys.path.insert(0, GROUP_DIR)

from data import CLOSE, DATE, HIGH, LOW, OPEN
from method.MoPhongDeals import MoPhongDeals
import research_bb_family_cycles_v2_mophong as bb


OUT_FILE = os.path.join(CURRENT_DIR, "YEARLY_MONEY_RF_BBReversion_From_BBFamilyV2_MoPhongDeals.csv")

CONFIG = {
    "group": "bb20_reversion_cycles",
    "buy_threshold": 0.52,
    "sell_threshold": 0.40,
    "entry_mode": "lower_reversion",
    "side_mode": "buy_only",
    "session_mode": "london_ny",
}

PARAM_TINH_TIEN = {
    "start_fund": 6 * 50,
    "rut": (20000, 0.05),
    "density": 6 / (6 * 50),
    "time_add_fund": [f" {i}:00" for i in range(17, 18, 1)],
    "commision": 16,
}


def _fixed_deal(frame, candle, side):
    order = "buy" if side == 1 else "sell"
    entry = float(frame[OPEN][candle + 1])
    return {
        "start_index": candle + 1,
        "entry": round(entry, 5),
        "sl": round(entry - bb.base.R1 if order == "buy" else entry + bb.base.R1, 5),
        "tp": round(entry + bb.base.R1 * bb.base.RR if order == "buy" else entry - bb.base.R1 * bb.base.RR, 5),
        "type": "market",
        "order": order,
    }


def simulate_year_money(data, orders):
    frame = data.copy().reset_index(drop=True)
    frame[DATE] = frame[DATE].astype(str)
    mophong = MoPhongDeals(frame, bb.base.EXPIRED, tinh_tien=PARAM_TINH_TIEN)
    for candle in range(250, len(frame) - 2):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=bb.base.MAX_RUNNING_DEALS)
        side = int(orders[candle])
        if side == 0:
            continue
        mophong.add_new_deal(deal=_fixed_deal(frame, candle, side), min_deal_gap=0)
    for candle in range(max(250, len(frame) - 2), len(frame)):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=bb.base.MAX_RUNNING_DEALS)

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
        "fund": round(float(mophong.current_fund), 2),
        "max_fund": round(float(mophong.max_fund), 2),
        "add_fund": round(float(mophong.add_fund), 2),
        "rut_fund": round(float(mophong.rut_fund), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default=",".join(str(year) for year in bb.base.ALL_YEARS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    requested_years = [int(value.strip()) for value in args.years.split(",") if value.strip()]

    years = joblib.load(bb.PREPARED_FILE, mmap_mode="r")
    scores = joblib.load(bb.SCORE_FILE, mmap_mode="r")
    models = joblib.load(bb.MODEL_FILE, mmap_mode="r")
    rows = []
    done_years = set()
    if os.path.exists(OUT_FILE) and not args.force:
        old = pd.read_csv(OUT_FILE)
        old = old[old["year"].astype(str) != "TOTAL"]
        rows = old.to_dict("records")
        done_years = set(old["year"].astype(int).tolist())

    for year in requested_years:
        if year in done_years:
            print(f"skip done {year}", flush=True)
            continue
        started = time.time()
        if year in years and year in scores[CONFIG["group"]]:
            data = years[year]["data"]
            score = scores[CONFIG["group"]][year]
        else:
            raw = bb.base._load_year_data(year)
            data = bb._prepare_features(raw)
            model_bundle = models[CONFIG["group"]]
            frame = data[model_bundle["features"]]
            valid = (~frame.isna().any(axis=1)).to_numpy()
            x = frame.fillna(0).to_numpy(float)
            score = {
                "valid": valid,
                "buy": model_bundle["buy"].predict_proba(x)[:, 1],
                "sell": model_bundle["sell"].predict_proba(x)[:, 1],
            }
        orders = bb._build_orders(data, score, CONFIG)
        stats = simulate_year_money(data, orders)
        row = {**CONFIG, "year": year, **stats, "seconds": round(time.time() - started, 2)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        print(row, flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows")
        return
    wins = int(df["wins"].sum())
    resolved = int(df["resolved"].sum())
    total = {
        **CONFIG,
        "year": "TOTAL",
        "raw_deals": int(df["raw_deals"].sum()),
        "results": int(df["results"].sum()),
        "resolved": resolved,
        "wins": wins,
        "loses": int(df["loses"].sum()),
        "nores": int(df["nores"].sum()),
        "winrate": round(wins * 100 / resolved, 2) if resolved else 0.0,
        "max_fund": round(float(df["max_fund"].max()), 2),
        "add_fund": round(float(df["add_fund"].sum()), 2),
        "rut_fund": round(float(df["rut_fund"].sum()), 2),
        "seconds": round(float(df["seconds"].sum()), 2),
    }
    out = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print("saved", OUT_FILE)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
