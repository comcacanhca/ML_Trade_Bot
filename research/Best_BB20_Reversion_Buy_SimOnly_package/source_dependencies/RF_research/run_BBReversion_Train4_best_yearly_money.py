import argparse
import json
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
GROUP_DIR = os.path.join(PROJECT_ROOT, "srateries", "RF_Group_indicators")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if GROUP_DIR not in sys.path:
    sys.path.insert(0, GROUP_DIR)

from data import DATE, OPEN
from method.MoPhongDeals import MoPhongDeals
from plot_chart.draw_cha import draw_MA
import research_bb_family_cycles_v2_mophong as bb
from RF_BBReversion_Train4Validate4_MoPhongDeals import COMPACT_FILE, _dataset, _score_model


warnings.filterwarnings("ignore", category=FutureWarning)

STRATEGY_NAME = "RF_BBReversion_Train4_Best"
SIM_NAME = f"{STRATEGY_NAME}_MoPhongDeals"

RR = 1
R1 = 6.0
MAX_RUNNING_DEALS = 10
MIN_DEAL_GAP = 0
EXPERIED = 5

# Main strategy knobs - edit these first.
GROUP = "bb20_reversion_cycles"
SIDE_MODE = "buy_only"
ENTRY_MODE = "lower_reversion"
SESSION_MODE = "london_ny"
SCORE_THRESHOLD = 0.53
TRAIN_YEARS = (2018, 2021, 2022, 2023)

# Keep False to match the old SimOnly style. Set True if you want to close/check
# remaining running deals at the last candles.
CHECK_OPEN_AT_END = False

OUT_FILE = os.path.join(CURRENT_DIR, f"YEARLY_MONEY_{SIM_NAME}.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", SIM_NAME)
MODEL_FILE = os.path.join(MODEL_DIR, f"{SIM_NAME}_model.joblib")
MODEL_META_FILE = os.path.join(MODEL_DIR, f"{SIM_NAME}_meta.json")

PARAM_TINH_TIEN = {
    "start_fund": 6 * 30,
    "rut": (20000, 0.05),
    "density": 6 / (6 * 30),
    "time_add_fund": [f" {i}:00" for i in range(17, 18, 1)],
    "commision": 16,
}


def parse_years(value):
    years = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x.strip()) for x in part.split("-", 1)]
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


def _rf_model(seed):
    return RandomForestClassifier(
        n_estimators=180,
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


def load_compact_data():
    if not os.path.exists(COMPACT_FILE):
        raise RuntimeError(
            f"Missing compact feature cache: {COMPACT_FILE}. "
            "Run RF_BBReversion_Train4Validate4_MoPhongDeals.py first."
        )
    return joblib.load(COMPACT_FILE, mmap_mode="r")


def _model_meta():
    return {
        "strategy_name": STRATEGY_NAME,
        "group": GROUP,
        "side_mode": SIDE_MODE,
        "entry_mode": ENTRY_MODE,
        "session_mode": SESSION_MODE,
        "score_threshold": SCORE_THRESHOLD,
        "train_years": list(TRAIN_YEARS),
        "rr": RR,
        "r1": R1,
        "expired": EXPERIED,
        "max_running_deals": MAX_RUNNING_DEALS,
    }


def _same_model_meta(meta):
    expected = _model_meta()
    keys = ["group", "side_mode", "train_years"]
    return all(meta.get(key) == expected.get(key) for key in keys)


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    with open(MODEL_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(_model_meta(), fh, indent=2, ensure_ascii=False)
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved model meta: {MODEL_META_FILE}")


def load_model(compact, retrain=False):
    if os.path.exists(MODEL_FILE) and os.path.exists(MODEL_META_FILE) and not retrain:
        with open(MODEL_META_FILE, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if _same_model_meta(meta):
            print(f"Loaded model: {MODEL_FILE}")
            return joblib.load(MODEL_FILE)
        print("Existing model meta does not match current GROUP/SIDE_MODE/TRAIN_YEARS. Retraining.")

    x, y = _dataset(compact, GROUP, TRAIN_YEARS, "buy" if SIDE_MODE == "buy_only" else "sell")
    model = _rf_model(abs(hash((GROUP, SIDE_MODE, TRAIN_YEARS))) % 100000)
    model.fit(x, y)
    print(f"Trained runtime model: group={GROUP}, side={SIDE_MODE}, train_years={TRAIN_YEARS}")
    save_model(model)
    return model


def prepare_year_data(compact, year, model):
    if year not in compact:
        raise RuntimeError(
            f"Year {year} is not in compact cache. "
            "Rebuild compact cache before simulating this year."
        )

    data = compact[year]["data"].copy().reset_index(drop=True)
    group_data = compact[year]["groups"][GROUP]
    score = _score_model(model, compact, GROUP, (year,))[year]

    data["_bb_valid"] = group_data["valid"]
    data["_bb_signal"] = group_data["signal"]
    data["_bb_score"] = score
    data["_bb_entry_ok"] = bb._entry_mask(data, {"group": GROUP, "entry_mode": ENTRY_MODE})
    data["_bb_session_ok"] = bb.base._session_mask(data, SESSION_MODE)
    return data


def should_open_order(data, candle):
    if not bool(data["_bb_valid"][candle]):
        return False
    if not bool(data["_bb_entry_ok"][candle]):
        return False
    if not bool(data["_bb_session_ok"][candle]):
        return False
    if float(data["_bb_score"][candle]) < SCORE_THRESHOLD:
        return False

    signal = int(data["_bb_signal"][candle])
    if SIDE_MODE in ["buy_only", "both"] and signal == 1:
        return "buy"
    if SIDE_MODE in ["sell_only", "both"] and signal == -1:
        return "sell"
    return False


def build_deal(data, candle, order):
    entry = float(data[OPEN][candle + 1])
    return {
        "start_index": candle + 1,
        "entry": round(entry, 5),
        "sl": round(entry - R1 if order == "buy" else entry + R1, 5),
        "tp": round(entry + R1 * RR if order == "buy" else entry - R1 * RR, 5),
        "type": "market",
        "order": order,
    }


def simulate_year(data, interactive=False):
    print("----------------------------------------------------------------------------------------")
    print(f"From date {data[DATE][0]} to {data[DATE][len(data) - 1]}")

    frame = data.copy().reset_index(drop=True)
    frame[DATE] = frame[DATE].astype(str)
    mophong = MoPhongDeals(frame, EXPERIED, tinh_tien=PARAM_TINH_TIEN)
    raw_deals = 0

    for candle in range(250, len(frame) - 2):
        if candle % 1000 == 0:
            print(f"Processing candle {round(candle * 100 / len(frame), 2)}%", end="\r")

        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=MAX_RUNNING_DEALS)

        order = should_open_order(frame, candle)
        if order is False:
            continue

        mophong.add_new_deal(deal=build_deal(frame, candle, order), min_deal_gap=MIN_DEAL_GAP)
        raw_deals += 1

    # if CHECK_OPEN_AT_END:
    #     for candle in range(max(250, len(frame) - 2), len(frame)):
    #         if len(mophong.get_deals()) > 0:
    #             mophong.check_deals(candle, max_running_deal=MAX_RUNNING_DEALS)

    results = mophong.get_results()
    wins = sum(1 for deal in results if deal["res"] == "Win")
    loses = sum(1 for deal in results if deal["res"] == "Lose")
    nores = sum(1 for deal in results if deal["res"] == "No res")
    resolved = wins + loses

    try:
        fund, max_fund, add_fund, rut_fund = mophong.get_fund()
    except Exception:
        fund, max_fund, add_fund, rut_fund = 0, 0, 0, 0

    winrate = round(wins * 100 / resolved, 2) if resolved else 0
    print(f"Found {PARAM_TINH_TIEN['start_fund']} become {fund} max fund: {max_fund} add fund {add_fund} rut {rut_fund}")
    print(f"Win rate: {winrate} = {wins}/{resolved} | No res: {nores} | Raw deals: {raw_deals}")

    if interactive:
        draw_MA(
            frame,
            deals=results,
            results=results,
            BB=None,
            range_list=[],
            price_indicator_lines=[],
            indicator_lines=[],
            mfivas=(),
            interactive=True,
        )

    return {
        "raw_deals": raw_deals,
        "results": len(results),
        "resolved": resolved,
        "wins": wins,
        "loses": loses,
        "nores": nores,
        "winrate": winrate,
        "fund": round(float(fund), 2),
        "max_fund": round(float(max_fund), 2),
        "add_fund": round(float(add_fund), 2),
        "rut_fund": round(float(rut_fund), 2),
    }


def run_backtest(years, force=False, interactive=False, retrain=False):
    compact = load_compact_data()
    model = load_model(compact, retrain=retrain)

    yearly = []
    done = set()
    # if os.path.exists(OUT_FILE) and not force:
    #     old = pd.read_csv(OUT_FILE)
    #     old = old[old["year"].astype(str) != "TOTAL"]
    #     yearly = old.to_dict("records")
    #     done = set(old["year"].astype(int).tolist())

    for year in years:
        if year in done:
            print(f"Skip done {year}")
            continue

        round_start = time.time()
        data = prepare_year_data(compact, year, model)
        result = simulate_year(data, interactive=interactive)
        result.update(
            {
                "group": GROUP,
                "side_mode": SIDE_MODE,
                "entry_mode": ENTRY_MODE,
                "session_mode": SESSION_MODE,
                "threshold": SCORE_THRESHOLD,
                "train_years": ",".join(map(str, TRAIN_YEARS)),
                "year": year,
                "seconds": round(time.time() - round_start, 2),
            }
        )
        yearly.append(result)
        pd.DataFrame(yearly).to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Year {year}: {result} | Time to run: {result['seconds']} seconds")

    summary = pd.DataFrame(yearly)
    if summary.empty:
        return summary

    rows_no_total = summary[summary["year"].astype(str) != "TOTAL"].copy()
    wins = int(rows_no_total["wins"].sum())
    resolved = int(rows_no_total["resolved"].sum())
    total = {
        "group": GROUP,
        "side_mode": SIDE_MODE,
        "entry_mode": ENTRY_MODE,
        "session_mode": SESSION_MODE,
        "threshold": SCORE_THRESHOLD,
        "train_years": ",".join(map(str, TRAIN_YEARS)),
        "year": "TOTAL",
        "raw_deals": int(rows_no_total["raw_deals"].sum()),
        "results": int(rows_no_total["results"].sum()),
        "resolved": resolved,
        "wins": wins,
        "loses": int(rows_no_total["loses"].sum()),
        "nores": int(rows_no_total["nores"].sum()),
        "winrate": round(wins * 100 / resolved, 2) if resolved else 0,
        "fund": 0,
        "max_fund": round(float(rows_no_total["max_fund"].max()), 2),
        "add_fund": round(float(rows_no_total["add_fund"].sum()), 2),
        "rut_fund": round(float(rows_no_total["rut_fund"].sum()), 2),
        "seconds": round(float(rows_no_total["seconds"].sum()), 2),
    }
    output = pd.concat([rows_no_total, pd.DataFrame([total])], ignore_index=True)
    output.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(output[
        ["year", "resolved", "wins", "loses", "nores", "winrate", "max_fund", "add_fund", "rut_fund"]
    ])
    print(f"Saved: {OUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2024,2025,2026")#2016,2017,2018,2019,2020,2021,2022,2023,
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="Retrain and overwrite saved model before simulation")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    start = time.time()
    run_backtest(parse_years(args.years), force=args.force, interactive=args.interactive, retrain=args.retrain)
    print(f"End after: {round(time.time() - start, 2)} seconds")
