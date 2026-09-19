from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG, PROJECT_ROOT
from data_io import CLOSE, DATE, HIGH, LOW, OPEN

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_handler import MAU_NEN  # noqa: E402
from method.MoPhongDeals import MoPhongDeals  # noqa: E402


PARAM_TINH_TIEN = {
    "start_fund": 6 * 50,
    "rut": (2000, 0.05),
    "density": 6 / (6 * 50),
    "time_add_fund": [f" {i}:00" for i in range(17, 18, 1)],
    "commision": CFG.trade.commission,
}


def build_buy_deal(data: pd.DataFrame, candle: int) -> dict:
    entry = float(data[OPEN].iloc[candle + 1])
    return {
        "start_index": candle + 1,
        "entry": round(entry, 5),
        "sl": round(entry - CFG.trade.r1, 5),
        "tp": round(entry + CFG.trade.r1 * CFG.trade.rr, 5),
        "type": "market",
        "order": "buy",
    }


def orders_from_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    require_signal: bool = True,
    session_filter: str = "london_ny",
) -> np.ndarray:
    orders = np.zeros(len(frame), dtype=np.int8)
    valid = np.isfinite(scores)
    if require_signal and "buy_signal" in frame.columns:
        valid &= frame["buy_signal"].to_numpy(np.int8) == 1
    if session_filter == "london_ny" and "session_london_ny" in frame.columns:
        valid &= frame["session_london_ny"].to_numpy(np.int8) == 1
    elif session_filter == "ny" and "session_ny" in frame.columns:
        valid &= frame["session_ny"].to_numpy(np.int8) == 1
    orders[valid & (scores >= threshold)] = 1
    return orders


def simulate_orders(data: pd.DataFrame, orders: np.ndarray, tinh_tien: bool = True) -> tuple[dict, list[dict]]:
    frame = data[[DATE, OPEN, LOW, HIGH, CLOSE]].copy().reset_index(drop=True)
    frame[DATE] = frame[DATE].astype(str)
    frame[MAU_NEN] = frame[CLOSE] - frame[OPEN]
    mophong = MoPhongDeals(frame, CFG.trade.expired, tinh_tien=PARAM_TINH_TIEN if tinh_tien else None)
    if tinh_tien and hasattr(mophong, "RES_TINH_TIEN"):
        float_cols = ["start", "Profit", "Loss", "Fund", "Add", "Rut", "Com", "Lot", "winrate_range"]
        for col in float_cols:
            if col in mophong.RES_TINH_TIEN.columns:
                mophong.RES_TINH_TIEN[col] = mophong.RES_TINH_TIEN[col].astype("float64")
        if "Date_fill" in mophong.RES_TINH_TIEN.columns:
            mophong.RES_TINH_TIEN["Date_fill"] = mophong.RES_TINH_TIEN["Date_fill"].astype("object")

    for candle in range(CFG.trade.start_candle, len(frame) - 2):
        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=CFG.trade.max_running_deals)
        if int(orders[candle]) == 1:
            mophong.add_new_deal(build_buy_deal(frame, candle), min_deal_gap=CFG.trade.min_deal_gap)

    results = mophong.get_results()
    wins = sum(item["res"] == "Win" for item in results)
    losses = sum(item["res"] == "Lose" for item in results)
    no_res = sum(item["res"] == "No res" for item in results)
    resolved = wins + losses
    summary = {
        "deals": len(results),
        "resolved": resolved,
        "wins": wins,
        "losses": losses,
        "no_res": no_res,
        "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
    }
    if tinh_tien:
        try:
            fund, max_fund, add_fund, rut_fund = mophong.get_fund()
        except Exception:
            fund, max_fund, add_fund, rut_fund = mophong.current_fund, mophong.max_fund, mophong.add_fund, mophong.rut_fund
        summary.update({"fund": float(fund), "max_fund": float(max_fund), "add_fund": float(add_fund), "rut_fund": float(rut_fund)})
    return summary, results
