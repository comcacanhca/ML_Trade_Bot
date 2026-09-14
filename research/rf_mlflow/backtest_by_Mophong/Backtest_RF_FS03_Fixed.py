from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
RF_MLFLOW_DIR = THIS_DIR.parent
PROJECT_ROOT = RF_MLFLOW_DIR.parents[2]

for path in [PROJECT_ROOT, RF_MLFLOW_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import CFG  # noqa: E402
from data_handler import CLOSE, DATE, HIGH, LOW, MAU_NEN, OPEN  # noqa: E402
from data_io import load_years, parse_years  # noqa: E402
from method.MoPhongDeals import MoPhongDeals  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logging.disable(logging.CRITICAL)


MODEL_FILE = RF_MLFLOW_DIR / "models" / "fs03_fixed_params_1788583848_bundle.joblib"
OUTPUT_DIR = THIS_DIR / "outputs"

# Default from reproduced research10_1788533519 fs03 threshold curve:
# threshold 0.57 => proxy 2024-2026 resolved=6816, winrate=58.26%.
DEFAULT_THRESHOLD = 0.565

RR = 1.0
R1 = 6.0
EXPIRATED = 5
MAX_RUNNING_DEALS = 30
MIN_DEAL_GAP = 0
START_CANDLE = 300
SESSION_FILTER = "london_ny"

PARAM_TINH_TIEN = {
    "start_fund": 6 * 30,
    "rut": (2000, 0.05),
    "density": 6 / (6 * 50),
    "time_add_fund": [f" {i}:00" for i in range(17, 18, 1)],
    "commision": 16,
}


def _load_model(model_file: Path = MODEL_FILE) -> dict:
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model bundle: {model_file}")
    bundle = joblib.load(model_file)
    required = {"model", "scaler", "feature_columns"}
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"Invalid model bundle, missing keys: {sorted(missing)}")
    return bundle


def _build_buy_deal(data: pd.DataFrame, candle: int) -> dict:
    entry = float(data[OPEN].iloc[candle + 1])
    return {
        "start_index": candle + 1,
        "entry": round(entry, 5),
        "sl": round(entry - R1, 5),
        "tp": round(entry + R1 * RR, 5),
        "type": "market",
        "order": "buy",
    }


def _score_frame(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    features = list(bundle["feature_columns"])
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise KeyError(f"Prepared data_handler missing model features: {missing[:20]}")

    valid = frame[features].notna().all(axis=1).to_numpy()
    scores = np.full(len(frame), np.nan, dtype=float)
    x = bundle["scaler"].transform(frame.loc[valid, features].to_numpy(float))
    scores[valid] = bundle["model"].predict_proba(x)[:, 1]
    return scores


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_fs03_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build only the 16 columns used by the fixed FS03 model.

    The generic build_features() path creates hundreds of BB/overlay columns.
    For this model that is unnecessary and can allocate >800MB in one pandas
    consolidation step on a single year of M1 data_handler.
    """
    out = data[[DATE, OPEN, LOW, HIGH, CLOSE]].copy()
    point = 100.0

    ret_1 = (out[CLOSE] - out[OPEN]) * point
    returns = out[CLOSE].diff() * point
    for lag in [5, 8, 13, 21]:
        out[f"close_diff_lag_{lag}"] = returns.shift(lag).astype("float32")
        out[f"body_lag_{lag}"] = ret_1.shift(lag).astype("float32")

    dt = pd.to_datetime(out[DATE])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440).astype("float32")
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 5).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 5).astype("float32")
    out["session_london_ny"] = ((dt.dt.hour >= 7) & (dt.dt.hour <= 21)).astype("int8")
    out["session_ny"] = ((dt.dt.hour >= 12) & (dt.dt.hour <= 21)).astype("int8")

    mid20 = out[CLOSE].rolling(20, min_periods=20).mean()
    std20 = out[CLOSE].rolling(20, min_periods=20).std()
    lower20 = mid20 - 2 * std20
    bb20_pos = (out[CLOSE] - lower20) / ((4 * std20).replace(0, np.nan))
    rsi_14 = _rsi(out[CLOSE], 14)
    ema8 = out[CLOSE].ewm(span=8, min_periods=8, adjust=False).mean()
    ema55 = out[CLOSE].ewm(span=55, min_periods=55, adjust=False).mean()
    ema_dist_8 = (out[CLOSE] - ema8) * point
    ema_dist_55 = (out[CLOSE] - ema55) * point
    ret_sum_10 = returns.rolling(10, min_periods=10).sum()
    z_close_20 = (out[CLOSE] - mid20) / std20.replace(0, np.nan)

    out["sig_bb_reversion_buy"] = ((bb20_pos < 0.18) & (rsi_14 < 45)).astype("int8")
    out["sig_pullback_trend_buy"] = ((ema_dist_55 > 0) & (ema_dist_8 < 0) & (ret_1 > 0)).astype("int8")
    out["sig_momentum_buy"] = ((ret_sum_10 > 0) & (z_close_20 > -0.5) & (z_close_20 < 1.5)).astype("int8")
    out["buy_signal"] = (
        (out["sig_bb_reversion_buy"] == 1)
        | (out["sig_pullback_trend_buy"] == 1)
        | ((out["sig_momentum_buy"] == 1) & (out["session_london_ny"] == 1))
    ).astype("int8")

    return out


def _build_orders(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> np.ndarray:
    orders = np.zeros(len(frame), dtype=np.int8)
    valid = np.isfinite(scores)
    if "buy_signal" in frame.columns:
        valid &= frame["buy_signal"].to_numpy(np.int8) == 1
    if SESSION_FILTER == "london_ny" and "session_london_ny" in frame.columns:
        valid &= frame["session_london_ny"].to_numpy(np.int8) == 1
    elif SESSION_FILTER == "ny" and "session_ny" in frame.columns:
        valid &= frame["session_ny"].to_numpy(np.int8) == 1
    orders[valid & (scores >= threshold)] = 1
    return orders


def _ensure_mophong_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[[DATE, OPEN, LOW, HIGH, CLOSE]].copy().reset_index(drop=True)
    frame[DATE] = frame[DATE].astype(str)
    frame[MAU_NEN] = frame[CLOSE] - frame[OPEN]
    return frame


def backtest_one_year(
    data: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    year: int,
    tinh_tien: bool = True,
    export_fund_excel: bool = False,
) -> tuple[dict, list[dict]]:
    frame = _ensure_mophong_frame(data)
    orders = _build_orders(data, scores, threshold)
    mophong = MoPhongDeals(frame, EXPIRATED, tinh_tien=PARAM_TINH_TIEN if tinh_tien else None)

    started = time.time()
    for candle in range(START_CANDLE, len(frame) - 2):
        if candle % 20000 == 0:
            print(f"{year}: processing {round(candle * 100 / len(frame), 2)}%", end="\r")

        if len(mophong.get_deals()) > 0:
            mophong.check_deals(candle, max_running_deal=MAX_RUNNING_DEALS)

        if int(orders[candle]) == 1:
            mophong.add_new_deal(_build_buy_deal(frame, candle), min_deal_gap=MIN_DEAL_GAP)

    results = mophong.get_results()
    wins = sum(item["res"] == "Win" for item in results)
    losses = sum(item["res"] == "Lose" for item in results)
    no_res = sum(item["res"] == "No res" for item in results)
    resolved = wins + losses
    summary = {
        "year": year,
        "threshold": threshold,
        "raw_orders": int(np.count_nonzero(orders == 1)),
        "deals": len(results),
        "resolved": resolved,
        "wins": wins,
        "losses": losses,
        "no_res": no_res,
        "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
        "seconds": round(time.time() - started, 2),
    }
    if tinh_tien:
        try:
            fund, max_fund, add_fund, rut_fund = mophong.get_fund()
        except Exception:
            fund, max_fund, add_fund, rut_fund = (
                mophong.current_fund,
                mophong.max_fund,
                mophong.add_fund,
                mophong.rut_fund,
            )
        summary.update(
            {
                "fund": round(float(fund), 4),
                "max_fund": round(float(max_fund), 4),
                "add_fund": round(float(add_fund), 4),
                "rut_fund": round(float(rut_fund), 4),
            }
        )

    if tinh_tien:
        print(f"{year}: WR={summary['winrate']} deals={summary['deals']} resolved={resolved}, {max_fund}, add: {add_fund}, {rut_fund}")
    else:
        print(f"{year}: WR={summary['winrate']} deals={summary['deals']} resolved={resolved}")
    return summary, results


def add_total_row(yearly: pd.DataFrame) -> pd.DataFrame:
    totals = {
        "year": "TOTAL",
        "threshold": yearly["threshold"].iloc[0] if len(yearly) else np.nan,
        "raw_orders": int(yearly["raw_orders"].sum()),
        "deals": int(yearly["deals"].sum()),
        "resolved": int(yearly["resolved"].sum()),
        "wins": int(yearly["wins"].sum()),
        "losses": int(yearly["losses"].sum()),
        "no_res": int(yearly["no_res"].sum()),
        "seconds": round(float(yearly["seconds"].sum()), 2),
    }
    totals["winrate"] = round(totals["wins"] * 100 / totals["resolved"], 4) if totals["resolved"] else 0.0
    for col in ["fund", "max_fund", "add_fund", "rut_fund"]:
        if col in yearly.columns:
            totals[col] = round(float(yearly[col].sum()), 4) if col != "max_fund" else round(float(yearly[col].max()), 4)
    return pd.concat([yearly, pd.DataFrame([totals])], ignore_index=True)


def run_backtest(
    years: list[int],
    threshold: float,
    model_file: Path,
    tinh_tien: bool = True,
    export_fund_excel: bool = False,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Starting backtest...')
    bundle = _load_model(model_file)
    rows = []
    for year in years:
        raw = load_years([year])
        prepared = build_fs03_features(raw)
        prepared["year"] = pd.to_datetime(prepared[DATE]).dt.year.astype(int)
        scores = _score_frame(prepared, bundle)

        mask = prepared["year"].to_numpy() == year
        chunk = prepared.loc[mask].copy().reset_index(drop=True)
        year_scores = scores[mask]
        summary, results = backtest_one_year(
            chunk,
            year_scores,
            threshold,
            year,
            tinh_tien=tinh_tien,
            export_fund_excel=export_fund_excel,
        )
        rows.append(summary)
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / f"deals_fs03_fixed_t{threshold:.2f}_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        del raw, prepared, scores, mask, chunk, year_scores, results
        gc.collect()

    yearly = add_total_row(pd.DataFrame(rows))
    yearly_path = OUTPUT_DIR / f"yearly_fs03_fixed_t{threshold:.2f}_{min(years)}_{max(years)}.csv"
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    result = {
        "model_file": str(model_file),
        "threshold": threshold,
        "years": years,
        "yearly_path": str(yearly_path),
        "summary": yearly.to_dict(orient="records"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2025-2026", help="Example: 2024-2026 or 2024,2025")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model-file", default=str(MODEL_FILE))
    parser.add_argument("--no-tinh-tien", action="store_true")
    parser.add_argument("--export-fund-excel", action="store_true")
    args = parser.parse_args()
    run_backtest(
        years=parse_years(args.years),
        threshold=float(args.threshold),
        model_file=Path(args.model_file),
        tinh_tien=not args.no_tinh_tien,
        export_fund_excel=bool(args.export_fund_excel),
    )


if __name__ == "__main__":
    main()
