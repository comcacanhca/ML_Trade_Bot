from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from config import CFG, ensure_dirs
from data_io import load_years, parse_years
from features import build_features
from mophong_adapter import orders_from_scores, simulate_orders


def main(years: list[int], threshold: float | None = None, model_file: str | None = None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    model_path = model_file or str(CFG.model_dir / "rf_buy_winlose_bundle.joblib")
    bundle = joblib.load(model_path)
    cols = bundle["feature_columns"]
    threshold = float(bundle["selected_threshold"] if threshold is None else threshold)

    frame = build_features(load_years(years))
    frame["year"] = pd.to_datetime(frame["dates"]).dt.year.astype(int)
    valid = frame[cols].notna().all(axis=1).to_numpy()
    scores = np.full(len(frame), np.nan, dtype=float)
    x = bundle["scaler"].transform(frame.loc[valid, cols].to_numpy(float))
    scores[valid] = bundle["model"].predict_proba(x)[:, 1]

    rows = []
    for year in years:
        chunk = frame[frame["year"] == year].copy().reset_index(drop=True)
        score_chunk = scores[frame["year"].to_numpy() == year]
        orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(CFG.outputs_dir / f"strategy_deals_{year}.csv", index=False, encoding="utf-8-sig")

    out = pd.DataFrame(rows)
    out_file = CFG.outputs_dir / f"strategy_yearly_{min(years)}_{max(years)}.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    result = {"model_file": model_path, "threshold": threshold, "yearly": rows, "output_file": str(out_file)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2024-2026")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model-file", default=None)
    args = parser.parse_args()
    main(parse_years(args.years), threshold=args.threshold, model_file=args.model_file)
