import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import run_BBReversion_Train4_best_yearly_money as base


STRATEGY_NAME = "RF_BBReversion_Train4_Best_Dedup"
SIM_NAME = f"{STRATEGY_NAME}_MoPhongDeals"

OUT_FILE = os.path.join(base.CURRENT_DIR, f"YEARLY_MONEY_{SIM_NAME}.csv")
MODEL_DIR = os.path.join(base.PROJECT_ROOT, "models", SIM_NAME)
MODEL_FILE = os.path.join(MODEL_DIR, f"{SIM_NAME}_model.joblib")
MODEL_META_FILE = os.path.join(MODEL_DIR, f"{SIM_NAME}_meta.json")


def _expanded_feature_names(compact, year=None):
    if year is None:
        year = next(iter(compact))
    group_data = compact[year]["groups"][base.GROUP]
    logical_features = list(group_data["features"])
    return list(compact[year]["data"][logical_features].columns)


def _dedup_feature_index(compact):
    expanded = _expanded_feature_names(compact)
    seen = set()
    keep_idx = []
    keep_names = []
    dropped = []
    for i, name in enumerate(expanded):
        if name in seen:
            dropped.append((i, name))
            continue
        seen.add(name)
        keep_idx.append(i)
        keep_names.append(name)
    return np.array(keep_idx, dtype=np.int64), keep_names, dropped


def _dataset_dedup(compact, years, side, keep_idx):
    label_key = "buy_label" if side == "buy" else "sell_label"
    direction = 1 if side == "buy" else -1
    xs, ys = [], []
    for year in years:
        item = compact[year]
        group_data = item["groups"][base.GROUP]
        labels = item[label_key]
        idx = np.where((labels != 0) & group_data["valid"] & (group_data["signal"] == direction))[0]
        if len(idx) > 90000:
            rng = np.random.default_rng(year + (13 if side == "buy" else 113))
            idx = rng.choice(idx, 90000, replace=False)
        xs.append(group_data["x"][idx][:, keep_idx])
        ys.append((labels[idx] == 1).astype(np.int8))
    return np.vstack(xs), np.concatenate(ys)


def _model_meta(keep_names, dropped):
    meta = base._model_meta()
    meta.update(
        {
            "strategy_name": STRATEGY_NAME,
            "deduplicate_feature_names": True,
            "n_features": len(keep_names),
            "features": keep_names,
            "dropped_duplicate_columns": [{"index": int(i), "feature": name} for i, name in dropped],
        }
    )
    return meta


def _same_model_meta(meta, keep_names):
    keys = ["group", "side_mode", "train_years", "deduplicate_feature_names", "n_features"]
    expected = _model_meta(keep_names, [])
    return all(meta.get(key) == expected.get(key) for key in keys)


def save_model(model, keep_names, dropped):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    with open(MODEL_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(_model_meta(keep_names, dropped), fh, indent=2, ensure_ascii=False)
    print(f"Saved dedup model: {MODEL_FILE}")
    print(f"Saved dedup model meta: {MODEL_META_FILE}")


def load_model(compact, retrain=False):
    keep_idx, keep_names, dropped = _dedup_feature_index(compact)
    if os.path.exists(MODEL_FILE) and os.path.exists(MODEL_META_FILE) and not retrain:
        with open(MODEL_META_FILE, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if _same_model_meta(meta, keep_names):
            print(f"Loaded dedup model: {MODEL_FILE}")
            return joblib.load(MODEL_FILE), keep_idx
        print("Existing dedup model meta does not match current config. Retraining.")

    side = "buy" if base.SIDE_MODE == "buy_only" else "sell"
    x, y = _dataset_dedup(compact, base.TRAIN_YEARS, side, keep_idx)
    model = base._rf_model(abs(hash((base.GROUP, base.SIDE_MODE, base.TRAIN_YEARS, "dedup"))) % 100000)
    model.fit(x, y)
    print(
        f"Trained dedup model: group={base.GROUP}, side={base.SIDE_MODE}, "
        f"train_years={base.TRAIN_YEARS}, features={x.shape[1]}, dropped={dropped}"
    )
    save_model(model, keep_names, dropped)
    return model, keep_idx


def prepare_year_data(compact, year, model, keep_idx):
    if year not in compact:
        raise RuntimeError(f"Year {year} is not in compact cache.")

    data = compact[year]["data"].copy().reset_index(drop=True)
    group_data = compact[year]["groups"][base.GROUP]
    score = model.predict_proba(group_data["x"][:, keep_idx])[:, 1]

    data["_bb_valid"] = group_data["valid"]
    data["_bb_signal"] = group_data["signal"]
    data["_bb_score"] = score
    data["_bb_entry_ok"] = base.bb._entry_mask(data, {"group": base.GROUP, "entry_mode": base.ENTRY_MODE})
    data["_bb_session_ok"] = base.bb.base._session_mask(data, base.SESSION_MODE)
    return data


def run_backtest(years, force=False, interactive=False, retrain=False):
    compact = base.load_compact_data()
    model, keep_idx = load_model(compact, retrain=retrain)

    yearly = []
    done = set()
    if os.path.exists(OUT_FILE) and not force:
        old = pd.read_csv(OUT_FILE)
        old = old[old["year"].astype(str) != "TOTAL"]
        yearly = old.to_dict("records")
        done = set(old["year"].astype(int).tolist())

    for year in years:
        if year in done:
            print(f"Skip done {year}")
            continue

        round_start = time.time()
        data = prepare_year_data(compact, year, model, keep_idx)
        result = base.simulate_year(data, interactive=interactive)
        result.update(
            {
                "group": base.GROUP,
                "side_mode": base.SIDE_MODE,
                "entry_mode": base.ENTRY_MODE,
                "session_mode": base.SESSION_MODE,
                "threshold": base.SCORE_THRESHOLD,
                "train_years": ",".join(map(str, base.TRAIN_YEARS)),
                "deduplicate_feature_names": True,
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
        "group": base.GROUP,
        "side_mode": base.SIDE_MODE,
        "entry_mode": base.ENTRY_MODE,
        "session_mode": base.SESSION_MODE,
        "threshold": base.SCORE_THRESHOLD,
        "train_years": ",".join(map(str, base.TRAIN_YEARS)),
        "deduplicate_feature_names": True,
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
    print(output[["year", "resolved", "wins", "loses", "nores", "winrate", "max_fund", "add_fund", "rut_fund"]])
    print(f"Saved: {OUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    start = time.time()
    run_backtest(base.parse_years(args.years), force=args.force, interactive=args.interactive, retrain=args.retrain)
    print(f"End after: {round(time.time() - start, 2)} seconds")
