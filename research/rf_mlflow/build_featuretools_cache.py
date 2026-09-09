from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import featuretools as ft
import numpy as np
import pandas as pd

from config import CFG, ensure_dirs
from research_random50_initial_features import META_COLS, _year_path, build_cache
from research_rolling_zscore_50_h4h1 import DEFAULT_CACHE_FAMILY, DROP_DEFAULT, PASSTHROUGH_FEATURES


FEATURETOOLS_CACHE_FAMILY = "featuretools_candidates"
DEFAULT_BASE_FEATURES = [
    "ret_1",
    "hl_range",
    "upper_wick",
    "lower_wick",
    "body_abs",
    "close_pos_range",
    "atr14",
    "rsi_14",
    "rsi_50",
    "rsi_gap_14_50",
    "ret_sum_5",
    "ret_sum_10",
    "ret_sum_20",
    "volatility_5",
    "volatility_20",
    "volatility_100",
    "m15_ret",
    "m15_ret_sum_3",
    "m15_ret_sum_6",
    "m15_volatility_3",
    "m15_volatility_6",
    "m15_volatility_12",
    "h1_ret",
    "h1_ret_sum_3",
    "h1_ret_sum_6",
    "h1_volatility_3",
    "h1_volatility_6",
    "h1_volatility_12",
    "h1_ema_dist_55",
    "h1_rsi_14",
    "h4_ret",
    "h4_ret_sum_3",
    "h4_ret_sum_6",
    "h4_ret_sum_12",
    "h4_volatility_3",
    "h4_volatility_6",
    "h4_volatility_12",
    "h4_ema_dist_8",
    "h4_ema_dist_55",
    "h4_rsi_14",
]
DEFAULT_PRIMITIVES = [
    "diff",
    "percent_change",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
]
PRESERVE_FEATURES = [
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "session_london_ny",
    "session_ny",
    "session_asia",
    "sig_bb_reversion_buy",
    "sig_pullback_trend_buy",
    "sig_momentum_buy",
    "buy_signal",
]


def _cache_dir(cache_family: str) -> Path:
    return CFG.cache_dir / cache_family


def _out_year_path(cache_family: str, year: int) -> Path:
    return _cache_dir(cache_family) / f"candidates_{year}.parquet"


def _sanitize_feature_name(name: str) -> str:
    cleaned = name.lower()
    cleaned = cleaned.replace("(", "_").replace(")", "")
    cleaned = cleaned.replace(",", "_").replace(" ", "")
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"ft_{cleaned}"
    return f"ft_{cleaned}" if not cleaned.startswith("ft_") else cleaned


def _load_base_features(path: str | None, all_cols: list[str]) -> list[str]:
    available = set(all_cols) - set(META_COLS) - set(PRESERVE_FEATURES) - DROP_DEFAULT
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("features", values.get("selected_features", []))
        base = [str(v) for v in values]
    else:
        base = DEFAULT_BASE_FEATURES
    return [c for c in dict.fromkeys(base) if c in available]


def _build_year(year: int, base_features: list[str], primitives: list[str], max_rows: int | None, cache_family: str) -> list[str]:
    meta_cols = ["dates", "candle_index", "entry_index", "entry_price", "year", "label"]
    preserve = [c for c in PRESERVE_FEATURES if c not in meta_cols]
    read_cols = list(dict.fromkeys(meta_cols + preserve + base_features))
    frame = pd.read_parquet(_year_path(year), columns=read_cols)
    if max_rows and len(frame) > max_rows:
        frame = frame.iloc[:max_rows].reset_index(drop=True)
    original_len = len(frame)

    ft_input = frame[["dates", *base_features]].copy()
    ft_input["ft_index"] = np.arange(original_len, dtype=np.int64)
    es = ft.EntitySet(id=f"xau_{year}")
    es = es.add_dataframe(dataframe_name="candles", dataframe=ft_input, index="ft_index", time_index="dates")
    matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="candles",
        trans_primitives=primitives,
        max_depth=1,
        features_only=False,
    )
    matrix = matrix.reset_index(drop=True)

    generated = matrix.drop(columns=[c for c in ["dates", *base_features] if c in matrix.columns], errors="ignore")
    rename_map = {c: _sanitize_feature_name(c) for c in generated.columns}
    generated = generated.rename(columns=rename_map)
    generated = generated.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in generated.columns:
        generated[col] = pd.to_numeric(generated[col], errors="coerce").fillna(0.0).astype("float32")

    out = pd.concat([frame[meta_cols + preserve].reset_index(drop=True), frame[base_features].reset_index(drop=True), generated], axis=1)
    feature_cols = base_features + list(generated.columns) + preserve
    out.to_parquet(_out_year_path(cache_family, year), index=False, compression="zstd")
    return feature_cols


def run(
    cache_family: str,
    base_features_path: str | None,
    primitives: list[str],
    max_rows_per_year: int | None,
    force: bool,
) -> dict:
    ensure_dirs()
    all_cols = build_cache(False)
    base_features = _load_base_features(base_features_path, all_cols)
    if not base_features:
        raise ValueError("No usable base features for Featuretools cache.")
    _cache_dir(cache_family).mkdir(parents=True, exist_ok=True)
    feature_cols: list[str] | None = None
    built_years = []
    years = list(dict.fromkeys(CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years))
    for year in years:
        path = _out_year_path(cache_family, year)
        if path.exists() and not force:
            print({"phase": "featuretools_cache_exists", "year": year, "path": str(path)}, flush=True)
            continue
        print({"phase": "featuretools_build_year", "year": year, "base_features": len(base_features), "primitives": primitives}, flush=True)
        feature_cols = _build_year(year, base_features, primitives, max_rows_per_year, cache_family)
        built_years.append(year)
        gc.collect()
    cols_path = _cache_dir(cache_family) / "feature_columns.json"
    if feature_cols is None:
        if not cols_path.exists():
            sample_year = CFG.split.train_years[0]
            cols = pd.read_parquet(_out_year_path(cache_family, sample_year)).columns.tolist()
            feature_cols = [c for c in cols if c not in META_COLS]
        else:
            feature_cols = json.loads(cols_path.read_text(encoding="utf-8"))
    cols_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "cache_family": cache_family,
        "source_cache_family": DEFAULT_CACHE_FAMILY,
        "base_features": base_features,
        "primitives": primitives,
        "preserved_features": [c for c in PRESERVE_FEATURES if c in feature_cols],
        "feature_count": len(feature_cols),
        "built_years": built_years,
        "feature_columns_path": str(cols_path),
    }
    report_path = _cache_dir(cache_family) / "featuretools_cache_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-family", type=str, default=FEATURETOOLS_CACHE_FAMILY)
    parser.add_argument("--base-features", type=str, default=None)
    parser.add_argument("--primitives", type=str, default=",".join(DEFAULT_PRIMITIVES))
    parser.add_argument("--max-rows-per-year", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    primitives = [p.strip() for p in args.primitives.split(",") if p.strip()]
    run(
        cache_family=args.cache_family,
        base_features_path=args.base_features,
        primitives=primitives,
        max_rows_per_year=args.max_rows_per_year or None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
