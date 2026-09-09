from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from data_io import CLOSE, DATE, HIGH, LOW, OPEN, VOLUME, load_year
from labels import buy_label_arrays
from method.OverlayIndicatorsNormalization import rolling_rank_scale


FS14_NAME = "fs14_bb_overlay_norm_light"
POINT = 100.0


@njit(cache=True)
def _rolling_rank_selected(values: np.ndarray, indices: np.ndarray, window: int, min_periods: int, use_past_only: bool) -> np.ndarray:
    out = np.empty(len(indices), dtype=np.float32)
    for k in range(len(indices)):
        i = int(indices[k])
        end = i if use_past_only else i + 1
        start = end - window
        if start < 0:
            start = 0
        last = values[i]
        if not np.isfinite(last):
            out[k] = np.nan
            continue
        valid = 0
        less = 0
        equal = 0
        for j in range(start, end):
            v = values[j]
            if np.isfinite(v):
                valid += 1
                if v < last:
                    less += 1
                elif v == last:
                    equal += 1
        if valid < min_periods:
            out[k] = np.nan
        else:
            rank_pct = (less + 0.5 * equal) / valid
            out[k] = (rank_pct - 0.5) * 2.0
    return out


def _rank_selected(values, idx_keep: np.ndarray, window: int = 500, min_periods: int = 100, use_past_only: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return _rolling_rank_selected(arr, idx_keep.astype(np.int64), int(window), int(min_periods), bool(use_past_only))


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _sample_index(n: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(n), size=max_rows, replace=False)
    return np.sort(idx)


def _build_year_candidates(year: int, max_rows: int, seed: int, as_dict: bool = False) -> tuple[pd.DataFrame | dict[str, np.ndarray], list[str]]:
    raw = load_year(year)
    base = raw[[DATE, OPEN, HIGH, LOW, CLOSE] + ([VOLUME] if VOLUME in raw.columns else [])].copy()
    close = base[CLOSE]
    high = base[HIGH]
    low = base[LOW]
    open_ = base[OPEN]

    atr14 = _atr(high, low, close, 14)
    ret_1 = (close - open_) * POINT
    returns = close.diff() * POINT
    rsi_14 = _rsi(close, 14)
    ema8 = close.ewm(span=8, min_periods=8, adjust=False).mean()
    ema55 = close.ewm(span=55, min_periods=55, adjust=False).mean()
    ema_dist_8 = (close - ema8) * POINT
    ema_dist_55 = (close - ema55) * POINT
    roll20 = close.rolling(20, min_periods=20)
    mid20 = roll20.mean()
    std20 = roll20.std()
    upper20 = mid20 + 2 * std20
    lower20 = mid20 - 2 * std20
    bb20_pos = (close - lower20) / (upper20 - lower20).replace(0, np.nan)
    z_close_20 = (close - mid20) / std20.replace(0, np.nan)
    ret_sum_10 = returns.rolling(10, min_periods=10).sum()

    dt = pd.to_datetime(base[DATE])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    tod_sin = np.sin(2 * np.pi * minute_of_day / 1440)
    tod_cos = np.cos(2 * np.pi * minute_of_day / 1440)
    dow_sin = np.sin(2 * np.pi * dt.dt.dayofweek / 5)
    dow_cos = np.cos(2 * np.pi * dt.dt.dayofweek / 5)
    session_london_ny = ((dt.dt.hour >= 7) & (dt.dt.hour <= 21)).astype(np.int8)
    session_ny = ((dt.dt.hour >= 12) & (dt.dt.hour <= 21)).astype(np.int8)

    sig_bb_reversion_buy = ((bb20_pos < 0.18) & (rsi_14 < 45)).astype(np.int8)
    sig_pullback_trend_buy = ((ema_dist_55 > 0) & (ema_dist_8 < 0) & (ret_1 > 0)).astype(np.int8)
    sig_momentum_buy = ((ret_sum_10 > 0) & (z_close_20 > -0.5) & (z_close_20 < 1.5)).astype(np.int8)
    buy_signal = (
        (sig_bb_reversion_buy == 1)
        | (sig_pullback_trend_buy == 1)
        | ((sig_momentum_buy == 1) & (session_london_ny == 1))
    ).astype(np.int8)

    labels, entry_index, entry_price = buy_label_arrays(base)
    pre_mask = (np.asarray(buy_signal, dtype=np.int8) == 1) & ~np.isnan(labels)

    idx_all = np.flatnonzero(pre_mask)
    idx_keep = idx_all[_sample_index(len(idx_all), max_rows, seed)]

    def add_selected_feature(name: str, values, dtype: str = "float32") -> None:
        arr = values.to_numpy(copy=False) if isinstance(values, pd.Series) else np.asarray(values)
        feature_data[name] = arr[idx_keep].astype(dtype, copy=False)

    def add_selected_lag_feature(name: str, values, lag: int, dtype: str = "float32") -> None:
        arr = values.to_numpy(copy=False) if isinstance(values, pd.Series) else np.asarray(values)
        src_idx = idx_keep - int(lag)
        out = np.full(len(idx_keep), np.nan, dtype=np.float32)
        ok = src_idx >= 0
        out[ok] = arr[src_idx[ok]].astype(dtype, copy=False)
        feature_data[name] = out

    feature_data: dict[str, np.ndarray] = {
        "tod_sin": np.asarray(tod_sin)[idx_keep].astype("float32"),
        "tod_cos": np.asarray(tod_cos)[idx_keep].astype("float32"),
        "dow_sin": np.asarray(dow_sin)[idx_keep].astype("float32"),
        "dow_cos": np.asarray(dow_cos)[idx_keep].astype("float32"),
        "session_london_ny": np.asarray(session_london_ny)[idx_keep].astype("int8"),
        "session_ny": np.asarray(session_ny)[idx_keep].astype("int8"),
        "sig_bb_reversion_buy": np.asarray(sig_bb_reversion_buy)[idx_keep].astype("int8"),
        "sig_pullback_trend_buy": np.asarray(sig_pullback_trend_buy)[idx_keep].astype("int8"),
        "sig_momentum_buy": np.asarray(sig_momentum_buy)[idx_keep].astype("int8"),
        "buy_signal": np.asarray(buy_signal)[idx_keep].astype("int8"),
        "atr14": np.asarray(atr14)[idx_keep].astype("float32"),
    }
    for lag in [5, 8, 13, 21]:
        add_selected_lag_feature(f"close_diff_lag_{lag}", returns, lag)
        add_selected_lag_feature(f"body_lag_{lag}", ret_1, lag)

    mids: dict[int, pd.Series] = {}
    uppers: dict[int, pd.Series] = {}
    lowers: dict[int, pd.Series] = {}
    widths: dict[int, pd.Series] = {}
    poss: dict[int, pd.Series] = {}
    zs: dict[int, pd.Series] = {}

    for period in [20, 100, 200]:
        p = f"bb{period}"
        mid = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        width = upper - lower
        pos = (close - lower) / width.replace(0, np.nan)
        z = (close - mid) / std.replace(0, np.nan)
        mids[period], uppers[period], lowers[period], widths[period], poss[period], zs[period] = mid, upper, lower, width, pos, z

        for line_name, indicator in [("mid", mid), ("upper", upper), ("lower", lower)]:
            name = f"{p}_{line_name}_ov"
            value = (indicator - close) / atr14.replace(0, np.nan)
            add_selected_feature(f"{name}_value", value)
            for lag in (1, 3, 5, 8, 13):
                add_selected_lag_feature(f"{name}_value_lag{lag}", value, lag)

            add_selected_feature(f"{name}_distance", value)
            for lag in (1, 3, 5, 8, 13):
                add_selected_lag_feature(f"{name}_distance_lag{lag}", value, lag)

            rel_std = value.rolling(min(period, 50), min_periods=min(period, 50)).std()
            feature_data[f"{name}_std"] = _rank_selected(rel_std, idx_keep)

            for vel_period in (1, 3, 5, 8):
                velocity = indicator.diff(vel_period) / atr14.replace(0, np.nan)
                add_selected_feature(f"{name}_velocity{vel_period}", velocity)
                for lag in (1, 3):
                    add_selected_lag_feature(f"{name}_velocity{vel_period}_lag{lag}", velocity, lag)
                del velocity

            for acc_period in (1, 3, 5):
                velocity = indicator.diff(acc_period) / atr14.replace(0, np.nan)
                add_selected_feature(f"{name}_acceleration{acc_period}", velocity.diff(acc_period))
                del velocity
            del value, rel_std

        pos_clip = pos.clip(-0.5, 1.5)
        width_atr = (width * POINT) / (atr14 * POINT).replace(0, np.nan)
        mid_gap_atr = (close - mid) / atr14.replace(0, np.nan)
        lower_gap_atr = (close - lower) / atr14.replace(0, np.nan)
        direct = {
            f"{p}_pos_cycle_sin": np.sin(2 * np.pi * pos_clip),
            f"{p}_pos_cycle_cos": np.cos(2 * np.pi * pos_clip),
            f"{p}_pos_rank_500": _rank_selected(pos, idx_keep),
            f"{p}_z_rank_500": _rank_selected(z, idx_keep),
            f"{p}_width_atr": width_atr,
            f"{p}_width_atr_rank_500": _rank_selected(width_atr, idx_keep),
        }
        for win in [20, 100, 200]:
            direct[f"{p}_mid_gap_atr_rollmean_{win}"] = mid_gap_atr.rolling(win, min_periods=win).mean()
            direct[f"{p}_mid_gap_atr_rollstd_{win}"] = mid_gap_atr.rolling(win, min_periods=win).std()
            direct[f"{p}_lower_gap_atr_rollmean_{win}"] = lower_gap_atr.rolling(win, min_periods=win).mean()
            direct[f"{p}_lower_gap_atr_rollstd_{win}"] = lower_gap_atr.rolling(win, min_periods=win).std()
        for col, values in direct.items():
            arr = values.to_numpy(copy=False) if isinstance(values, pd.Series) else np.asarray(values)
            selected = arr if len(arr) == len(idx_keep) else arr[idx_keep]
            feature_data[col] = selected.astype("float32", copy=False)

    cross = {
        "bb20_bb100_pos_diff": poss[20] - poss[100],
        "bb20_bb200_pos_diff": poss[20] - poss[200],
        "bb100_bb200_pos_diff": poss[100] - poss[200],
        "bb20_width_over_bb100": widths[20] / widths[100].replace(0, np.nan),
        "bb20_width_over_bb200": widths[20] / widths[200].replace(0, np.nan),
        "bb100_width_over_bb200": widths[100] / widths[200].replace(0, np.nan),
        "bb_lower_agree_20_100": ((poss[20] <= 0.20) & (poss[100] <= 0.35)).astype(np.int8),
        "bb_lower_agree_20_200": ((poss[20] <= 0.20) & (poss[200] <= 0.40)).astype(np.int8),
        "bb_lower_agree_all": ((poss[20] <= 0.20) & (poss[100] <= 0.35) & (poss[200] <= 0.40)).astype(np.int8),
        "bb_trend_context_buy": (((mids[100] - mids[100].shift(5)) > 0) | ((mids[200] - mids[200].shift(5)) > 0)).astype(np.int8),
    }
    for col, values in cross.items():
        src = values.to_numpy(copy=False) if isinstance(values, pd.Series) else np.asarray(values)
        arr = src[idx_keep]
        feature_data[col] = arr.astype("float32" if arr.dtype.kind == "f" else "int8")

    feature_data[DATE] = base[DATE].iloc[idx_keep].to_numpy()
    feature_data["candle_index"] = idx_keep.astype("int32")
    feature_data["entry_index"] = entry_index[idx_keep].astype("int32")
    feature_data["entry_price"] = entry_price[idx_keep].astype("float32")
    feature_data["year"] = np.full(len(idx_keep), year, dtype="int16")
    feature_data["label"] = labels[idx_keep].astype("int8")
    feature_cols = [col for col in feature_data.keys() if col not in {DATE, "candle_index", "entry_index", "entry_price", "year", "label"}]
    finite_mask = np.ones(len(idx_keep), dtype=bool)
    for col in feature_cols:
        arr = feature_data[col]
        if np.issubdtype(arr.dtype, np.floating):
            finite_mask &= np.isfinite(arr)
    if not finite_mask.all():
        for col in list(feature_data.keys()):
            feature_data[col] = feature_data[col][finite_mask]
    if as_dict:
        out = feature_data
    else:
        out = pd.DataFrame(feature_data, copy=False).reset_index(drop=True)

    del raw, base, labels, entry_index, entry_price
    gc.collect()
    return out, feature_cols


class RobustClipScalerLite:
    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, x: np.ndarray):
        self.lower_ = np.nanquantile(x, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(x, self.upper_q, axis=0)
        clipped = np.clip(x, self.lower_, self.upper_)
        self.median_ = np.nanmedian(clipped, axis=0)
        q25 = np.nanquantile(clipped, 0.25, axis=0)
        q75 = np.nanquantile(clipped, 0.75, axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.clip(x, self.lower_, self.upper_)
        arr = (arr - self.median_) / self.scale_
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def transform_inplace(self, x: np.ndarray) -> np.ndarray:
        np.clip(x, self.lower_, self.upper_, out=x)
        x -= self.median_.astype(x.dtype, copy=False)
        x /= self.scale_.astype(x.dtype, copy=False)
        return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins * 5 or len(actual) < bins * 5:
        return np.nan
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _decile_charts(train_df: pd.DataFrame, cols: list[str], top_features: list[str], chart_dir: Path) -> pd.DataFrame:
    rows = []
    for feature in top_features:
        if train_df[feature].nunique(dropna=True) < 5:
            continue
        temp = train_df[[feature, "label"]].dropna()
        try:
            temp["_bin"] = pd.qcut(temp[feature], q=10, duplicates="drop")
        except ValueError:
            continue
        grouped = temp.groupby("_bin", observed=True)["label"].agg(["count", "mean"]).reset_index()
        grouped["feature"] = feature
        grouped["decile"] = np.arange(1, len(grouped) + 1)
        rows.append(grouped[["feature", "decile", "count", "mean"]])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(grouped["decile"], grouped["mean"] * 100, marker="o")
        ax.axhline(train_df["label"].mean() * 100, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{feature}: train decile winrate")
        ax.set_xlabel("Feature decile")
        ax.set_ylabel("Winrate %")
        fig.tight_layout()
        fig.savefig(chart_dir / f"decile_lift_{feature}.png", dpi=140)
        plt.close(fig)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(max_rows_per_year: int, max_mi_rows: int, max_probe_rows: int) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FS14_NAME}_{run_id}"
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    feature_cols: list[str] | None = None
    for year in CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years:
        print({"phase": "build_year", "year": year}, flush=True)
        year_df, cols = _build_year_candidates(year, max_rows=max_rows_per_year, seed=9000 + year)
        feature_cols = feature_cols or cols
        frames.append(year_df)
        print({"phase": "build_year_done", "year": year, "rows": len(year_df)}, flush=True)

    assert feature_cols is not None
    data = pd.concat(frames, ignore_index=True)
    train = data[data["year"].isin(CFG.split.train_years)].reset_index(drop=True)
    valid = data[data["year"].isin(CFG.split.valid_years)].reset_index(drop=True)
    test = data[data["year"].isin(CFG.split.test_years)].reset_index(drop=True)

    x_train_raw = train[feature_cols].to_numpy("float32")
    y_train = train["label"].to_numpy("int8")
    x_valid_raw = valid[feature_cols].to_numpy("float32")
    y_valid = valid["label"].to_numpy("int8")
    x_test_raw = test[feature_cols].to_numpy("float32")
    y_test = test["label"].to_numpy("int8")

    scaler = RobustClipScalerLite().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)
    x_test = scaler.transform(x_test_raw)

    print({"phase": "behavior_mi", "train_rows": len(y_train), "n_features": len(feature_cols)}, flush=True)
    mi_idx = _sample_index(len(y_train), max_mi_rows, 901)
    mi = mutual_info_classif(x_train[mi_idx], y_train[mi_idx], random_state=901)
    mi_df = pd.DataFrame({"feature": feature_cols, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    mi_path = out_dir / "mutual_info.csv"
    mi_df.to_csv(mi_path, index=False, encoding="utf-8-sig")

    top_mi = mi_df.head(15)["feature"].tolist()
    lift_df = _decile_charts(train.iloc[mi_idx].reset_index(drop=True), feature_cols, top_mi[:10], chart_dir)
    lift_path = out_dir / "decile_lift_top10.csv"
    lift_df.to_csv(lift_path, index=False, encoding="utf-8-sig")

    print({"phase": "behavior_probe_rf"}, flush=True)
    probe_idx = _sample_index(len(y_train), max_probe_rows, 902)
    probe = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=250,
        min_samples_split=700,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.8,
        class_weight=None,
        random_state=902,
        n_jobs=-1,
    )
    probe.fit(x_train[probe_idx], y_train[probe_idx])
    p_valid = probe.predict_proba(x_valid)[:, 1]
    p_test = probe.predict_proba(x_test)[:, 1]
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": probe.feature_importances_}).sort_values("importance", ascending=False)
    imp_path = out_dir / "probe_rf_importance.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    top_imp = imp_df.head(20).sort_values("importance")
    ax.barh(top_imp["feature"], top_imp["importance"])
    ax.set_title("Probe RF importance top 20")
    fig.tight_layout()
    imp_chart = chart_dir / "probe_rf_importance_top20.png"
    fig.savefig(imp_chart, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print({"phase": "behavior_drift"}, flush=True)
    psi_rows = []
    for feature in sorted(set(top_mi + imp_df.head(15)["feature"].tolist())):
        train_vals = train[feature].to_numpy(float)
        for year in CFG.split.valid_years + CFG.split.test_years:
            actual = data.loc[data["year"] == year, feature].to_numpy(float)
            psi_rows.append({"feature": feature, "year": year, "psi_vs_train": _psi(train_vals, actual)})
    psi_df = pd.DataFrame(psi_rows)
    psi_path = out_dir / "top_feature_psi_vs_train.csv"
    psi_df.to_csv(psi_path, index=False, encoding="utf-8-sig")

    summary = {
        "run_id": run_id,
        "feature_set": FS14_NAME,
        "n_features": len(feature_cols),
        "max_rows_per_year": max_rows_per_year,
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "rows_test": int(len(test)),
        "valid_auc_probe": float(roc_auc_score(y_valid, p_valid)),
        "test_auc_probe": float(roc_auc_score(y_test, p_test)),
        "top_mi": mi_df.head(15).to_dict(orient="records"),
        "top_probe_importance": imp_df.head(15).to_dict(orient="records"),
        "max_top_feature_psi": float(psi_df["psi_vs_train"].max()) if len(psi_df) else np.nan,
        "elapsed_sec": round(time.time() - started, 2),
        "artifacts": {
            "mutual_info": str(mi_path),
            "decile_lift": str(lift_path),
            "probe_importance": str(imp_path),
            "psi": str(psi_path),
            "importance_chart": str(imp_chart),
            "chart_dir": str(chart_dir),
        },
    }
    summary_path = out_dir / "behavior_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name=f"{FS14_NAME}_behavior_{run_id}"):
            mlflow.log_params({
                "feature_set": FS14_NAME,
                "n_features": len(feature_cols),
                "max_rows_per_year": max_rows_per_year,
                "max_mi_rows": max_mi_rows,
                "max_probe_rows": max_probe_rows,
            })
            mlflow.log_metric("valid_auc_probe", summary["valid_auc_probe"])
            mlflow.log_metric("test_auc_probe", summary["test_auc_probe"])
            mlflow.log_metric("max_top_feature_psi", summary["max_top_feature_psi"])
            for path in [mi_path, lift_path, imp_path, psi_path, summary_path]:
                mlflow.log_artifact(str(path))
            for path in chart_dir.glob("*.png"):
                mlflow.log_artifact(str(path), artifact_path="charts")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows-per-year", type=int, default=25000)
    parser.add_argument("--max-mi-rows", type=int, default=80000)
    parser.add_argument("--max-probe-rows", type=int, default=100000)
    args = parser.parse_args()
    run(
        max_rows_per_year=args.max_rows_per_year,
        max_mi_rows=args.max_mi_rows,
        max_probe_rows=args.max_probe_rows,
    )
