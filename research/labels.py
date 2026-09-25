from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from research.config import CFG
from research.data_io import CLOSE, HIGH, LOW, OPEN


@njit(cache=True)
def _buy_labels_numba(open_, high, low, close, r1, max_forward):
    n = len(open_)
    labels = np.empty(n, dtype=np.float64)
    entry_index = np.empty(n, dtype=np.int32)
    entry_price = np.empty(n, dtype=np.float64)
    for i in range(n):
        labels[i] = np.nan
        entry_index[i] = -1
        entry_price[i] = np.nan

    last_start = n - 2
    for candle in range(last_start):
        start = candle + 1
        entry = open_[start]
        sl = entry - r1
        tp = entry + r1
        entry_index[candle] = start
        entry_price[candle] = entry
        stop = start + max_forward + 1
        if stop > n:
            stop = n
        for j in range(start, stop):
            hit_tp = low[j] <= tp <= high[j]
            hit_sl = low[j] <= sl <= high[j]
            if hit_tp and hit_sl:
                labels[candle] = 1.0 if close[j] < open_[j] else 0.0
                break
            if hit_tp:
                labels[candle] = 1.0
                break
            if hit_sl:
                labels[candle] = 0.0
                break
            lo = open_[j] if open_[j] < entry else entry
            hi = entry if open_[j] < entry else open_[j]
            if lo <= sl <= hi:
                labels[candle] = 0.0
                break
            if lo <= tp <= hi:
                labels[candle] = 1.0
                break
    return labels, entry_index, entry_price


def buy_labels_next_open(data: pd.DataFrame) -> pd.DataFrame:
    """Create Buy Win/Lose labels for entry at next candle open.

    1 = TP hit first, 0 = SL hit first, NaN = unresolved within horizon.
    Ambiguous same-candle TP+SL follows the existing MoPhongDeals convention:
    for Buy, bearish candle resolves as Win, otherwise Lose.
    """

    high = data[HIGH].to_numpy(float)
    low = data[LOW].to_numpy(float)
    open_ = data[OPEN].to_numpy(float)
    close = data[CLOSE].to_numpy(float)
    labels, entry_index, entry_price = _buy_labels_numba(
        open_, high, low, close, float(CFG.trade.r1), int(CFG.label_max_forward)
    )

    out = data.copy()
    out["label"] = labels
    out["label_res"] = np.where(np.isnan(labels), "No res", np.where(labels == 1.0, "Win", "Lose"))
    out["entry_index"] = entry_index
    out["entry_price"] = entry_price
    return out


def buy_label_arrays(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Buy labels without copying the source DataFrame.

    This is intended for large M1 research jobs where creating a full
    ``data_handler.copy()`` only to read labels can push RAM into paging/OOM.
    """

    high = data[HIGH].to_numpy(float)
    low = data[LOW].to_numpy(float)
    open_ = data[OPEN].to_numpy(float)
    close = data[CLOSE].to_numpy(float)
    return _buy_labels_numba(open_, high, low, close, float(CFG.trade.r1), int(CFG.label_max_forward))
