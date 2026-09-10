import numpy as np
import pandas as pd

from data_handler import CLOSE, HIGH, LOW


BUY = 1
SELL = -1
SIDEWAY = 0


def _series(data, name, default=None):
    if name is None:
        return default
    if name not in data.columns:
        raise KeyError(f"Missing column: {name}")
    value = data[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value.reset_index(drop=True)


def _sign_to_dir(values, zero_mode="ffill"):
    direction = np.sign(pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)).astype(np.int8)
    if zero_mode == "sideway":
        return pd.Series(direction)
    if zero_mode != "ffill":
        raise ValueError(f"Unsupported zero_mode: {zero_mode}")

    out = direction.copy()
    last = BUY
    for i in range(len(out)):
        if out[i] == SIDEWAY:
            out[i] = last
        else:
            last = out[i]
    return pd.Series(out)


def build_indicator_signal(data, indicator_type, **params):
    """
    Return a directional signal series aligned with data_handler.

    Supported indicator_type values:
    - ma_cross / line_cross: fast_col crosses slow_col. buy when fast > slow.
    - price_line_cross: price_col crosses line_col. buy when price > line.
    - zero_cross: value_col crosses zero. buy when value > 0.
    - threshold: buy when value >= upper, sell when value <= lower; sideway is carried by default.
    - slope_sign: buy when value - value.shift(lookback) > 0.
    - di_cross: buy when plus_col > minus_col.
    - macd_cross: buy when macd_col > signal_col.
    - band_mid_cross: buy when price_col > mid_col.
    - signal_col: use an existing column containing buy/sell, 1/-1, or signed values.
    """
    zero_mode = params.get("zero_mode", "ffill")

    if indicator_type in ["ma_cross", "line_cross"]:
        fast = _series(data, params.get("fast_col"))
        slow = _series(data, params.get("slow_col"))
        return _sign_to_dir(fast - slow, zero_mode=zero_mode)

    if indicator_type == "price_line_cross":
        price = _series(data, params.get("price_col", CLOSE))
        line = _series(data, params.get("line_col"))
        return _sign_to_dir(price - line, zero_mode=zero_mode)

    if indicator_type == "zero_cross":
        value = _series(data, params.get("value_col"))
        return _sign_to_dir(value, zero_mode=zero_mode)

    if indicator_type == "threshold":
        value = _series(data, params.get("value_col"))
        upper = params.get("upper")
        lower = params.get("lower")
        if upper is None or lower is None:
            raise ValueError("threshold requires upper and lower")
        direction = pd.Series(np.zeros(len(data), dtype=np.int8))
        direction[value >= upper] = BUY
        direction[value <= lower] = SELL
        return _sign_to_dir(direction, zero_mode=zero_mode)

    if indicator_type == "slope_sign":
        value = _series(data, params.get("value_col"))
        lookback = int(params.get("lookback", 3))
        return _sign_to_dir(value - value.shift(lookback), zero_mode=zero_mode)

    if indicator_type == "di_cross":
        plus = _series(data, params.get("plus_col", "PLUS_DI14"))
        minus = _series(data, params.get("minus_col", "MINUS_DI14"))
        return _sign_to_dir(plus - minus, zero_mode=zero_mode)

    if indicator_type == "macd_cross":
        macd = _series(data, params.get("macd_col", "MACD"))
        signal = _series(data, params.get("signal_col", "MACD_SIGNAL"))
        return _sign_to_dir(macd - signal, zero_mode=zero_mode)

    if indicator_type == "band_mid_cross":
        price = _series(data, params.get("price_col", CLOSE))
        mid = _series(data, params.get("mid_col"))
        return _sign_to_dir(price - mid, zero_mode=zero_mode)

    if indicator_type == "signal_col":
        raw = _series(data, params.get("signal_col"))
        if raw.dtype == object:
            mapped = raw.astype(str).str.lower().map({"buy": BUY, "sell": SELL, "sideway": SIDEWAY, "hold": SIDEWAY})
            raw = mapped.fillna(0)
        return _sign_to_dir(raw, zero_mode=zero_mode)

    raise ValueError(f"Unsupported indicator_type: {indicator_type}")


def build_cycle_features_from_signal(
    signal,
    prefix="cycle",
    max_clip=120,
    age_buckets=None,
    include_future=False,
):
    """
    Convert a directional signal into cycle features.

    The first candle after a direction change has cycle_bar = 0, the next is 1,
    and the counter continues until the signal changes again.

    include_future=False by default because cycle length and bars_to_end need
    future candles and will leak information in model training.
    """
    direction = _sign_to_dir(signal, zero_mode="ffill").astype(np.int8)
    changed = np.r_[True, direction.to_numpy()[1:] != direction.to_numpy()[:-1]]
    cycle_id = np.cumsum(changed)
    cycle_bar = pd.Series(np.arange(len(direction))).groupby(cycle_id).cumcount().to_numpy()

    features = pd.DataFrame(index=range(len(direction)))
    features[f"{prefix}_signal_dir"] = direction.to_numpy(np.int8)
    features[f"{prefix}_id"] = cycle_id.astype(np.int64)
    features[f"{prefix}_bar"] = cycle_bar.astype(np.int32)
    features[f"{prefix}_bar_clip"] = np.minimum(cycle_bar, max_clip).astype(np.int32)
    features[f"{prefix}_bar_log"] = np.log1p(cycle_bar)
    features[f"{prefix}_start"] = (cycle_bar == 0).astype(float)
    features[f"{prefix}_changed"] = changed.astype(float)

    if age_buckets is None:
        age_buckets = [(0, 3), (4, 10), (11, 30), (31, 60), (61, None)]
    for start, end in age_buckets:
        if end is None:
            mask = cycle_bar >= start
            name = f"{prefix}_age_{start}_plus"
        else:
            mask = (cycle_bar >= start) & (cycle_bar <= end)
            name = f"{prefix}_age_{start}_{end}"
        features[name] = mask.astype(float)

    if include_future:
        cycle_len = pd.Series(cycle_bar).groupby(cycle_id).transform("max").to_numpy() + 1
        bars_to_end = cycle_len - cycle_bar - 1
        features[f"{prefix}_len"] = cycle_len.astype(np.int32)
        features[f"{prefix}_progress"] = np.where(cycle_len > 1, cycle_bar / (cycle_len - 1), 0.0)
        features[f"{prefix}_bars_to_end"] = bars_to_end.astype(np.int32)
        features[f"{prefix}_end"] = (bars_to_end == 0).astype(float)

    return features


def add_cycle_context_features(data, features, prefix, reference_col=None, norm_col=None, high_col=HIGH, low_col=LOW):
    """
    Add optional context features around the cycle.

    reference_col is usually the indicator line that price returns to, for
    example MA20, BB mid, VWAP, PSAR, or any custom indicator column.
    """
    if reference_col is None:
        return features

    ref = _series(data, reference_col)
    close = _series(data, CLOSE)
    high = _series(data, high_col)
    low = _series(data, low_col)
    direction = features[f"{prefix}_signal_dir"].to_numpy(np.int8)
    cycle_id = features[f"{prefix}_id"].to_numpy()

    scale = _series(data, norm_col) if norm_col is not None else pd.Series(np.ones(len(data)))
    scale = scale.replace(0, np.nan)

    extension = ((close - ref) / scale) * direction
    features[f"{prefix}_price_extension"] = extension

    touched = ((low <= ref) & (high >= ref)).fillna(False).to_numpy()
    first_touch = np.zeros(len(data), dtype=float)
    touch_count = np.zeros(len(data), dtype=np.int16)
    last_cycle = None
    touches = 0
    for i, cid in enumerate(cycle_id):
        if cid != last_cycle:
            last_cycle = cid
            touches = 0
        if touched[i]:
            touches += 1
            if touches == 1:
                first_touch[i] = 1.0
        touch_count[i] = touches

    features[f"{prefix}_first_touch_ref"] = first_touch
    features[f"{prefix}_touch_ref_count"] = touch_count
    features[f"{prefix}_touch_ref_count_clip"] = np.minimum(touch_count, 10)
    return features


def create_cycle_features(data, indicator_type, prefix=None, include_future=False, **params):
    """
    Main helper for strategy research.

    Example:
        features = create_cycle_features(
            data_handler,
            "ma_cross",
            prefix="ma5_ma20",
            fast_col="MA5",
            slow_col="MA20",
            reference_col="MA20",
            norm_col="norm_scale",
        )

    Returns a DataFrame with only generated feature columns. Join it to the
    original data_handler with pd.concat([data_handler, features], axis=1).
    """
    if prefix is None:
        prefix = str(indicator_type)

    signal = build_indicator_signal(data, indicator_type, **params)
    features = build_cycle_features_from_signal(
        signal,
        prefix=prefix,
        max_clip=int(params.get("max_clip", 120)),
        age_buckets=params.get("age_buckets"),
        include_future=include_future,
    )
    features = add_cycle_context_features(
        data,
        features,
        prefix,
        reference_col=params.get("reference_col"),
        norm_col=params.get("norm_col"),
        high_col=params.get("high_col", HIGH),
        low_col=params.get("low_col", LOW),
    )
    return features


def append_cycle_features(data, indicator_type, prefix=None, include_future=False, **params):
    features = create_cycle_features(data, indicator_type, prefix=prefix, include_future=include_future, **params)
    return pd.concat([data.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
