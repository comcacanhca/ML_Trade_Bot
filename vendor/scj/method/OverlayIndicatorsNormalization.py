# Chuẩn hóa cho các features của OverlayIndicators và các đại lượng không có
# biên trên/dưới hoặc các đại lượng trong tương lai có thể nằm ngoài dữ liệu
# của data_handler train như các giá trị OHLC.
#
# Phương pháp 1: Dùng rolling range / rolling z-score scaling
# Thay vì chuẩn hóa trên toàn bộ tập Train (Global), chuẩn hóa theo cửa sổ
# di động (Moving Window):
#     Z_t = (X_t - mean_k) / std_k
# Trong đó mean_k và std_k là trung bình và độ lệch chuẩn của K khoảng thời gian
# gần nhất trước đó. Khi use_past_only=True, giá trị tại row t chỉ dùng các row
# trước t, không dùng row tương lai.
#
# Phương pháp 2: Dùng Log-Transform
# Nếu giá trị tuyệt đối tăng quá nhanh, dùng log(x) hoặc signed log:
#     sign(x) * log(1 + abs(x))
# để nén các giá trị cực lớn và giữ được dấu âm/dương.
#
# Phương pháp 3: Tín hiệu vi phân / tốc độ thay đổi
# Thay vì đưa giá trị tuyệt đối X_t cho mô hình, chuyển thành mức biến động:
#     diff = X_t - X_{t-1}
#     pct_change = (X_t - X_{t-1}) / X_{t-1}
# Các đại lượng này thường ổn định hơn cho mô hình cây.
#
# Quy chuẩn từ nghiên cứu hiện tại:
# - Overlay value/distance ưu tiên dùng khoảng cách so với Close chia ATR:
#       (indicator_t - close_t) / ATR_t
# - Overlay velocity/acceleration ưu tiên chia ATR để giảm phụ thuộc scale giá.
# - Overlay std tốt nhất hiện tại dùng:
#       relative_distance_std_rolling_rank_500
# - Mọi rolling z/rank mặc định là causal: row t chỉ bị ảnh hưởng bởi các row
#   trước t.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


NumberLike = pd.Series | np.ndarray | list[float]
RollingMethod = Literal["rolling_zscore", "rolling_wma_zscore", "rolling_minmax", "rolling_rank"]
StationaryMethod = Literal["diff", "pct_change", "signed_log_diff"]
OverlayValueMethod = Literal["atr_gap", "pct_distance", "rolling_zscore", "rolling_wma_zscore", "rolling_minmax", "rolling_rank"]


@dataclass(frozen=True)
class OverlayNormalizationConfig:
    """
    Cấu hình chuẩn hóa overlay indicators.

    Mặc định dùng rolling causal:
    - window=500
    - min_periods=100
    - use_past_only=True
    """

    window: int = 500
    min_periods: int = 100
    use_past_only: bool = True
    eps: float = 1e-12


def _to_series(values: NumberLike, name: str | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        out = values.copy()
    else:
        out = pd.Series(values)
    if name is not None:
        out.name = name
    return out.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def _safe_div(a: NumberLike, b: NumberLike, eps: float = 1e-12) -> pd.Series:
    left = _to_series(a)
    right = _to_series(b).replace(0, np.nan)
    return left / right.where(right.abs() > eps)


def _past(series: pd.Series, use_past_only: bool = True) -> pd.Series:
    return series.shift(1) if use_past_only else series


def rolling_zscore(
    values: NumberLike,
    window: int = 500,
    min_periods: int = 100,
    use_past_only: bool = True,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Rolling Z-score causal:
        z_t = (x_t - mean(x_{t-window:t-1})) / std(x_{t-window:t-1})

    Nếu use_past_only=False, rolling window bao gồm cả row hiện tại.
    Khi train/inference time-series nên giữ mặc định True.
    """

    x = _to_series(values)
    ref = _past(x, use_past_only=use_past_only)
    mean = ref.rolling(window, min_periods=min_periods).mean()
    std = ref.rolling(window, min_periods=min_periods).std()
    return (x - mean) / std.where(std.abs() > eps)


def linear_wma(
    values: NumberLike,
    window: int = 14,
    min_periods: int = 1,
) -> pd.Series:
    """
    Linear Weighted Moving Average:
        WMA_t = sum(x_i * weight_i) / sum(weight_i)

    Weights tang tuyến tính 1..window, giá trị mới nhất có weight lớn nhất.
    Hàm này chỉ dùng rolling window quá khứ tới row hiện tại, không dùng dữ liệu tương lai.
    """

    x = _to_series(values)
    weights = np.arange(1, int(window) + 1, dtype=np.float64)

    def calc(arr: np.ndarray) -> float:
        valid = np.isfinite(arr)
        if not valid.any():
            return np.nan
        arr = arr[valid]
        active_weights = weights[-len(arr):]
        return float(np.dot(arr, active_weights) / active_weights.sum())

    return x.rolling(window=int(window), min_periods=int(min_periods)).apply(calc, raw=True)


def rolling_wma_zscore(
    values: NumberLike,
    window: int = 14,
    min_periods: int = 1,
    use_past_only: bool = False,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Rolling WMA Z-score:
        z_t = (x_t - WMA_t) / rolling_std_t

    Mặc định use_past_only=False để khớp công thức anh đưa:
    - WMA và std tại row t dùng cửa sổ kết thúc ở chính row t.
    - Không dùng dữ liệu tương lai.

    Nếu use_past_only=True:
    - WMA/std tại row t chỉ dùng dữ liệu tới row t-1.
    - Phù hợp khi muốn feature không chứa chính giá trị hiện tại trong baseline normalization.
    """

    x = _to_series(values)
    ref = _past(x, use_past_only=use_past_only)
    wma = linear_wma(ref, window=window, min_periods=min_periods)
    std = ref.rolling(window=int(window), min_periods=int(min_periods)).std()
    z = (x - wma) / std.where(std.abs() > eps)
    same_value = (x - wma).abs() <= eps
    return z.where(std.abs() > eps, np.where(same_value, 0.0, np.nan))


def rolling_minmax_scale(
    values: NumberLike,
    window: int = 500,
    min_periods: int = 100,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    use_past_only: bool = True,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Rolling MinMax causal về khoảng mặc định [-1, 1].
    """

    x = _to_series(values)
    ref = _past(x, use_past_only=use_past_only)
    roll_min = ref.rolling(window, min_periods=min_periods).min()
    roll_max = ref.rolling(window, min_periods=min_periods).max()
    denom = (roll_max - roll_min).where((roll_max - roll_min).abs() > eps)
    lo, hi = feature_range
    scaled_01 = (x - roll_min) / denom
    return scaled_01 * (hi - lo) + lo


def rolling_rank_scale(
    values: NumberLike,
    window: int = 500,
    min_periods: int = 100,
    use_past_only: bool = True,
) -> pd.Series:
    """
    Rolling rank causal về [-1, 1].

    Giá trị tại row t được xếp hạng so với các row trước đó trong rolling window.
    """

    x = _to_series(values)
    if use_past_only:

        def rank_last(arr: np.ndarray) -> float:
            current = arr[-1]
            past = arr[:-1]
            if len(past) == 0 or not np.isfinite(current):
                return np.nan
            past = past[np.isfinite(past)]
            if len(past) == 0:
                return np.nan
            return float(np.mean(past <= current))

        return (x.rolling(window + 1, min_periods=min_periods + 1).apply(rank_last, raw=True) - 0.5) * 2.0

    return (x.rolling(window, min_periods=min_periods).rank(pct=True) - 0.5) * 2.0


def signed_log1p(values: NumberLike) -> pd.Series:
    """
    Log transform dùng được cho giá trị âm/dương:
        sign(x) * log(1 + abs(x))
    """

    x = _to_series(values)
    return np.sign(x) * np.log1p(x.abs())


def stationary_transform(
    values: NumberLike,
    method: StationaryMethod = "diff",
    periods: int = 1,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Biến đổi chuỗi không dừng thành đại lượng biến động.
    """

    x = _to_series(values)
    if method == "diff":
        return x.diff(periods)
    if method == "pct_change":
        prev = x.shift(periods)
        return (x - prev) / prev.where(prev.abs() > eps)
    if method == "signed_log_diff":
        return signed_log1p(x.diff(periods))
    raise ValueError(f"Unsupported stationary method: {method}")


def lag_features(
    values: NumberLike,
    name: str,
    periods: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Tao lag snapshot cho chuoi da duoc bieu dien.

    Ham nay chi shift ve qua khu, khong dung row tuong lai.
    Nen uu tien lag cua chuoi da normalize/stationary nhu value, distance, velocity.
    """

    x = _to_series(values)
    out = pd.DataFrame(index=range(len(x)))
    for p in periods:
        p = int(p)
        if p <= 0:
            raise ValueError(f"lag period must be positive: {p}")
        out[f"{name}_lag{p}"] = x.shift(p)
    return out


def clip_by_train_quantile(
    train_values: NumberLike,
    values: NumberLike,
    low_q: float = 0.005,
    high_q: float = 0.995,
) -> pd.Series:
    """
    Winsorization theo ngưỡng học từ train.

    Dùng khi đã chia train/test. Không fit percentile trên full dataset.
    """

    train = _to_series(train_values)
    x = _to_series(values)
    low = float(train.quantile(low_q))
    high = float(train.quantile(high_q))
    return x.clip(low, high)


def apply_rolling_method(
    values: NumberLike,
    method: RollingMethod,
    config: OverlayNormalizationConfig | None = None,
) -> pd.Series:
    cfg = config or OverlayNormalizationConfig()
    if method == "rolling_zscore":
        return rolling_zscore(values, cfg.window, cfg.min_periods, cfg.use_past_only, cfg.eps)
    if method == "rolling_wma_zscore":
        return rolling_wma_zscore(values, window=14, min_periods=1, use_past_only=False, eps=cfg.eps)
    if method == "rolling_minmax":
        return rolling_minmax_scale(values, cfg.window, cfg.min_periods, use_past_only=cfg.use_past_only, eps=cfg.eps)
    if method == "rolling_rank":
        return rolling_rank_scale(values, cfg.window, cfg.min_periods, cfg.use_past_only)
    raise ValueError(f"Unsupported rolling method: {method}")

def normalize_overlay_indicator(
    indicator: NumberLike,
    close: NumberLike,
    atr: NumberLike,
    name: str,
    config: OverlayNormalizationConfig | None = None,
    value_method: OverlayValueMethod = "atr_gap",
    value_lag_periods: Iterable[int] = (1, 3, 5),
    distance_lag_periods: Iterable[int] = (1, 3, 5),
    raw_lag_periods: Iterable[int] = (),
    velocity_periods: Iterable[int] = (1, 3, 5),
    velocity_lag_periods: Iterable[int] = (),
    acceleration_periods: Iterable[int] = (1, 3, 5),
    std_window: int = 20,
    std_mode: Literal[
        "raw",
        "raw_over_atr",
        "relative_distance",
        "relative_distance_rolling_z",
        "relative_distance_rolling_rank",
    ] = "relative_distance_rolling_rank",
) -> pd.DataFrame:
    """
    Tạo bộ feature chuẩn hóa cho một overlay indicator.

    Output mặc định:
    - {name}_value
    - {name}_distance
    - {name}_std
    - {name}_velocity{p}
    - {name}_acceleration{p}
    """

    cfg = config or OverlayNormalizationConfig()
    x = _to_series(indicator, name=name)
    close_s = _to_series(close)
    atr_s = _to_series(atr)

    if value_method == "atr_gap":
        value = overlay_value_atr_gap(x, close_s, atr_s, eps=cfg.eps)
    elif value_method == "pct_distance":
        value = overlay_pct_distance(x, close_s, eps=cfg.eps)
    elif value_method in {"rolling_zscore", "rolling_wma_zscore", "rolling_minmax", "rolling_rank"}:
        value = apply_rolling_method(x, value_method, cfg)
    else:
        raise ValueError(f"Unsupported value_method: {value_method}")

    distance = overlay_value_atr_gap(x, close_s, atr_s, eps=cfg.eps)

    out = pd.DataFrame(index=range(len(x)))
    out[f"{name}_value"] = value
    if value_lag_periods:
        out = pd.concat([out, lag_features(value, f"{name}_value", value_lag_periods)], axis=1)

    out[f"{name}_distance"] = distance
    if distance_lag_periods:
        out = pd.concat([out, lag_features(distance, f"{name}_distance", distance_lag_periods)], axis=1)

    if raw_lag_periods:
        out = pd.concat([out, lag_features(x, f"{name}_raw", raw_lag_periods)], axis=1)

    out[f"{name}_std"] = overlay_std(x, close_s, atr_s, std_window=std_window, mode=std_mode, config=cfg)

    for p in velocity_periods:
        p = int(p)
        velocity = overlay_velocity(x, atr=atr_s, periods=p, eps=cfg.eps)
        out[f"{name}_velocity{p}"] = velocity
        if velocity_lag_periods:
            out = pd.concat([out, lag_features(velocity, f"{name}_velocity{p}", velocity_lag_periods)], axis=1)

    for p in acceleration_periods:
        p = int(p)
        out[f"{name}_acceleration{p}"] = overlay_acceleration(x, atr=atr_s, periods=p, eps=cfg.eps)

    return out.replace([np.inf, -np.inf], np.nan)


def normalize_overlay_indicators(
    data: pd.DataFrame,
    indicator_cols: Iterable[str],
    close_col: str = "Close",
    atr_col: str = "ATR14",
    config: OverlayNormalizationConfig | None = None,
    value_method: OverlayValueMethod = "atr_gap",
    value_lag_periods: Iterable[int] = (1, 3, 5),
    distance_lag_periods: Iterable[int] = (1, 3, 5),
    raw_lag_periods: Iterable[int] = (),
    velocity_periods: Iterable[int] = (1, 3, 5),
    velocity_lag_periods: Iterable[int] = (),
    acceleration_periods: Iterable[int] = (1, 3, 5),
    std_mode: Literal[
        "raw",
        "raw_over_atr",
        "relative_distance",
        "relative_distance_rolling_z",
        "relative_distance_rolling_rank",
    ] = "relative_distance_rolling_rank",
) -> pd.DataFrame:
    """
    Chuẩn hóa nhiều overlay indicators từ một DataFrame.
    """

    if close_col not in data.columns:
        raise KeyError(f"Missing close_col: {close_col}")
    if atr_col not in data.columns:
        raise KeyError(f"Missing atr_col: {atr_col}")

    frames = []
    for col in indicator_cols:
        if col not in data.columns:
            raise KeyError(f"Missing indicator column: {col}")
        frames.append(
            normalize_overlay_indicator(
                indicator=data[col],
                close=data[close_col],
                atr=data[atr_col],
                name=col,
                config=config,
                value_method=value_method,
                value_lag_periods=value_lag_periods,
                distance_lag_periods=distance_lag_periods,
                raw_lag_periods=raw_lag_periods,
                velocity_periods=velocity_periods,
                velocity_lag_periods=velocity_lag_periods,
                acceleration_periods=acceleration_periods,
                std_mode=std_mode,
            )
        )

    if not frames:
        return pd.DataFrame(index=data.index)

    return pd.concat(frames, axis=1).set_index(data.index)


###############################################OVERLAY INDS FEATURE REQUIRED###############################################
def overlay_value_atr_gap(
    indicator: NumberLike,
    close: NumberLike,
    atr: NumberLike,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Chuẩn hóa giá trị overlay theo khoảng cách đến Close / ATR:
        (indicator_t - close_t) / ATR_t
    """

    return _safe_div(_to_series(indicator) - _to_series(close), _to_series(atr), eps=eps)


def overlay_pct_distance(
    indicator: NumberLike,
    close: NumberLike,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Percentage distance:
        (indicator_t - close_t) / close_t
    """

    return _safe_div(_to_series(indicator) - _to_series(close), _to_series(close), eps=eps)


def overlay_velocity(
    indicator: NumberLike,
    atr: NumberLike | None = None,
    periods: int = 1,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Vận tốc overlay.

    Nếu có ATR:
        (indicator_t - indicator_{t-periods}) / ATR_t
    Nếu không có ATR:
        indicator_t - indicator_{t-periods}
    """

    x = _to_series(indicator)
    velocity = x.diff(periods)
    if atr is None:
        return velocity
    return _safe_div(velocity, _to_series(atr), eps=eps)


def overlay_acceleration(
    indicator: NumberLike,
    atr: NumberLike | None = None,
    periods: int = 1,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Gia tốc overlay = diff của vận tốc.
    """

    v = overlay_velocity(indicator, atr=atr, periods=periods, eps=eps)
    return v.diff(periods)


def overlay_std(
    indicator: NumberLike,
    close: NumberLike | None = None,
    atr: NumberLike | None = None,
    std_window: int = 20,
    mode: Literal[
        "raw",
        "raw_over_atr",
        "relative_distance",
        "relative_distance_rolling_z",
        "relative_distance_rolling_rank",
    ] = "relative_distance_rolling_rank",
    config: OverlayNormalizationConfig | None = None,
) -> pd.Series:
    """
    Chuẩn hóa đại lượng std của overlay indicator.

    Mode khuyến nghị theo nghiên cứu hiện tại:
        relative_distance_rolling_rank
    """

    cfg = config or OverlayNormalizationConfig()
    x = _to_series(indicator)

    if mode == "raw":
        return x.rolling(std_window, min_periods=std_window).std()

    if atr is None:
        raise ValueError(f"mode={mode} requires atr")

    atr_s = _to_series(atr)

    if mode == "raw_over_atr":
        return _safe_div(x.rolling(std_window, min_periods=std_window).std(), atr_s, eps=cfg.eps)

    if close is None:
        raise ValueError(f"mode={mode} requires close")

    rel = _safe_div(x - _to_series(close), atr_s, eps=cfg.eps)
    rel_std = rel.rolling(std_window, min_periods=std_window).std()

    if mode == "relative_distance":
        return rel_std
    if mode == "relative_distance_rolling_z":
        return rolling_zscore(rel_std, cfg.window, cfg.min_periods, cfg.use_past_only, cfg.eps)
    if mode == "relative_distance_rolling_rank":
        return rolling_rank_scale(rel_std, cfg.window, cfg.min_periods, cfg.use_past_only)

    raise ValueError(f"Unsupported overlay std mode: {mode}")




__all__ = [
    "OverlayNormalizationConfig",
    "rolling_zscore",
    "linear_wma",
    "rolling_wma_zscore",
    "rolling_minmax_scale",
    "rolling_rank_scale",
    "signed_log1p",
    "stationary_transform",
    "lag_features",
    "clip_by_train_quantile",
    "apply_rolling_method",
    "overlay_value_atr_gap",
    "overlay_pct_distance",
    "overlay_velocity",
    "overlay_acceleration",
    "overlay_std",
    "normalize_overlay_indicator",
    "normalize_overlay_indicators",
]
