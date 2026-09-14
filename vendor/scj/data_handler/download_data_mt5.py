import datetime
import math

from . import *

import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

    class _MT5Stub:
        TIMEFRAME_M1 = 'M1'
        TIMEFRAME_M5 = 'M5'
        TIMEFRAME_M15 = 'M15'
        TIMEFRAME_H1 = 'H1'
        TIMEFRAME_H4 = 'H4'
        TIMEFRAME_D1 = 'D1'
        TIMEFRAME_W1 = 'W1'

        def __getattr__(self, name):
            raise RuntimeError("MetaTrader5 package is required for live MT5 operations.")

    mt5 = _MT5Stub()


def _ensure_mt5_available():
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package is not installed. Live MT5 download functions are unavailable.")


def parse_and_format_date(date_str):
    """Hàm con để thử các định dạng ngày tháng khác nhau và chuyển đổi."""
    formats_to_try = [
        '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
        '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
        '%d %b %Y', '%d %b %Y %H:%M:%S', '%b %d, %Y', '%b %d, %Y %H:%M:%S',
        '%Y%m%d', '%Y%m%d %H%M%S',
        '%Y-%m-%d %H:%M:%S%z', '%Y/%m/%d %H:%M:%S%z',
    ]
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def _normalize_datetime_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().any():
        parsed = parsed.fillna(series.apply(parse_and_format_date))
    return parsed


def _prepare_price_frame(data: pd.DataFrame, detect_vol=True) -> pd.DataFrame:
    frame = data.copy()
    if DATE in frame.columns:
        normalized_dates = _normalize_datetime_series(frame[DATE])
        frame[DATE] = normalized_dates
        frame = frame.dropna(subset=[DATE])
        frame.index = pd.DatetimeIndex(frame[DATE])
        frame[DATE] = pd.DatetimeIndex(frame.index).strftime('%Y-%m-%d %H:%M:%S')

    return frame


def _rates_to_frame(rates, detect_vol=False):
    rates_frame = pd.DataFrame(rates)
    if rates_frame.empty:
        return rates_frame
    rates_frame.index = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame = rates_frame.rename(columns={'time': DATE})
    rates_frame[DATE] = pd.DatetimeIndex(rates_frame.index).strftime('%Y-%m-%d %H:%M:%S')

    return rates_frame


def remove_weekends(data: pd.DataFrame):
    cleaned = data.copy()
    normalized_dates = _normalize_datetime_series(cleaned[DATE])
    cleaned[DATE] = normalized_dates
    weekday_mask = pd.DatetimeIndex(normalized_dates).dayofweek < 5
    cleaned = cleaned.loc[weekday_mask].copy()
    cleaned.index = pd.DatetimeIndex(cleaned[DATE])
    cleaned[DATE] = pd.DatetimeIndex(cleaned.index).strftime('%Y-%m-%d %H:%M:%S')
    return cleaned


def download_data(tick="EURUSD", time_frame=mt5.TIMEFRAME_M15, quanlity=10000, detect_vol=True):
    _ensure_mt5_available()
    if not mt5.initialize():
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"initialize() failed: {error}")
    try:
        rates = mt5.copy_rates_from_pos(tick, time_frame, 0, quanlity)
    finally:
        mt5.shutdown()
    return _rates_to_frame(rates, detect_vol=detect_vol)


def convert_tf(df, time_frame):
    converted = _prepare_price_frame(df, detect_vol=False)
    if converted.empty:
        return converted
    converted.index = pd.DatetimeIndex(pd.to_datetime(converted[DATE]))
    converted = converted.loc[:, ~converted.columns.duplicated()].copy()

    def _first_valid(series):
        values = series.dropna()
        return values.iloc[0] if len(values) else pd.NA

    def _last_valid(series):
        values = series.dropna()
        return values.iloc[-1] if len(values) else pd.NA

    ohlc_dict = {
        OPEN: _first_valid,
        HIGH: 'max',
        LOW: 'min',
        CLOSE: _last_valid,
    }
    for volume_col in ['tick_volume', 'real_volume', VOLUME]:
        if volume_col in converted.columns and volume_col not in ohlc_dict:
            ohlc_dict[volume_col] = 'sum'
    freq_map = {
        'M5': '5min',
        mt5.TIMEFRAME_M5: '5min',
        'M15': '15min',
        mt5.TIMEFRAME_M15: '15min',
        'H1': '1h',
        mt5.TIMEFRAME_H1: '1h',
        'H4': '4h',
        mt5.TIMEFRAME_H4: '4h',
        'D': '1D',
        mt5.TIMEFRAME_D1: '1D',
        'W1': '1W',
        mt5.TIMEFRAME_W1: '1W',
    }
    freq = freq_map.get(time_frame, '1min')
    available_agg = {col: agg for col, agg in ohlc_dict.items() if col in converted.columns}
    new_df = converted.resample(freq).agg(available_agg).dropna()
    new_df[DATE] = pd.DatetimeIndex(new_df.index).strftime('%Y-%m-%d %H:%M:%S')
    return new_df


def download_data_from_to(tick="EURUSD", time_frame=mt5.TIMEFRAME_M15, from_=None, to_=None, detect_vol=True):
    _ensure_mt5_available()
    from_ = datetime.datetime.strptime(from_, "%Y-%m-%d")
    to_ = datetime.datetime.strptime(to_, "%Y-%m-%d")
    if not mt5.initialize():
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"initialize() failed: {error}")
    try:
        rates = mt5.copy_rates_range(tick, time_frame, from_, to_)
    finally:
        mt5.shutdown()
    return _rates_to_frame(rates)


def get_data_from_csv(file):
    data = pd.read_csv(file)
    return _prepare_price_frame(data)


def get_preH1Candle(df):
    """
    return: OLHC of pre H1
    """
    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index(DATE)
    ohlc_dict = {OPEN: 'first', HIGH: 'max', LOW: 'min', CLOSE: 'last'}
    h1df = df.resample('1h').apply(ohlc_dict).dropna()
    h1df[DATE] = h1df.index.strftime('%Y-%m-%d %H:%M:%S')
    h1df['tick_volume'] = [0 for _ in range(len(h1df))]
    return h1df


def getPreM15(df):
    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index(DATE)
    ohlc_dict = {OPEN: 'first', HIGH: 'max', LOW: 'min', CLOSE: 'last'}
    m15_df = df.resample('15min').apply(ohlc_dict).dropna()
    m15_df[DATE] = m15_df.index.strftime('%Y-%m-%d %H:%M:%S')
    m15_df['tick_volume'] = [0 for _ in range(len(m15_df))]
    return m15_df


def get_data_from_to(csv_file, start_date, end_date):
    if isinstance(csv_file, str):
        df = pd.read_csv(csv_file)
    elif isinstance(csv_file, pd.DataFrame):
        df = csv_file.copy()
    else:
        raise TypeError('Wrong format of csv file not in DataFrame or path')

    df[DATE] = _normalize_datetime_series(df[DATE])
    date_series = df[DATE]
    series_timezone = getattr(getattr(date_series, 'dt', None), 'tz', None)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if series_timezone is not None:
        if getattr(start_date, 'tzinfo', None) is None:
            start_date = start_date.tz_localize(series_timezone)
        else:
            start_date = start_date.tz_convert(series_timezone)
        if getattr(end_date, 'tzinfo', None) is None:
            end_date = end_date.tz_localize(series_timezone)
        else:
            end_date = end_date.tz_convert(series_timezone)
    filtered_df = df[(date_series >= start_date) & (date_series <= end_date)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return _prepare_price_frame(filtered_df)


def chiaData(csv_file, quantity):
    if isinstance(csv_file, str):
        df = pd.read_csv(csv_file)
    elif isinstance(csv_file, pd.DataFrame):
        df = csv_file.copy()
    else:
        raise TypeError('Wrong format of csv file not in DataFrame or path')
    if quantity <= 0:
        raise ValueError('quantity must be > 0')
    chunk_size = math.ceil(len(df) / quantity) if len(df) else 0
    if chunk_size == 0:
        return []
    return [df.iloc[i:i + chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]


#print(download_data_from_to(from_='2023-01-02', to_='2023-05-03'))
# start_date = "2015-02-03"
# end_date = "2022-03-03"
# filtered_data = get_data_from_to(r'E:\filter_tradeBot\data_handler\output\2022_new.csv', start_date, end_date)
# print(filtered_data)
# data_handler = pd.read_csv(r'E:\filter_tradeBot\data_handler\output\fullyear.csv')
# data_handler[date] = pd.to_datetime(data_handler[date], format="mixed")
# data_handler.to_csv(r'E:\filter_tradeBot\data_handler\output\fullyear.csv', index=False)
