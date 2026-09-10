from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SCJ_DIR = REPO_ROOT / "vendor" / "scj"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VENDOR_SCJ_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_SCJ_DIR))

from data_handler import CLOSE, DATE, HIGH, LOW, OPEN, VOLUME  # noqa: E402
from data_handler.download_data_mt5 import MT5_AVAILABLE, mt5  # noqa: E402


DEFAULT_KEY_FILE = REPO_ROOT / "t-bounty-508213-u7-f4627e44cf81.json"
DEFAULT_DATA_DIR = Path(os.getenv("ML_TRADE_DATA_DIR", str(REPO_ROOT / "data_handler" / "raw" / "1M")))
REQUIRED_COLUMNS = [DATE, OPEN, LOW, HIGH, CLOSE]


def _require_mt5() -> None:
    if not MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 package is not installed in the Python env running this script. "
            "Install it in D:\\ML_Trade_Bot\\.venv with: pip install MetaTrader5"
        )


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    if "time" in out.columns and DATE not in out.columns:
        out[DATE] = pd.to_datetime(out["time"], unit="s", errors="coerce")
    else:
        out[DATE] = pd.to_datetime(out[DATE], errors="coerce")

    rename_map = {
        "tick_volume": VOLUME,
        "real_volume": "real_volume",
    }
    out = out.rename(columns=rename_map)

    keep = [col for col in [DATE, OPEN, LOW, HIGH, CLOSE, VOLUME] if col in out.columns]
    out = out[keep].dropna(subset=REQUIRED_COLUMNS).copy()
    out = out.sort_values(DATE).drop_duplicates(subset=[DATE], keep="last").reset_index(drop=True)
    out[DATE] = pd.DatetimeIndex(out[DATE]).strftime("%Y-%m-%d %H:%M:%S")
    return out


def _validate_price_frame(frame: pd.DataFrame, *, source: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")

    dates = pd.to_datetime(frame[DATE], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{source}: invalid dates rows={int(dates.isna().sum())}")

    numeric = frame[[OPEN, LOW, HIGH, CLOSE]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.isna().sum().to_dict()
        raise ValueError(f"{source}: invalid OHLC numeric values: {bad}")

    if (numeric[HIGH] < numeric[LOW]).any():
        raise ValueError(f"{source}: found rows where high < low")


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[DATE, OPEN, LOW, HIGH, CLOSE, VOLUME])
    frame = pd.read_csv(path)
    frame = _normalize_price_frame(frame)
    _validate_price_frame(frame, source=str(path))
    return frame


def _infer_download_window(data_dir: Path, symbol_file_prefix: str, lookback_days: int) -> tuple[datetime, datetime]:
    now = datetime.now()
    current_file = data_dir / f"{symbol_file_prefix}_{now.year}.csv"
    if current_file.exists():
        existing = _read_existing(current_file)
        if not existing.empty:
            last_ts = pd.to_datetime(existing[DATE]).max().to_pydatetime()
            return last_ts - timedelta(days=1), now
    return now - timedelta(days=lookback_days), now


def _download_mt5(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    _require_mt5()
    if not mt5.initialize():
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"MT5 initialize() failed: {error}")

    try:
        selected = mt5.symbol_select(symbol, True)
        if not selected:
            raise RuntimeError(f"MT5 symbol_select({symbol!r}) failed: {mt5.last_error()}")
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    finally:
        mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no M1 rates for {symbol} from {start} to {end}")

    frame = _normalize_price_frame(pd.DataFrame(rates))
    _validate_price_frame(frame, source="mt5_download")
    return frame


def _merge_and_write_year(data_dir: Path, symbol_file_prefix: str, year: int, new_rows: pd.DataFrame) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{symbol_file_prefix}_{year}.csv"
    existing = _read_existing(out_path)
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = _normalize_price_frame(merged)
    _validate_price_frame(merged, source=str(out_path))
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print({"phase": "saved_raw_year", "year": year, "path": str(out_path), "rows": len(merged)}, flush=True)
    return out_path


def _upload_to_gcs(local_path: Path, bucket_name: str, gcs_prefix: str, key_file: Path) -> str:
    if not key_file.exists():
        raise FileNotFoundError(f"GCS service account key not found: {key_file}")

    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is not installed. Install it in D:\\ML_Trade_Bot\\.venv with: "
            "pip install google-cloud-storage"
        ) from exc

    client = storage.Client.from_service_account_json(str(key_file))
    bucket = client.bucket(bucket_name)
    prefix = gcs_prefix.strip("/")
    blob_name = f"{prefix}/{local_path.name}" if prefix else local_path.name
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type="text/csv")
    uri = f"gs://{bucket_name}/{blob_name}"
    print({"phase": "uploaded_gcs", "local_path": str(local_path), "gcs_uri": uri}, flush=True)
    return uri


def run(
    *,
    symbol: str,
    symbol_file_prefix: str,
    data_dir: Path,
    bucket_name: str,
    gcs_prefix: str,
    key_file: Path,
    lookback_days: int,
    from_date: str | None,
    to_date: str | None,
) -> dict:
    start, end = _infer_download_window(data_dir, symbol_file_prefix, lookback_days)
    if from_date:
        start = datetime.strptime(from_date, "%Y-%m-%d")
    if to_date:
        end = datetime.strptime(to_date, "%Y-%m-%d")

    print(
        {
            "phase": "download_start",
            "symbol": symbol,
            "data_dir": str(data_dir),
            "from": start.isoformat(sep=" "),
            "to": end.isoformat(sep=" "),
            "bucket": bucket_name,
            "gcs_prefix": gcs_prefix,
        },
        flush=True,
    )

    downloaded = _download_mt5(symbol, start, end)
    downloaded["_year"] = pd.to_datetime(downloaded[DATE]).dt.year.astype(int)

    written_files: list[Path] = []
    uploaded_uris: list[str] = []
    for year, part in downloaded.groupby("_year", sort=True):
        part = part.drop(columns=["_year"]).copy()
        written = _merge_and_write_year(data_dir, symbol_file_prefix, int(year), part)
        written_files.append(written)
        uploaded_uris.append(_upload_to_gcs(written, bucket_name, gcs_prefix, key_file))

    result = {
        "symbol": symbol,
        "downloaded_rows": int(len(downloaded)),
        "written_files": [str(path) for path in written_files],
        "uploaded_uris": uploaded_uris,
    }
    print({"phase": "done", **result}, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XAUUSD M1 raw data from MT5, merge yearly CSV, upload to GCS.")
    parser.add_argument("--symbol", default=os.getenv("MT5_SYMBOL", "XAUUSD"))
    parser.add_argument("--symbol-file-prefix", default=os.getenv("MT5_SYMBOL_FILE_PREFIX", "XAUUSDm"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gcs-bucket", default=os.getenv("GCS_BUCKET"))
    parser.add_argument("--gcs-prefix", default=os.getenv("GCS_PREFIX", "raw/1M"))
    parser.add_argument("--gcs-key-file", type=Path, default=Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", str(DEFAULT_KEY_FILE))))
    parser.add_argument("--lookback-days", type=int, default=int(os.getenv("MT5_LOOKBACK_DAYS", "7")))
    parser.add_argument("--from-date", default=None, help="Optional YYYY-MM-DD override.")
    parser.add_argument("--to-date", default=None, help="Optional YYYY-MM-DD override.")
    args = parser.parse_args()

    if not args.gcs_bucket:
        raise RuntimeError("Missing GCS bucket. Set env GCS_BUCKET or pass --gcs-bucket.")

    run(
        symbol=args.symbol,
        symbol_file_prefix=args.symbol_file_prefix,
        data_dir=args.data_dir,
        bucket_name=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        key_file=args.gcs_key_file,
        lookback_days=args.lookback_days,
        from_date=args.from_date,
        to_date=args.to_date,
    )


if __name__ == "__main__":
    main()
