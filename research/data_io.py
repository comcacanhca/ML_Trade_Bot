from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from research.config import DATA_DIR, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_handler import CLOSE, DATE, HIGH, LOW, OPEN, VOLUME  # noqa: E402
from data_handler.download_data_mt5 import get_data_from_csv  # noqa: E402


BASE_COLUMNS = [DATE, OPEN, LOW, HIGH, CLOSE]


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item.strip()) for item in part.split("-", 1)]
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


def year_path(year: int) -> Path:
    return DATA_DIR / f"XAUUSDm_{year}.csv"


def load_year(year: int) -> pd.DataFrame:
    print('Loading year:', year)
    path = year_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    data = get_data_from_csv(str(path))
    if data.index.name is not None or DATE in data.index.names:
        data = data.reset_index(drop=True)

    keep = [col for col in [DATE, OPEN, LOW, HIGH, CLOSE, VOLUME] if col in data.columns]
    data = data[keep].dropna(subset=BASE_COLUMNS).copy()
    data[DATE] = pd.to_datetime(data[DATE], errors="coerce")
    data = data.dropna(subset=[DATE]).sort_values(DATE).reset_index(drop=True)
    return data


def load_years(years: list[int] | tuple[int, ...]) -> pd.DataFrame:
    frames = [load_year(year) for year in years]
    if not frames:
        raise ValueError("No years requested")
    return pd.concat(frames, ignore_index=True).sort_values(DATE).reset_index(drop=True)
