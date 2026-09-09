from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT
WORK_DIR = REPO_ROOT / "research" / "rf_mlflow"
VENDOR_SCJ_DIR = REPO_ROOT / "vendor" / "scj"
if str(VENDOR_SCJ_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_SCJ_DIR))

DATA_DIR = Path(os.getenv("ML_TRADE_DATA_DIR", str(REPO_ROOT / "data" / "raw" / "1M")))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")


@dataclass(frozen=True)
class TradeConfig:
    rr: float = 1.0
    r1: float = 6.0
    expired: int = 5
    max_running_deals: int = 12
    min_deal_gap: int = 0
    commission: int = 16
    start_candle: int = 300
    buy_only: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)
    valid_years: tuple[int, ...] = (2023,)
    test_years: tuple[int, ...] = (2024, 2025, 2026)


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 300
    max_depth: int | None = 10
    min_samples_leaf: int = 500
    min_samples_split: int = 1000
    max_features: str = "sqrt"
    max_samples: float = 0.80
    class_weight: str = "balanced_subsample"
    random_state: int = 271828
    n_jobs: int = -1


@dataclass(frozen=True)
class PipelineConfig:
    experiment_name: str = "RF_MLFlow_XAUUSD_M1_Buy_WinLose"
    artifacts_dir: Path = WORK_DIR / "artifacts"
    cache_dir: Path = WORK_DIR / "cache"
    outputs_dir: Path = WORK_DIR / "outputs"
    model_dir: Path = WORK_DIR / "models"
    split: SplitConfig = field(default_factory=SplitConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    # Market entry is filled at next open; MoPhongDeals does not expire an
    # already-running market deal. Use a long M1 horizon for supervised labels.
    label_max_forward: int = 1440
    threshold_grid: tuple[float, ...] = (0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60
    )


CFG = PipelineConfig()


def ensure_dirs() -> None:
    for path in [CFG.artifacts_dir, CFG.cache_dir, CFG.outputs_dir, CFG.model_dir]:
        path.mkdir(parents=True, exist_ok=True)
