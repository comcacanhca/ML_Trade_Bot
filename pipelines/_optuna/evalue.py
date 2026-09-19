from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_fs03 import run
from fs03_common import parse_thresholds

__all__ = ["run"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backward-compatible wrapper for FS03 Optuna evaluation.")
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.56,0.57,0.58")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--skip-simulate", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run(args.model_bundle, parse_thresholds(args.thresholds), args.force_dataset, args.skip_simulate, args.enable_mlflow, args.output_root)
