from __future__ import annotations

import argparse
from pathlib import Path

from train_optuna_fs03 import run

__all__ = ["run"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backward-compatible wrapper for FS03 Optuna LightGBM training.")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run(args.n_trials, args.seed, args.force_dataset, args.max_train_rows, args.enable_mlflow, args.output_root)
