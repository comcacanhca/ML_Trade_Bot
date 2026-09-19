from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_fs03 import run as evaluate_run
from fs03_common import parse_thresholds
from train_optuna_fs03 import run as train_run


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end CI/CD Optuna LightGBM pipeline for fs03.")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--thresholds", default="0.56,0.57,0.58")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--skip-simulate", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    train_summary = train_run(
        n_trials=args.n_trials,
        seed=args.seed,
        force_dataset=args.force_dataset,
        max_train_rows=args.max_train_rows,
        enable_mlflow=args.enable_mlflow,
        output_root=args.output_root,
    )
    evaluate_run(
        model_bundle=Path(train_summary["model_path"]),
        thresholds=parse_thresholds(args.thresholds),
        force_dataset=False,
        skip_simulate=args.skip_simulate,
        enable_mlflow=args.enable_mlflow,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
