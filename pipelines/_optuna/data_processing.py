from __future__ import annotations

import argparse
import json

from fs03_common import FEATURE_STORE_META_PATH, FEATURE_STORE_PATH, load_fs03_splits, materialize_feature_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/cache FS03 dataset for downstream Optuna pipeline steps.")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    frame = materialize_feature_store(force=args.force_dataset)
    data = load_fs03_splits(force_dataset=False, max_train_rows=args.max_train_rows)
    print(
        json.dumps(
            {
                "feature_store_path": str(FEATURE_STORE_PATH),
                "feature_store_meta_path": str(FEATURE_STORE_META_PATH),
                "feature_store_rows": int(len(frame)),
                "feature_count": len(data["feature_columns"]),
                "train_samples": int(len(data["y_train"])),
                "valid_samples": int(len(data["y_valid"])),
                "test_samples": int(len(data["y_test"])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
