from __future__ import annotations

import json
from pathlib import Path

from research_h4edge_feature_transforms_optuna import run


TARGET_TEST_WR = 58.8773
TARGET_MIN_DEALS = 5000
SUMMARY_PATH = Path(__file__).resolve().parent / "outputs" / "transform_wave2_until_best_summary.json"


CONFIGS = [
    {"name": "products_top6_t80", "variants": ["products"], "n_trials": 80, "top_n": 6, "seed": 92201},
    {"name": "products_top8_t80", "variants": ["products"], "n_trials": 80, "top_n": 8, "seed": 92202},
    {"name": "products_top10_t80_s2", "variants": ["products"], "n_trials": 80, "top_n": 10, "seed": 92203},
    {"name": "products_top12_t80", "variants": ["products"], "n_trials": 80, "top_n": 12, "seed": 92204},
    {"name": "products_top14_t80", "variants": ["products"], "n_trials": 80, "top_n": 14, "seed": 92205},
    {"name": "products_top16_t80", "variants": ["products"], "n_trials": 80, "top_n": 16, "seed": 92206},
    {"name": "diff_ratio_top8_t80", "variants": ["diff_ratio"], "n_trials": 80, "top_n": 8, "seed": 92207},
    {"name": "diff_ratio_top12_t80", "variants": ["diff_ratio"], "n_trials": 80, "top_n": 12, "seed": 92208},
]


def main() -> None:
    if SUMMARY_PATH.exists():
        results = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        results = []
    completed = {row["config"] for row in results}
    for cfg in CONFIGS:
        if cfg["name"] in completed:
            print({"phase": "config_skip_completed", "name": cfg["name"]}, flush=True)
            continue
        print({"phase": "config_start", **cfg}, flush=True)
        result = run(
            variants=cfg["variants"],
            n_trials=cfg["n_trials"],
            max_train_rows=1000000,
            top_n=cfg["top_n"],
            max_mi_rows=50000,
            seed=cfg["seed"],
            shap_rows=80,
            enable_mlflow=True,
            min_valid_resolved=3000,
            target_valid_resolved=4200,
            max_valid_resolved=9000,
        )
        best = result["best"]
        row = {
            "config": cfg["name"],
            "run_id": result["run_id"],
            "out_dir": result["out_dir"],
            "bundle_path": result["bundle_path"],
            "variant": best["variant"],
            "valid_auc": best["valid_auc"],
            "valid_wr": best["valid_wr"],
            "valid_resolved": best["valid_resolved"],
            "test_auc": best["test_auc"],
            "test_wr": best["test_wr"],
            "test_resolved": best["test_resolved"],
            "test_min_year_deals": best["test_min_year_deals"],
            "threshold": best["selected_threshold"],
        }
        results.append(row)
        print({"phase": "config_done", **row}, flush=True)
        SUMMARY_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if float(best["test_wr"]) > TARGET_TEST_WR and int(best["test_resolved"]) >= TARGET_MIN_DEALS:
            print({"phase": "target_beaten", "target_test_wr": TARGET_TEST_WR, **row}, flush=True)
            return
    print(
        {
            "phase": "target_not_beaten",
            "target_test_wr": TARGET_TEST_WR,
            "best": max(results, key=lambda r: r["test_wr"]) if results else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
