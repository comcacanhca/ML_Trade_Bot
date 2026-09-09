# %%
"""
Notebook-style Python file for:

    total features + rolling_zscore + RFECV + LightGBM

Run từng cell trong PyCharm / VS Code bằng các marker `# %%`.
File này không phải .ipynb JSON và không gọi shell command.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# %%
# 1. Setup path/config

PROJECT = Path(r"D:\SCJ999\srateries\RandomForest\RF_MLFlow")
CACHE_FAMILY = "initial_non_bb_candidates"

import os
import sys

os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

print("cwd:", Path.cwd())


# %%
# 2. Import pipeline/dependencies

import lightgbm as lgb
import sklearn

from config import CFG
from research_random50_initial_features import build_cache
from research_total_features_rfecv_lgbm import run as run_total_rfecv_lgbm

print("lightgbm:", lgb.__version__)
print("sklearn:", sklearn.__version__)
print("experiment:", CFG.experiment_name)
print("train years:", CFG.split.train_years)
print("valid years:", CFG.split.valid_years)
print("test years :", CFG.split.test_years)


# %%
# 3. Check cache feature count

cols_path = PROJECT / "cache" / CACHE_FAMILY / "feature_columns.json"
cols = json.loads(cols_path.read_text(encoding="utf-8"))

feature_summary = {
    "cache_family": CACHE_FAMILY,
    "total_features": len(cols),
    "overlay_features": sum("_ov_" in c for c in cols),
    "featuretools_features": sum(c.startswith("ft_") for c in cols),
    "time_features": sum(c in {"tod_sin", "tod_cos", "dow_sin", "dow_cos"} for c in cols),
    "signal_features": sum(c.startswith("sig_") or c == "buy_signal" for c in cols),
}

feature_summary


# %%
# 4. Group feature families

feature_df = pd.DataFrame({"feature": cols})
feature_df["family"] = np.select(
    [
        feature_df["feature"].str.startswith(("ema", "sma")) & feature_df["feature"].str.contains("_ov_", regex=False),
        feature_df["feature"].str.startswith("bb"),
        feature_df["feature"].str.startswith(("keltner", "donchian")),
        feature_df["feature"].str.startswith("vwap"),
        feature_df["feature"].str.startswith(("ichimoku", "parabolic_sar", "supertrend")),
        feature_df["feature"].str.startswith("h4_"),
        feature_df["feature"].str.startswith("h1_"),
        feature_df["feature"].str.startswith("m15_"),
        feature_df["feature"].str.startswith("m5_"),
        feature_df["feature"].str.startswith(("tod_", "dow_", "session_")),
        feature_df["feature"].str.startswith("sig_") | (feature_df["feature"] == "buy_signal"),
    ],
    [
        "ma_overlay",
        "bb_overlay",
        "channel_overlay",
        "vwap_overlay",
        "trendline_overlay",
        "h4",
        "h1",
        "m15",
        "m5",
        "time_session",
        "signal",
    ],
    default="other",
)

feature_family_count = feature_df.groupby("family").size().sort_values(ascending=False)
feature_family_count


# %%
# 5. Check parquet sample

sample_path = PROJECT / "cache" / CACHE_FAMILY / "candidates_2018.parquet"
sample = pd.read_parquet(sample_path)

META = {"dates", "candle_index", "entry_index", "entry_price", "year", "label"}
sample_features = [c for c in sample.columns if c not in META]

sample_summary = {
    "rows_2018": len(sample),
    "parquet_cols": len(sample.columns),
    "parquet_features": len(sample_features),
    "overlay_features": sum("_ov_" in c for c in sample_features),
    "label_rate": float(sample["label"].mean()) if "label" in sample else None,
}

sample_summary


# %%
# 6. Optional rebuild cache
#
# Chỉ chạy cell này nếu cache chưa có overlay feature hoặc cần rebuild lại.
# Có thể tốn RAM/thời gian.

RUN_REBUILD_CACHE = False

if RUN_REBUILD_CACHE:
    build_cache(True)


# %%
# 7. Smoke test config
#
# Dùng để kiểm tra pipeline chạy được. Không dùng kết quả này để kết luận model.

RUN_ID = int(time.time())

smoke_config = dict(
    run_id=RUN_ID,
    cache_family=CACHE_FAMILY,
    force_cache=False,
    max_train_rows=3000,
    rfecv_rows=1500,
    preselect_top_n=120,
    preselect_corr_threshold=0.98,
    preselect_corr_rows=1500,
    preselect_mutual_info_rows=1500,
    preselect_n_estimators=30,
    min_features_to_select=20,
    rfecv_step=300,
    cv_splits=2,
    seed=860906,
    n_estimators_rfecv=20,
    n_estimators_final=40,
    n_jobs=2,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    protect_time_features=True,
    protect_passthrough=False,
    enable_mlflow=False,
    save_predictions=False,
    log_diagnostics=True,
    shap_rows=0,
)

smoke_config


# %%
# 8. Run smoke test

RUN_SMOKE = False

if RUN_SMOKE:
    smoke_summary = run_total_rfecv_lgbm(**smoke_config)
    print(json.dumps(smoke_summary, ensure_ascii=False, indent=2))


# %%
# 9. RAM-safe research config

RUN_ID = int(time.time())

ram_safe_config = dict(
    run_id=RUN_ID,
    cache_family=CACHE_FAMILY,
    force_cache=False,
    max_train_rows=60000,
    rfecv_rows=20000,
    preselect_top_n=300,
    preselect_corr_threshold=0.98,
    preselect_corr_rows=30000,
    preselect_mutual_info_rows=20000,
    preselect_n_estimators=250,
    min_features_to_select=40,
    rfecv_step=100,
    cv_splits=3,
    seed=860906,
    n_estimators_rfecv=150,
    n_estimators_final=700,
    n_jobs=2,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    protect_time_features=True,
    protect_passthrough=False,
    enable_mlflow=True,
    save_predictions=False,
    log_diagnostics=True,
    shap_rows=0,
)

ram_safe_config


# %%
# 10. Run RAM-safe research

RUN_RAM_SAFE = False

if RUN_RAM_SAFE:
    ram_safe_summary = run_total_rfecv_lgbm(**ram_safe_config)
    print(json.dumps(ram_safe_summary, ensure_ascii=False, indent=2))


# %%
# 11. Standard research config

RUN_ID = int(time.time())

standard_config = dict(
    run_id=RUN_ID,
    cache_family=CACHE_FAMILY,
    force_cache=False,
    max_train_rows=120000,
    rfecv_rows=50000,
    preselect_top_n=300,
    preselect_corr_threshold=0.98,
    preselect_corr_rows=30000,
    preselect_mutual_info_rows=20000,
    preselect_n_estimators=250,
    min_features_to_select=60,
    rfecv_step=50,
    cv_splits=3,
    seed=860906,
    n_estimators_rfecv=250,
    n_estimators_final=1200,
    n_jobs=4,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    protect_time_features=True,
    protect_passthrough=False,
    enable_mlflow=True,
    save_predictions=False,
    log_diagnostics=True,
    shap_rows=0,
)

standard_config


# %%
# 12. Run standard research

RUN_STANDARD = False

if RUN_STANDARD:
    standard_summary = run_total_rfecv_lgbm(**standard_config)
    print(json.dumps(standard_summary, ensure_ascii=False, indent=2))


# %%
# 13. Load result by RUN_ID
#
# Set RESULT_RUN_ID bằng run_id bạn muốn xem.

RESULT_RUN_ID = RUN_ID

out_dir = PROJECT / "outputs" / f"total_features_rfecv_lgbm_{RESULT_RUN_ID}"
summary_path = out_dir / "summary.json"
metrics_path = out_dir / "metrics.json"

print("out_dir:", out_dir)
print("summary exists:", summary_path.exists())
print("metrics exists:", metrics_path.exists())


# %%
# 14. Read summary/metrics

if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    display(summary)

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    display(metrics)


# %%
# 15. RFECV ranking

rank_path = out_dir / "rfecv_feature_ranking.csv"

if rank_path.exists():
    rank = pd.read_csv(rank_path)
    display(rank.head(50))
    display(
        rank.groupby(["selected", "ranking"])
        .size()
        .reset_index(name="n")
        .sort_values(["selected", "ranking"], ascending=[False, True])
        .head(50)
    )


# %%
# 16. Selected features

selected_path = out_dir / "selected_features_with_protected.json"
if not selected_path.exists():
    selected_path = out_dir / "selected_features.json"

if selected_path.exists():
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected_features"]
    selected_df = pd.DataFrame({"feature": selected})
    selected_df["is_overlay"] = selected_df["feature"].str.contains("_ov_", regex=False)
    print("selected count:", len(selected))
    print("selected overlay:", int(selected_df["is_overlay"].sum()))
    display(selected_df.head(100))


# %%
# 17. LightGBM feature importance

imp_path = out_dir / "model" / "lgbm_feature_importance.csv"

if imp_path.exists():
    imp = pd.read_csv(imp_path)
    display(imp.head(80))


# %%
# 18. Importance by feature family

if imp_path.exists():
    imp2 = imp.copy()
    imp2["family"] = np.select(
        [
            imp2["feature"].str.contains("_ov_", regex=False),
            imp2["feature"].str.startswith("h4_"),
            imp2["feature"].str.startswith("h1_"),
            imp2["feature"].str.startswith("m15_"),
            imp2["feature"].str.startswith("m5_"),
            imp2["feature"].str.startswith("sig_"),
            imp2["feature"].str.startswith(("tod_", "dow_", "session_")),
        ],
        ["overlay", "h4", "h1", "m15", "m5", "signal", "time_session"],
        default="other",
    )
    display(
        imp2.groupby("family")
        .agg(n=("feature", "count"), importance_sum=("importance", "sum"), importance_mean=("importance", "mean"))
        .sort_values("importance_sum", ascending=False)
    )


# %%
# 19. Threshold tables

valid_curve_path = out_dir / "model" / "threshold_valid.csv"
test_curve_path = out_dir / "model" / "threshold_test.csv"

if valid_curve_path.exists():
    valid_curve = pd.read_csv(valid_curve_path)
    display(valid_curve.sort_values("winrate", ascending=False).head(30))

if test_curve_path.exists():
    test_curve = pd.read_csv(test_curve_path)
    display(test_curve.sort_values("winrate", ascending=False).head(30))


# %%
# 20. Yearly/monthly test diagnostics

yearly_path = out_dir / "model" / "total_features_rfecv_lgbm_yearly_test.csv"
monthly_path = out_dir / "model" / "total_features_rfecv_lgbm_monthly_test.csv"

if yearly_path.exists():
    yearly = pd.read_csv(yearly_path)
    display(yearly)

if monthly_path.exists():
    monthly = pd.read_csv(monthly_path)
    display(monthly.head(60))


# %%
# 21. Show chart artifacts

from IPython.display import Image, display

chart_files = [
    out_dir / "model" / "lgbm_feature_importance_top40.png",
    out_dir / "model" / "total_features_rfecv_lgbm_roc_valid.png",
    out_dir / "model" / "total_features_rfecv_lgbm_roc_test.png",
    out_dir / "model" / "total_features_rfecv_lgbm_pr_valid.png",
    out_dir / "model" / "total_features_rfecv_lgbm_pr_test.png",
    out_dir / "model" / "total_features_rfecv_lgbm_calibration_valid.png",
    out_dir / "model" / "total_features_rfecv_lgbm_calibration_test.png",
    out_dir / "model" / "total_features_rfecv_lgbm_prob_distribution_valid_test.png",
    out_dir / "model" / "total_features_rfecv_lgbm_confusion_test.png",
    out_dir / "model" / "total_features_rfecv_lgbm_yearly_test_selected_threshold.png",
    out_dir / "model" / "total_features_rfecv_lgbm_monthly_test_selected_threshold.png",
]

for path in chart_files:
    if path.exists():
        print(path.name)
        display(Image(filename=str(path)))


# %%
# 22. Optional: refit with SHAP
#
# Chỉ chạy sau khi bản no-SHAP ổn. SHAP có thể tốn RAM.

RUN_SHAP_REFIT = False

shap_config = dict(standard_config)
shap_config["run_id"] = int(time.time())
shap_config["shap_rows"] = 200

if RUN_SHAP_REFIT:
    shap_summary = run_total_rfecv_lgbm(**shap_config)
    print(json.dumps(shap_summary, ensure_ascii=False, indent=2))
