# Migration inventory

Nguồn migrate:

`D:\SCJ999\srateries\RandomForest\RF_MLFlow`

## Đã copy

- Core utilities:
  - `config.py`
  - `data_io.py`
  - `features.py`
  - `labels.py`
  - `plots.py`
  - `mophong_adapter.py`
- Research scripts:
  - H4/time edge MLOps
  - H4 edge feature transforms + Optuna
  - rolling zscore H4/H1
  - random50 initial features
  - feature selection / RFECV / LightGBM variants
  - historical FS03/FS14 research scripts
- Runbooks/notebooks:
  - rolling zscore runbook
  - total features RFECV LGBM runbooks/notebooks
  - random50 MLOps notebook
- Backtest adapter folder:
  - `research/rf_mlflow/backtest_by_Mophong`
- Internal vendor modules:
  - `vendor/scj/method/OverlayIndicatorsNormalization.py`
  - `vendor/scj/method/MoPhongDeals.py`
  - `vendor/scj/data/download_data_mt5.py`
  - `vendor/scj/data/__init__.py`

## Không copy vào git

- `cache/`
- `models/`
- `outputs/`
- `artifacts/`
- `mlruns/`
- `*.log`
- `*.err.log`
- `mlflow.db`

Các nhóm trên phải được quản lý bằng DVC hoặc object storage/MLflow registry.

## Baseline hiện tại

Best known candidate từ nghiên cứu cũ:

- run id: `1788956979`
- run name: `h4edge_feature_transforms_optuna_1788956979`
- variant: `products`
- top_n: `10`
- trials: `50`
- test WR: `58.8773%`
- test resolved: `6449`
- test AUC: `0.52525`
- threshold: `0.53`

## Quy ước MLOps mới

- Không commit raw data/model/output nặng vào git.
- DVC quản lý `data/`, `models/`, `outputs/`.
- MLflow quản lý experiment metrics/artifacts.
- GitHub Actions chỉ chạy lint/test/sanity; không train full.
- Kubeflow pipeline dùng container image để train/evaluate/deploy trên cluster.
