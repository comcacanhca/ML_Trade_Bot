# ML Trade Bot

Repo MLOps mới cho nghiên cứu ML trading, được tách từ:

`D:\SCJ999\srateries\RandomForest\RF_MLFlow`

Trọng tâm hiện tại:

- XAUUSD M1 buy win/lose classification.
- H4/time edge research.
- `rolling_zscore` scaling cho numeric market features.
- Time/session/signal features giữ cố định, không đưa qua rolling scaler.
- Model chính: LightGBM + Optuna.
- MLflow tracking + DVC artifact/data versioning + Kubeflow pipeline skeleton.

## Cấu trúc

```text
D:\ML_Trade_Bot
├── research/rf_mlflow/        # legacy research scripts đã migrate
├── research/docs/             # runbook/notebook nghiên cứu cũ
├── vendor/scj/                # module nội bộ tối thiểu: data, method
├── pipelines/kubeflow/        # Kubeflow pipeline
├── .github/workflows/         # GitHub Actions CI
├── data/                      # data do DVC quản lý
├── models/                    # model artifacts do DVC quản lý
├── outputs/                   # outputs/reports do DVC quản lý
└── params.yaml                # params chuẩn cho experiment
```

## Setup local

```powershell
cd D:\ML_Trade_Bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu data M1 đang nằm ngoài repo:

```powershell
$env:ML_TRADE_DATA_DIR="D:\RunningSCJ999\data\1M"
$env:MLFLOW_TRACKING_URI="sqlite:///D:/ML_Trade_Bot/mlflow.db"
$env:PYTHONPATH="D:\ML_Trade_Bot\research\rf_mlflow;D:\ML_Trade_Bot\vendor\scj"
```

## Chạy research chính hiện tại

```powershell
cd D:\ML_Trade_Bot\research\rf_mlflow
python research_h4edge_feature_transforms_optuna.py `
  --variants products `
  --top-n 10 `
  --n-trials 50 `
  --max-train-rows 1000000 `
  --seed 92101 `
  --enable-mlflow
```

Best migrated baseline cần so sánh:

- `h4edge_feature_transforms_optuna_1788956979`
- Variant: `products_top10_t50`
- Test WR: `58.8773%`
- Test resolved: `6449`
- Test AUC: `0.52525`

## DVC

DVC chưa được init nếu máy chưa cài `dvc`. Sau khi cài:

```powershell
cd D:\ML_Trade_Bot
dvc init
dvc add data/raw/1M
dvc repro feature_transforms
dvc status
```

Cấu hình remote ví dụ:

```powershell
dvc remote add -d storage s3://your-bucket/ml-trade-bot
dvc push
```

Nếu dùng local remote:

```powershell
dvc remote add -d localremote D:\dvc_remote\ml_trade_bot
dvc push
```

## MLflow

Local:

```powershell
mlflow ui --backend-store-uri sqlite:///D:/ML_Trade_Bot/mlflow.db --port 5000
```

## Kubeflow

Compile pipeline:

```powershell
python pipelines/kubeflow/pipeline.py
```

Output:

```text
pipelines/kubeflow/ml_trade_bot_pipeline.yaml
```

Pipeline hiện là skeleton chuẩn để build cache → train/evaluate → register candidate. Phần production deploy/promotion cần nối thêm registry thật sau khi chốt MLflow/DVC remote và container registry.

## CI

GitHub Actions hiện chạy:

- install dependencies
- ruff lint cho phần MLOps mới: `tests`, `pipelines`
- pytest import smoke
- dvc sanity nếu DVC khả dụng

Full training không chạy trong CI để tránh thiếu data/RAM và thời gian chạy dài.

Legacy scripts trong `research/rf_mlflow` chưa bật lint strict ngay để tránh block migration bởi code nghiên cứu cũ.
