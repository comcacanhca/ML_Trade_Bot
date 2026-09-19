# FS03 LightGBM Optuna CI/CD Pipeline

Pipeline này chuyển logic từ `research/rf_mlflow/reproduce_fs03_fixed_params.py` sang luồng CI/CD có Optuna:

- Feature store FS03 được lưu tại `data/processed/fs03_lags_cycle_feature_store.parquet` để DVC versioning và tái sử dụng.
- `train_optuna_fs03.py`: tune LightGBM `LGBMClassifier` bằng Optuna trên validation AUC, lưu model bundle, trials, threshold curves và summary.
- `evaluate_fs03.py`: evaluate/backtest một model bundle đã lưu.
- `run_fs03_optuna_pipeline.py`: entrypoint end-to-end cho CI/CD.

MLflow khi bật `--enable-mlflow` sẽ lưu backend DB tại `mlflow.db` và artifacts tại `mlruns/` ở repo root.

Build/update feature store:

```powershell
python pipelines/_optuna/data_processing.py --force-dataset
```

Chạy nhanh trong CI:

```powershell
python pipelines/_optuna/run_fs03_optuna_pipeline.py --n-trials 3 --max-train-rows 50000 --skip-simulate
```

Chạy đầy đủ hơn:

```powershell
python pipelines/_optuna/run_fs03_optuna_pipeline.py --n-trials 50 --thresholds 0.56,0.57,0.58 --enable-mlflow
```

Evaluate lại model:

```powershell
python pipelines/_optuna/evaluate_fs03.py --model-bundle models/<bundle>.joblib --thresholds 0.56,0.57,0.58
```

Artifacts mặc định nằm dưới `outputs/pipelines_optuna/...`, model bundle nằm dưới `models/`.
