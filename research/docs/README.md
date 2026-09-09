# RF_MLFlow - XAUUSD M1 Buy Win/Lose Research

Pipeline nghiên cứu RandomForest theo hướng MLOps cho bài toán dự đoán xác suất Win/Lose của sự kiện Buy.

Rule cố định:

- Symbol/timeframe: XAUUSD M1.
- Side: Buy only.
- Entry: open của nến tiếp theo sau signal.
- RR = 1.
- 1R = 6 giá.
- Mô phỏng đầu cuối dùng `method.MoPhongDeals.MoPhongDeals` qua adapter trong pipeline.
- Test chính: 2024, 2025, 2026.
- Target: winrate tối thiểu 58%, 2024/2025 tối thiểu khoảng 3000 deals/năm, tổng 2024-2026 khoảng 7000-10000 deals.

## Cấu trúc

- `config.py`: cấu hình đường dẫn, split, rule trade, hyperparameter.
- `features.py`: feature engineering time-series, rolling, lag, cycle, signal, multi-timeframe, scaler chống outlier.
- `labels.py`: tạo label Win/Lose cho Buy theo entry next open, TP/SL ±6.
- `mophong_adapter.py`: mô phỏng đầu cuối bằng `MoPhongDeals`.
- `train_rf_mlflow.py`: build dataset, train RF, log MLflow, quét threshold, lưu artifact.
- `run_strategy.py`: load model/scaler/feature list và chạy mô phỏng test.

## Chạy

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
python train_rf_mlflow.py --force
python run_strategy.py --years 2024-2026
```

Chạy research 10 bộ features + hyperparameter search:

```powershell
python research_10_feature_sets.py --n-iter 2 --max-train-samples 20000 --simulate-top 2
```

Muốn chạy kỹ hơn, tăng `--n-iter`, `--max-train-samples`, và chỉ dùng `--refit-full` khi đã chọn được feature set tiềm năng vì refit full rất chậm.

Nếu `mlflow` chưa cài, pipeline vẫn chạy và lưu artifact cục bộ; khi có `mlflow` sẽ log experiment tự động vào `D:\SCJ999\mlflow.db`.
