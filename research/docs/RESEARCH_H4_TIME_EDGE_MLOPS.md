# RESEARCH_H4_TIME_EDGE_MLOPS

Mục tiêu: tìm edge thực dụng cho bài toán Buy Win/Lose bằng hướng `gate/regime-first`, ưu tiên H4/time/BB20-reversion, sau đó mới train model nhỏ bằng LightGBM.

File chính:

```powershell
D:\SCJ999\srateries\RandomForest\RF_MLFlow\research_h4_time_edge_mlop.py
```

## Nguyên tắc của bản này

- Dùng cache candidate hiện tại: `cache\initial_non_bb_candidates`.
- Scaling cố định: `rolling_zscore`, window `500`, min_periods `100`.
- Time/session/signal feature là passthrough, không rolling scale:
  - `tod_sin`
  - `tod_cos`
  - `dow_sin`
  - `dow_cos`
  - `session_london_ny`
  - `session_ny`
  - `session_asia`
  - `sig_bb_reversion_buy`
  - `sig_pullback_trend_buy`
- Model: `LightGBM`.
- Không dùng RFECV/all-overlay full pool ở phase đầu vì kết quả trước đó cho thấy nhiễu lớn và AUC yếu.
- Mỗi model chạy tuần tự, `n_jobs=1`, đọc theo cột cần thiết để giữ RAM thấp.

## Phase 0 - Smoke test

Chạy để kiểm tra contract/cache/pipeline:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
python research_h4_time_edge_mlop.py --n-models 1 --max-train-rows 2000 --shap-rows 0
```

Kỳ vọng:

- Có folder `outputs\h4_time_edge_mlop_lgbm_rz500_<run_id>`.
- Có `edge_scan_gates.csv`.
- Có `summary.csv`.
- Có artifact model local trong `models\`.

## Phase 1 - Edge scan only + few models

Chạy nhanh để xem gate nào có WR lift trên valid/test:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
python research_h4_time_edge_mlop.py --n-models 8 --max-train-rows 50000 --shap-rows 0 --enable-mlflow
```

Đọc trước:

```text
outputs\h4_time_edge_mlop_lgbm_rz500_<run_id>\edge_scan_gates_pivot.csv
outputs\h4_time_edge_mlop_lgbm_rz500_<run_id>\summary.csv
```

## Phase 2 - Research RAM-safe

Luồng mặc định khuyến nghị:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
python research_h4_time_edge_mlop.py --n-models 24 --max-train-rows 120000 --shap-rows 80 --enable-mlflow
```

Nếu RAM còn dư và muốn kiểm tra sâu hơn:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
python research_h4_time_edge_mlop.py --n-models 40 --max-train-rows 180000 --shap-rows 120 --enable-mlflow
```

## Phase 3 - Resume khi bị dừng giữa chừng

Nếu process dừng ở model thứ N, chạy tiếp từ model N:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
python research_h4_time_edge_mlop.py --n-models 24 --max-train-rows 120000 --shap-rows 80 --enable-mlflow --start-index N
```

Lưu ý: lệnh resume tạo parent run mới. Nếu muốn gom chặt vào cùng parent MLflow, cần bổ sung `parent_run_id` sau; bản hiện tại ưu tiên đơn giản/RAM-safe trước.

## Artifact cần xem trong MLflow

Parent run:

- `setup/model_specs.json`
- `setup/feature_scaling.json`
- `setup/edge_scan_gates.csv`
- `setup/edge_scan_gates_pivot.csv`
- `setup/edge_scan_valid_winrate.png`
- `summary/summary.csv`
- `summary/summary.json`

Child run từng model:

- `models/<model_name>/<model_name>_metrics.json`
- `models/<model_name>/<model_name>_threshold_valid.csv`
- `models/<model_name>/<model_name>_threshold_test.csv`
- `models/<model_name>/<model_name>_threshold_valid.png`
- `models/<model_name>/<model_name>_threshold_test.png`
- `models/<model_name>/<model_name>_valid_roc.png`
- `models/<model_name>/<model_name>_valid_calibration.png`
- `models/<model_name>/<model_name>_valid_score_distribution.png`
- `models/<model_name>/<model_name>_test_roc.png`
- `models/<model_name>/<model_name>_test_calibration.png`
- `models/<model_name>/<model_name>_test_score_distribution.png`
- `models/<model_name>/<model_name>_lgbm_importance.csv`
- `models/<model_name>/<model_name>_lgbm_importance_top25.png`
- SHAP files nếu `--shap-rows > 0`.

## Cách đọc kết quả

Ưu tiên thứ tự:

1. `edge_scan_gates_pivot.csv`: gate nào có valid WR lift nhưng test không sập.
2. `summary.csv`: sort theo `objective_score`, không chỉ nhìn `valid_auc`.
3. Với top model:
   - `test_total_wr`
   - `test_total_resolved`
   - `test_min_year_deals`
   - `test_auc`
   - threshold curve valid/test
4. Nếu một model có WR cao nhưng `test_min_year_deals` quá thấp, coi là chưa đủ ổn định.

## Baseline anchor

Script này được thiết kế dựa trên 3 baseline cũ:

- `Best_BB20_Reversion_Buy_SimOnly_package`: BB20 lower reversion, London/NY, buy-only.
- `LGBM_OptLight_StdRelRank500_R1_6`: LightGBM + overlay WMA/std + dynamic Donchian structure.
- `Backtest_RF_FS03_Fixed.py`: FS03 fixed small feature set, threshold khoảng `0.565`, session London/NY.

Điểm quan trọng: không mở rộng toàn bộ 1404 feature ngay. Tìm gate trước, rồi train model nhỏ trong vùng có edge.
