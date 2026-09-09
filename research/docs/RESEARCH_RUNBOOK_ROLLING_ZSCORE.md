# Research runbook: RF rolling_zscore

File này dùng để chạy lần lượt các luồng nghiên cứu RandomForest hiện tại trong thư mục:

```powershell
D:\SCJ999\srateries\RandomForest\RF_MLFlow
```

Pipeline chính:

```powershell
research_rolling_zscore_50_h4h1.py
```

Pipeline feature selection:

```powershell
research_feature_selection_rolling_zscore.py
```

Pipeline Featuretools auto feature engineering:

```powershell
build_featuretools_cache.py
```

Scaling cố định:

```text
rolling_zscore(window=500, min_periods=100, use_past_only=True)
```

Các feature time/signal đang được passthrough, không đưa qua rolling scaler:

```text
tod_sin, tod_cos, dow_sin, dow_cos,
session_london_ny, session_ny, session_asia,
sig_bb_reversion_buy, sig_pullback_trend_buy, sig_momentum_buy, buy_signal
```

Split train/valid/test nằm trong:

```powershell
config.py
```

Hiện tại:

```text
Train: 2018, 2019, 2020, 2021, 2022
Valid: 2023
Test : 2024, 2025, 2026
```

---

## 0. Chuẩn bị

Mở PowerShell:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
```

Nếu muốn xem MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///D:/SCJ999/mlflow.db --host 127.0.0.1 --port 5000
```

Sau đó mở:

```text
http://127.0.0.1:5000
```

Experiment:

```text
RF_MLFlow_XAUUSD_M1_Buy_WinLose
```

---

## 1. Phase 0: feature selection train-only

Mục tiêu: dùng feature engineering cache hiện có, scale bằng `rolling_zscore`, sau đó chọn feature chỉ trên train years. Valid/test không được dùng trong phase này.

Thuật toán feature selection hiện dùng:

```text
1. Remove constant/near-constant features
2. Correlation pruning, default abs corr >= 0.95
3. FeatureWiz selection
4. Xuất selected_features.json cho phase train
```

Dependency:

```powershell
python -m pip install featurewiz
```

Nếu chưa cài `featurewiz`, script sẽ dừng rõ ràng và báo lỗi, không fallback ngầm sang method khác.

Chạy:

```powershell
$FsRunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_feature_selection_rolling_zscore.py `
  --run-id $FsRunId `
  --max-train-rows 120000 `
  --method featurewiz `
  --corr-threshold 0.95 `
  --max-corr-rows 50000 `
  --featurewiz-corr-limit 0.95 `
  --featurewiz-rows 50000 `
  --final-top-n 45 `
  --enable-mlflow
```

Output:

```powershell
outputs\feature_selection_rolling_zscore_$FsRunId\selected_features.json
outputs\feature_selection_rolling_zscore_$FsRunId\feature_selection_report.json
outputs\feature_selection_rolling_zscore_$FsRunId\selection_score.csv
outputs\feature_selection_rolling_zscore_$FsRunId\featurewiz_selected.csv
outputs\feature_selection_rolling_zscore_$FsRunId\correlation_dropped.csv
```

Chart:

```powershell
outputs\feature_selection_rolling_zscore_$FsRunId\charts\featurewiz_selected_top60.png
outputs\feature_selection_rolling_zscore_$FsRunId\charts\selection_score_top40.png
```

Nếu muốn chạy bản lai để so FeatureWiz với L1/RF:

```powershell
python research_feature_selection_rolling_zscore.py `
  --run-id $FsRunId `
  --max-train-rows 120000 `
  --method featurewiz_hybrid `
  --corr-threshold 0.95 `
  --max-corr-rows 50000 `
  --featurewiz-corr-limit 0.95 `
  --featurewiz-rows 50000 `
  --l1-c 0.03 `
  --rf-top-n 60 `
  --final-top-n 45 `
  --enable-mlflow
```

Biến dùng cho phase train:

```powershell
$FeaturePool = "outputs\feature_selection_rolling_zscore_$FsRunId\selected_features.json"
```

---

## 2. Phase 0B: Featuretools auto feature engineering

Mục tiêu: dùng Featuretools để tự động sinh thêm feature từ nhóm OHLC/indicator base, sau đó vẫn giữ nguyên time/signal feature.

Dependency:

```powershell
python -m pip install featuretools
```

Build cache Featuretools:

```powershell
python build_featuretools_cache.py `
  --cache-family featuretools_candidates `
  --primitives diff,percent_change,rolling_mean,rolling_std,rolling_min,rolling_max `
  --force
```

Feature time/signal được preserve nguyên bản:

```text
tod_sin, tod_cos, dow_sin, dow_cos,
session_london_ny, session_ny, session_asia,
sig_bb_reversion_buy, sig_pullback_trend_buy, sig_momentum_buy, buy_signal
```

Sau khi build cache Featuretools, chạy FeatureWiz trên cache này:

```powershell
$FsRunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_feature_selection_rolling_zscore.py `
  --run-id $FsRunId `
  --cache-family featuretools_candidates `
  --max-train-rows 120000 `
  --method featurewiz `
  --corr-threshold 0.95 `
  --max-corr-rows 50000 `
  --featurewiz-corr-limit 0.95 `
  --featurewiz-rows 50000 `
  --final-top-n 45 `
  --enable-mlflow
```

Set biến cho phase train:

```powershell
$FeaturePool = "outputs\feature_selection_rolling_zscore_$FsRunId\selected_features.json"
$CacheFamily = "featuretools_candidates"
```

---

## 2C. Phase 0C: rebuild cache với Overlay indicator normalization

Mục tiêu: rebuild cache `initial_non_bb_candidates` để bổ sung nhóm feature overlay-normalized từ:

```text
EMA, SMA, Bollinger Band, Keltner Channel, Donchian Channel,
VWAP, Ichimoku non-shifted lines, Parabolic SAR, SuperTrend
```

Các công thức overlay được add vào cache:

```text
*_value              = ATR-gap
*_distance           = ATR-gap distance
*_pct_distance       = pct distance vs close
*_std                = relative_distance_rolling_rank
*_velocity1/3/5/8
*_acceleration1/3/5
*_value_lag*
*_distance_lag*
*_pct_distance_lag*
*_velocity*_lag*
```

Lưu ý leakage:

```text
- Rolling baseline dùng past-only.
- Velocity/acceleration dùng diff quá khứ.
- Ichimoku không dùng Chikou.
- Ichimoku span_a/span_b dùng raw non-shifted line, không forward-shift.
```

Rebuild cache:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow

python -c "from research_random50_initial_features import build_cache; build_cache(True)"
```

Sau khi rebuild, kiểm tra số feature và một vài cột overlay:

```powershell
$Cols = Get-Content "cache\initial_non_bb_candidates\feature_columns.json" -Raw | ConvertFrom-Json
$Cols.Count
$Cols | Where-Object { $_ -match "_ov_" } | Select-Object -First 40
```

Set biến cho phase train rolling_zscore:

```powershell
$CacheFamily = "initial_non_bb_candidates"
$FeaturePool = ""
```

Nếu muốn chạy FeatureWiz trước khi train 50 model:

```powershell
$FsRunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_feature_selection_rolling_zscore.py `
  --run-id $FsRunId `
  --cache-family initial_non_bb_candidates `
  --max-train-rows 120000 `
  --method featurewiz `
  --corr-threshold 0.95 `
  --max-corr-rows 50000 `
  --featurewiz-corr-limit 0.95 `
  --featurewiz-rows 50000 `
  --final-top-n 60 `
  --enable-mlflow

$FeaturePool = "outputs\feature_selection_rolling_zscore_$FsRunId\selected_features.json"
$CacheFamily = "initial_non_bb_candidates"
```

Chạy 50 model theo chunk để tránh tràn RAM:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

.\run_rolling_zscore_50_chunked.ps1 `
  -RunId $RunId `
  -StartIndex 1 `
  -EndIndex 50 `
  -ChunkSize 10 `
  -MaxTrainRows 60000 `
  -ShapRows 0 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -SavePredictions $false `
  -LogDiagnostics $true `
  -PauseSecondsBetweenChunks 10
```

Nếu muốn scan rộng hơn sau khi cache overlay đã ổn:

```powershell
.\run_rolling_zscore_50_chunked.ps1 `
  -RunId $RunId `
  -StartIndex 1 `
  -EndIndex 50 `
  -ChunkSize 5 `
  -MaxTrainRows 120000 `
  -ShapRows 0 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -SavePredictions $false `
  -LogDiagnostics $true `
  -PauseSecondsBetweenChunks 15
```

Gợi ý thực dụng:

```text
- Lần đầu chỉ nên MaxTrainRows=60000, ShapRows=0.
- Overlay feature count tăng mạnh, không nên bật SHAP trong phase exploration.
- Sau khi có top10, mới refit với ShapRows=200 hoặc 500.
- Nếu RAM vẫn căng, giảm ChunkSize từ 10 xuống 5.
```

---

## 3. Phase 1: exploration 50 model

Mục tiêu: scan rộng 50 model với data vừa phải.

Runner sẽ tự tạo một MLflow parent run và gắn toàn bộ model con vào parent đó bằng `parent_run_id`. Trên MLflow UI, mở parent run để xem các child run `rf_rolling_zscore_XX`.

Khuyến nghị:

```text
MaxTrainRows = 60000
SHAP         = tắt hoặc rất nhỏ
MLflow       = bật
Diagnostics  = bật
Predictions  = bật
```

Chạy khuyến nghị, chia 50 model thành 5 chunk, mỗi chunk 10 model:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

.\run_rolling_zscore_50_chunked.ps1 `
  -RunId $RunId `
  -StartIndex 1 `
  -EndIndex 50 `
  -ChunkSize 10 `
  -MaxTrainRows 60000 `
  -ShapRows 0 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -PauseSecondsBetweenChunks 10
```

Runner này vẫn chạy từng model bằng Python process riêng, nhưng thêm checkpoint rõ sau mỗi 10 model. Nếu bị dừng giữa chừng, dùng lại cùng `$RunId` và resume từ index tiếp theo.

Ví dụ đã chạy xong 1-20, resume từ 21:

```powershell
.\run_rolling_zscore_50_chunked.ps1 `
  -RunId $RunId `
  -StartIndex 21 `
  -EndIndex 50 `
  -ChunkSize 10 `
  -MaxTrainRows 60000 `
  -ShapRows 0 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -PauseSecondsBetweenChunks 10
```

Output local:

```powershell
outputs\rolling_zscore_50_h4h1_$RunId
```

File quan trọng:

```powershell
outputs\rolling_zscore_50_h4h1_$RunId\summary.csv
outputs\rolling_zscore_50_h4h1_$RunId\summary.json
outputs\rolling_zscore_50_h4h1_$RunId\feature_scaling.json
```

---

## 4. Chọn top candidate sau exploration

Đọc summary:

```powershell
$RunId = 1788700001
$Summary = "outputs\rolling_zscore_50_h4h1_$RunId\summary.csv"
Import-Csv $Summary | Sort-Object {[double]$_.objective_score} -Descending | Select-Object -First 10 model_name,template,n_features,threshold,valid_auc,test_auc,test_total_wr,test_total_resolved,test_min_year_deals,objective_score
```

Top 10 theo `valid_auc`:

```powershell
$RunId = 1788700001
$Summary = "outputs\rolling_zscore_50_h4h1_$RunId\summary.csv"
Import-Csv $Summary |
  Sort-Object {[double]$_.valid_auc} -Descending |
  Select-Object -First 10 model_name,template,n_features,threshold,valid_auc,test_auc,test_total_wr,test_total_resolved,test_min_year_deals,objective_score
```

Lấy index từ `model_name`.

Ví dụ:

```text
rf_rolling_zscore_17 -> index 17
rf_rolling_zscore_32 -> index 32
```

Tạo chuỗi index:

```powershell
$Indexes = "17,32,21,7,35,9,2,44,13,48"
```

Lưu ý thực nghiệm: không nên chọn top chỉ bằng `valid_auc`. Nên so thêm:

```text
objective_score
test_total_wr
test_total_resolved
test_min_year_deals
auc_gap_valid_test
```

---

## 5. Phase 2: refit top10 có SHAP

Mục tiêu: train lại top10 với nhiều data hơn và log đủ chart/dữ liệu.

Runner top10 cũng tự tạo một MLflow parent run duy nhất. Các model từ `rf_rolling_zscore_XX` sẽ nằm dưới parent này thay vì bị tách thành nhiều parent riêng.

Khuyến nghị:

```text
MaxTrainRows = 180000
SHAP rows    = 500
MLflow       = bật
Diagnostics  = bật
Predictions  = bật
```

Chạy:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()
$Indexes = "17,32,21,7,35,9,2,44,13,48"

.\run_rolling_zscore_selected_process_loop.ps1 `
  -RunId $RunId `
  -Indexes $Indexes `
  -MaxTrainRows 180000 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -ShapRows 500
```

Nếu máy thiếu RAM hoặc chạy quá chậm:

```powershell
.\run_rolling_zscore_selected_process_loop.ps1 `
  -RunId $RunId `
  -Indexes $Indexes `
  -MaxTrainRows 180000 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -ShapRows 200
```

Nếu muốn tắt SHAP:

```powershell
.\run_rolling_zscore_selected_process_loop.ps1 `
  -RunId $RunId `
  -Indexes $Indexes `
  -MaxTrainRows 180000 `
  -FeaturePool $FeaturePool `
  -CacheFamily $CacheFamily `
  -ShapRows 0
```

---

## 6. Phase 3: final audit 1-3 model tốt nhất

Mục tiêu: train kỹ model cuối cùng với nhiều data hơn và SHAP nhiều hơn.

Ví dụ chạy model index 42:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_rolling_zscore_50_h4h1.py `
  --n-models 50 `
  --only-index 42 `
  --max-train-rows 300000 `
  --seed 860906 `
  --run-id $RunId `
  --enable-mlflow `
  --save-predictions `
  --log-diagnostics `
  --feature-pool $FeaturePool `
  --cache-family $CacheFamily `
  --shap-rows 1000
```

Nếu RAM căng:

```powershell
--max-train-rows 180000 --shap-rows 500
```

Nên chạy lại cùng model với vài seed khác nhau để kiểm tra độ ổn định:

```powershell
--seed 860906
--seed 860907
--seed 860908
```

Nếu kết quả đổi mạnh theo seed, model chưa đủ ổn để dùng làm bản chính.

---

## 7. Artifact cần kiểm tra trong mỗi model

Mỗi model sẽ nằm ở:

```powershell
outputs\rolling_zscore_50_h4h1_$RunId\explainability\rf_rolling_zscore_XX
```

Các file cần có:

```text
*_rf_importance.csv
*_rf_importance_top25.png

*_threshold_valid.csv
*_threshold_test.csv
*_threshold_valid.png
*_threshold_test.png

*_valid_predictions.parquet
*_test_predictions.parquet

*_yearly_test.csv
*_monthly_test.csv
*_yearly_test_selected_threshold.png
*_monthly_test_selected_threshold.png

*_roc_valid.png
*_roc_test.png
*_pr_valid.png
*_pr_test.png
*_calibration_valid.png
*_calibration_test.png
*_prob_distribution_valid_test.png
*_confusion_test.png

*_shap_importance.csv
*_shap_summary_bar.png
*_shap_beeswarm.png
*_shap_dependence_*.png
```

Nếu không thấy file SHAP:

1. Kiểm tra có truyền `--shap-rows` hoặc `-ShapRows` lớn hơn `0` không.
2. Kiểm tra có file `*_shap_error.txt` không.
3. Nếu có lỗi RAM, giảm `ShapRows` xuống `200` hoặc `100`.

---

## 8. Metric nên dùng để chọn model

Không chọn model chỉ bằng `valid_auc`.

Ưu tiên thứ tự:

```text
1. test_total_wr
2. test_total_resolved
3. test_min_year_deals
4. objective_score
5. valid_auc
6. test_auc
7. auc_gap_valid_test
8. yearly/monthly stability
```

Điều kiện loại nhanh:

```text
test_total_resolved quá thấp
test_min_year_deals quá thấp
valid_auc cao nhưng test_auc về gần 0.5
WR chỉ tốt ở 1 năm nhưng yếu ở các năm còn lại
monthly WR dao động quá mạnh
SHAP phụ thuộc quá nhiều vào time/session nếu không có lý do rõ
```

---

## 9. Các mức cấu hình thực dụng

### Chạy nhanh

```text
MaxTrainRows = 60000
ShapRows     = 0
```

### Chạy top10 chuẩn

```text
MaxTrainRows = 180000
ShapRows     = 500
```

### Chạy final kỹ

```text
MaxTrainRows = 300000
ShapRows     = 1000
```

### Khi máy bị force close / thiếu RAM

```text
MaxTrainRows = 60000 - 120000
ShapRows     = 100 - 200
Chạy qua runner .ps1 process-per-model
Không chạy trực tiếp trong PyCharm console
```

---

## 10. Checklist trước khi kết luận một model tốt

Một model chỉ nên coi là candidate tốt nếu qua đủ checklist:

```text
[ ] valid_auc không quá thấp
[ ] test_auc không sụp về dưới 0.5
[ ] test_total_wr tốt hơn baseline
[ ] test_total_resolved đủ lớn
[ ] test_min_year_deals đủ lớn
[ ] yearly WR không chỉ tốt ở một năm
[ ] monthly WR không quá lệch
[ ] threshold được chọn từ valid, không tune theo test
[ ] SHAP top features có logic thị trường
[ ] probability distribution valid/test không lệch quá mạnh
[ ] calibration không quá méo
[ ] chạy lại seed khác không thay đổi quá mạnh
```

---

## 11. Lệnh thường dùng

Xem log exploration:

```powershell
Get-Content rolling_zscore_50_h4h1_process_loop.log -Tail 80
Get-Content rolling_zscore_50_h4h1_process_loop.err.log -Tail 80
```

Xem log top10:

```powershell
Get-Content rolling_zscore_top_valid_auc_process_loop.log -Tail 80
Get-Content rolling_zscore_top_valid_auc_process_loop.err.log -Tail 80
```

Xem top result:

```powershell
$RunId = 1788700001
Import-Csv "outputs\rolling_zscore_50_h4h1_$RunId\summary.csv" |
  Sort-Object {[double]$_.objective_score} -Descending |
  Select-Object -First 20
```

Mở folder output:

```powershell
explorer "D:\SCJ999\srateries\RandomForest\RF_MLFlow\outputs"
```
