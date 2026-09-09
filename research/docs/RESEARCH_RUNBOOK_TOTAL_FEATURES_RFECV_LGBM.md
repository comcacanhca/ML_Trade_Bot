# Research runbook: Total features + RFECV + LightGBM

Thư mục làm việc:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow
```

Pipeline:

```powershell
research_total_features_rfecv_lgbm.py
```

Mục tiêu:

```text
1. Dùng toàn bộ feature trong cache.
2. Scaling cố định bằng rolling_zscore.
3. Chọn feature bằng RFECV trên train years בלבד.
4. Train final model bằng LightGBM.
5. Đánh giá trên valid/test, log artifact cần thiết vào local output và MLflow.
```

Feature time/session được giữ cố định sau RFECV theo mặc định:

```text
tod_sin, tod_cos, dow_sin, dow_cos,
session_london_ny, session_ny, session_asia
```

Signal features không bị ép giữ mặc định. Nếu muốn ép giữ toàn bộ passthrough gồm cả signal, thêm:

```powershell
--protect-passthrough
```

---

## 1. Dependency

Kiểm tra LightGBM:

```powershell
python -c "import lightgbm as lgb; print(lgb.__version__)"
```

Nếu thiếu:

```powershell
python -m pip install lightgbm
```

---

## 2. Cache đầu vào

Mặc định dùng cache:

```text
initial_non_bb_candidates
```

Cache này hiện có cả overlay-normalized features nếu đã rebuild sau khi chỉnh `research_random50_initial_features.py`.

Kiểm tra số feature:

```powershell
$Cols = Get-Content "cache\initial_non_bb_candidates\feature_columns.json" -Raw | ConvertFrom-Json
$Cols.Count
$Cols | Where-Object { $_ -match "_ov_" } | Measure-Object
```

Nếu cần rebuild cache overlay:

```powershell
python -c "from research_random50_initial_features import build_cache; build_cache(True)"
```

Lưu ý: rebuild cache có thể tốn RAM/thời gian vì hiện feature count lớn.

---

## 3. Smoke test

Chạy rất nhỏ để kiểm tra pipeline không lỗi:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_total_features_rfecv_lgbm.py `
  --run-id $RunId `
  --cache-family initial_non_bb_candidates `
  --rfecv-rows 1500 `
  --max-train-rows 3000 `
  --preselect-top-n 120 `
  --preselect-corr-threshold 0.98 `
  --preselect-corr-rows 1500 `
  --preselect-mutual-info-rows 1500 `
  --preselect-n-estimators 30 `
  --min-features-to-select 20 `
  --rfecv-step 300 `
  --cv-splits 2 `
  --n-estimators-rfecv 20 `
  --n-estimators-final 40 `
  --n-jobs 2 `
  --log-diagnostics `
  --shap-rows 0
```

Output:

```powershell
outputs\total_features_rfecv_lgbm_$RunId
```

---

## 4. Chạy nghiên cứu RAM-safe lần đầu

Config khuyến nghị cho lần đầu:

```text
rfecv_rows            = 20000
max_train_rows        = 60000
preselect_top_n       = 300
min_features_to_select = 40
rfecv_step            = 100
cv_splits             = 3
shap_rows             = 0
n_jobs                = 2
```

Lệnh:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_total_features_rfecv_lgbm.py `
  --run-id $RunId `
  --cache-family initial_non_bb_candidates `
  --rfecv-rows 20000 `
  --max-train-rows 60000 `
  --preselect-top-n 300 `
  --preselect-corr-threshold 0.98 `
  --preselect-corr-rows 30000 `
  --preselect-mutual-info-rows 20000 `
  --preselect-n-estimators 250 `
  --min-features-to-select 40 `
  --rfecv-step 100 `
  --cv-splits 3 `
  --n-estimators-rfecv 150 `
  --n-estimators-final 700 `
  --n-jobs 2 `
  --enable-mlflow `
  --log-diagnostics `
  --shap-rows 0
```

---

## 5. Chạy nghiên cứu chuẩn hơn

Config mạnh hơn:

```text
rfecv_rows             = 50000
max_train_rows         = 120000
preselect_top_n        = 300
min_features_to_select = 60
rfecv_step             = 50
cv_splits              = 3
shap_rows              = 0
n_jobs                 = 4
```

Lệnh:

```powershell
$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds()

python research_total_features_rfecv_lgbm.py `
  --run-id $RunId `
  --cache-family initial_non_bb_candidates `
  --rfecv-rows 50000 `
  --max-train-rows 120000 `
  --preselect-top-n 300 `
  --preselect-corr-threshold 0.98 `
  --preselect-corr-rows 30000 `
  --preselect-mutual-info-rows 20000 `
  --preselect-n-estimators 250 `
  --min-features-to-select 60 `
  --rfecv-step 50 `
  --cv-splits 3 `
  --n-estimators-rfecv 250 `
  --n-estimators-final 1200 `
  --n-jobs 4 `
  --enable-mlflow `
  --log-diagnostics `
  --shap-rows 0
```

---

## 6. Refit có SHAP

Chỉ bật SHAP sau khi bản không SHAP chạy ổn.

Nhẹ:

```powershell
python research_total_features_rfecv_lgbm.py `
  --run-id $RunId `
  --cache-family initial_non_bb_candidates `
  --rfecv-rows 50000 `
  --max-train-rows 120000 `
  --preselect-top-n 300 `
  --preselect-corr-threshold 0.98 `
  --preselect-corr-rows 30000 `
  --preselect-mutual-info-rows 20000 `
  --preselect-n-estimators 250 `
  --min-features-to-select 60 `
  --rfecv-step 50 `
  --cv-splits 3 `
  --n-estimators-rfecv 250 `
  --n-estimators-final 1200 `
  --n-jobs 4 `
  --enable-mlflow `
  --log-diagnostics `
  --shap-rows 200
```

Nếu RAM ổn:

```powershell
--shap-rows 500
```

---

## 7. File output cần kiểm tra

Folder:

```powershell
outputs\total_features_rfecv_lgbm_$RunId
```

File chính:

```text
run_config.json
summary.json
metrics.json
selected_features.json
selected_features_with_protected.json
rfecv_feature_ranking.csv
rfecv_cv_results.csv
```

Model/artifact:

```text
model\total_features_rfecv_lgbm_bundle.joblib
model\lgbm_feature_importance.csv
model\lgbm_feature_importance_top40.png
model\threshold_valid.csv
model\threshold_test.csv
model\total_features_rfecv_lgbm_yearly_test.csv
model\total_features_rfecv_lgbm_monthly_test.csv
```

Charts nếu `--log-diagnostics`:

```text
model\total_features_rfecv_lgbm_roc_valid.png
model\total_features_rfecv_lgbm_roc_test.png
model\total_features_rfecv_lgbm_pr_valid.png
model\total_features_rfecv_lgbm_pr_test.png
model\total_features_rfecv_lgbm_calibration_valid.png
model\total_features_rfecv_lgbm_calibration_test.png
model\total_features_rfecv_lgbm_prob_distribution_valid_test.png
model\total_features_rfecv_lgbm_confusion_test.png
model\total_features_rfecv_lgbm_yearly_test_selected_threshold.png
model\total_features_rfecv_lgbm_monthly_test_selected_threshold.png
```

Charts nếu `--shap-rows > 0`:

```text
model\total_features_rfecv_lgbm_shap_importance.csv
model\total_features_rfecv_lgbm_shap_summary_bar.png
model\total_features_rfecv_lgbm_shap_beeswarm.png
model\total_features_rfecv_lgbm_shap_dependence_*.png
```

---

## 8. Xem nhanh kết quả

```powershell
$RunId = 1789001111
Get-Content "outputs\total_features_rfecv_lgbm_$RunId\summary.json" -Raw
```

Xem top feature theo RFECV:

```powershell
Import-Csv "outputs\total_features_rfecv_lgbm_$RunId\rfecv_feature_ranking.csv" |
  Sort-Object {[int]$_.ranking}, feature |
  Select-Object -First 80
```

Xem LightGBM importance:

```powershell
Import-Csv "outputs\total_features_rfecv_lgbm_$RunId\model\lgbm_feature_importance.csv" |
  Select-Object -First 80
```

Xem selected feature count:

```powershell
$Selected = Get-Content "outputs\total_features_rfecv_lgbm_$RunId\selected_features_with_protected.json" -Raw | ConvertFrom-Json
$Selected.selected_features.Count
```

---

## 9. MLflow

Mở UI:

```powershell
mlflow ui --backend-store-uri sqlite:///D:/SCJ999/mlflow.db --host 127.0.0.1 --port 5000
```

URL:

```text
http://127.0.0.1:5000
```

Run name:

```text
total_features_rfecv_lgbm_$RunId
```

Metric cần xem:

```text
valid_auc
test_auc
test_total_wr
test_total_resolved
test_min_year_deals
valid_logloss
test_logloss
```

---

## 10. Diễn giải RFECV

RFECV sẽ train nhiều LightGBM nhỏ để loại feature dần.

Các tham số ảnh hưởng mạnh đến thời gian chạy:

```text
rfecv_rows
preselect_top_n
preselect_corr_rows
preselect_mutual_info_rows
rfecv_step
cv_splits
n_estimators_rfecv
total_features
```

Trade-off:

```text
rfecv_step lớn  -> chạy nhanh hơn, chọn thô hơn
rfecv_step nhỏ  -> chạy chậm hơn, chọn kỹ hơn
rfecv_rows lớn  -> ổn định hơn, tốn RAM/thời gian hơn
preselect_top_n nhỏ -> RFECV nhanh hơn, nhưng có thể loại nhầm feature tốt
cv_splits lớn   -> ổn định hơn, tốn thời gian hơn
```

Với 1400+ features, không nên để:

```text
rfecv_step < 25
rfecv_rows > 120000
shap_rows > 0 trong lần đầu
```

---

## 11. Checklist trước khi kết luận

```text
[ ] Cache đã rebuild và có overlay features.
[ ] RFECV chỉ dùng train years.
[ ] Valid/test không được dùng trong feature selection.
[ ] selected_features không quá lớn đến mức RFECV không thực sự loại được gì.
[ ] valid_auc và test_auc không lệch quá mạnh.
[ ] test_total_resolved đủ lớn.
[ ] yearly/monthly WR không chỉ tốt ở một giai đoạn.
[ ] Top LightGBM importance có logic thị trường.
[ ] Nếu bật SHAP, SHAP không phụ thuộc quá mức vào time/session.
```
