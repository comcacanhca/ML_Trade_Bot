# Airflow guide: MT5 raw data ingestion to GCS

Tài liệu này hướng dẫn dùng Airflow để tự động cào raw data XAUUSD M1 từ MT5 và upload lên Google Cloud Storage mỗi ngày lúc 00:00.

## Kiến trúc

Airflow chạy trong WSL/Linux. Task cào MT5 gọi Windows Python vì package `MetaTrader5` cần terminal MetaTrader 5 trên Windows.

```text
Airflow WSL, 00:00 Asia/Bangkok mỗi ngày
└── mt5_xauusd_daily_gcs
    └── gọi Windows Python
        └── D:\ML_Trade_Bot\data\write_data_2026.py
            ├── tải XAUUSD M1 từ MT5
            ├── merge vào raw CSV theo năm
            ├── validate OHLC
            └── upload CSV lên GCS
```

File chính:

```text
airflow/dags/mt5_xauusd_daily_gcs.py
data/write_data_2026.py
```

## 1. Cài dependency cho Python project Windows

Chạy trong PowerShell Windows:

```powershell
cd D:\ML_Trade_Bot
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tối thiểu cần:

```powershell
pip install MetaTrader5 google-cloud-storage
```

MT5 terminal phải đang mở và đã login tài khoản broker.

## 2. Cấu hình GCS

Service account key mặc định:

```text
D:\ML_Trade_Bot\t-bounty-508213-u7-f4627e44cf81.json
```

File key này đã được ignore trong `.gitignore`, không commit lên git.

Bạn cần set bucket thật:

```powershell
$env:GCS_BUCKET="your-gcs-bucket-name"
```

Ví dụ test thủ công:

```powershell
cd D:\ML_Trade_Bot
.\.venv\Scripts\Activate.ps1

$env:GCS_BUCKET="your-gcs-bucket-name"
$env:GCS_PREFIX="raw/1M"
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\ML_Trade_Bot\t-bounty-508213-u7-f4627e44cf81.json"
$env:ML_TRADE_DATA_DIR="D:\ML_Trade_Bot\data_handler\raw\1M"
$env:MT5_SYMBOL="XAUUSD"
$env:MT5_SYMBOL_FILE_PREFIX="XAUUSDm"

python data\write_data_2026.py
```

Nếu broker dùng symbol khác:

```powershell
$env:MT5_SYMBOL="XAUUSDm"
python data\write_data_2026.py
```

Output local:

```text
D:\ML_Trade_Bot\data_handler\raw\1M\XAUUSDm_2026.csv
```

Output GCS:

```text
gs://your-gcs-bucket-name/raw/1M/XAUUSDm_2026.csv
```

## 3. Link DAG vào Airflow trong WSL

Chạy trong WSL:

```bash
cd ~/airflow
source .venv/bin/activate

export AIRFLOW_HOME=~/airflow/airflow_home
mkdir -p "$AIRFLOW_HOME/dags"

ln -sf /mnt/d/ML_Trade_Bot/airflow/dags/mt5_xauusd_daily_gcs.py \
  "$AIRFLOW_HOME/dags/mt5_xauusd_daily_gcs.py"
```

## 4. Start Airflow

Chạy trong WSL:

```bash
cd ~/airflow
source .venv/bin/activate

export AIRFLOW_HOME=~/airflow/airflow_home
export GCS_BUCKET="your-gcs-bucket-name"
export GCS_PREFIX="raw/1M"

airflow standalone
```

Mở UI:

```text
http://localhost:8080
```

Bật DAG:

```text
mt5_xauusd_daily_gcs
```

Lịch chạy:

```text
00:00 mỗi ngày, timezone Asia/Bangkok
```

Lấy mật khẩu:

```bash
cat "$AIRFLOW_HOME/airflow_home/airflow.db" | grep "password"
```


## 5. Biến môi trường DAG

DAG đọc các biến môi trường sau:

```text
GCS_BUCKET                         bucket bắt buộc
GCS_PREFIX                         prefix trên GCS, default raw/1M
MT5_SYMBOL                         symbol tải từ MT5, default XAUUSD
MT5_SYMBOL_FILE_PREFIX             prefix tên file local, default XAUUSDm
MT5_LOOKBACK_DAYS                  số ngày tải lại nếu chưa có file, default 7
ML_TRADE_BOT_PROJECT_WIN           default D:\ML_Trade_Bot
ML_TRADE_BOT_PROJECT_WSL           default /mnt/d/ML_Trade_Bot
ML_TRADE_BOT_WINDOWS_PYTHON_EXE    default /mnt/d/ML_Trade_Bot/.venv/Scripts/python.exe
MT5_INGEST_SCRIPT_WIN              default D:\ML_Trade_Bot\data\write_data_2026.py
ML_TRADE_DATA_DIR_WIN              default D:\ML_Trade_Bot\data_handler\raw\1M
GOOGLE_APPLICATION_CREDENTIALS_WIN default D:\ML_Trade_Bot\t-bounty-508213-u7-f4627e44cf81.json
```

## 6. Test DAG

Trong WSL:

```bash
airflow dags list | grep mt5_xauusd_daily_gcs
airflow dags test mt5_xauusd_daily_gcs 2026-09-10
```

Nếu DAG fail, kiểm tra log task `ingest_rawdata_to_gcs` trong Airflow UI.

## 7. Lỗi phổ biến

- `Missing GCS_BUCKET`: chưa export bucket trước khi start Airflow.
- `MetaTrader5 package is not installed`: chưa cài `MetaTrader5` trong `D:\ML_Trade_Bot\.venv`.
- `MT5 initialize() failed`: MT5 terminal chưa mở hoặc chưa login.
- `symbol_select failed`: sai symbol, thử `XAUUSDm` thay vì `XAUUSD`.
- `GCS service account key not found`: sai path key JSON.

