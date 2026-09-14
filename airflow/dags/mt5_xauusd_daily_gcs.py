from __future__ import annotations

import os
import subprocess
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_WIN = os.getenv("ML_TRADE_BOT_PROJECT_WIN", r"D:\ML_Trade_Bot")
PROJECT_WSL = os.getenv("ML_TRADE_BOT_PROJECT_WSL", "/mnt/d/ML_Trade_Bot")
WINDOWS_PYTHON_EXE = os.getenv("ML_TRADE_BOT_WINDOWS_PYTHON_EXE", f"{PROJECT_WSL}/.venv/Scripts/python.exe")
SCRIPT_WIN = os.getenv("MT5_INGEST_SCRIPT_WIN", rf"{PROJECT_WIN}\data\write_data_2026.py")
DATA_DIR_WIN = os.getenv("ML_TRADE_DATA_DIR_WIN", rf"{PROJECT_WIN}\data_handler\raw\1M")
GCS_KEY_FILE_WIN = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_WIN", rf"{PROJECT_WIN}\t-bounty-508213-u7-f4627e44cf81.json")


def _run_windows_mt5_ingestion() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{PROJECT_WSL};{PROJECT_WIN}",  # Thêm cả 2 định dạng đường dẫn để đảm bảo
            "ML_TRADE_DATA_DIR": DATA_DIR_WIN,
            "MT5_SYMBOL": os.getenv("MT5_SYMBOL", "XAUUSD"),
            "MT5_SYMBOL_FILE_PREFIX": os.getenv("MT5_SYMBOL_FILE_PREFIX", "XAUUSDm"),
            "MT5_LOOKBACK_DAYS": os.getenv("MT5_LOOKBACK_DAYS", "7"),
        }
    )

    cmd = [
        WINDOWS_PYTHON_EXE,
        # 2. Sử dụng cờ -m để chạy script như một module (Khuyên dùng khi chạy script trong package)
        # HOẶC vẫn giữ SCRIPT_WIN nếu giữ nguyên cấu trúc truyền file trực tiếp
        SCRIPT_WIN,
        "--symbol",
        env["MT5_SYMBOL"],
        "--symbol-file-prefix",
        env["MT5_SYMBOL_FILE_PREFIX"],
        "--data-dir",
        DATA_DIR_WIN,
        env["MT5_LOOKBACK_DAYS"],
    ]


    result = subprocess.run(
        cmd,
        cwd=PROJECT_WSL,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"MT5 ingestion failed with exit code {result.returncode}")


with DAG(
    dag_id="mt5_xauusd_daily_gcs",
    description="Download XAUUSD M1 raw data from MT5 on Windows host and upload yearly CSV to GCS.",
    start_date=pendulum.datetime(2026, 9, 10, tz="Asia/Bangkok"),
    schedule="0 0 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=["mt5", "xauusd", "rawdata"],
) as dag:
    ingest_rawdata_to_gcs = PythonOperator(
        task_id="ingest_rawdata_to_gcs",
        python_callable=_run_windows_mt5_ingestion,
    )
