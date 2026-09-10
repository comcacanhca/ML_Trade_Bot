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
    bucket = os.getenv("GCS_BUCKET")
    if not bucket:
        raise RuntimeError("Missing GCS_BUCKET in Airflow environment.")

    env = os.environ.copy()
    env.update(
        {
            "ML_TRADE_DATA_DIR": DATA_DIR_WIN,
            "GOOGLE_APPLICATION_CREDENTIALS": GCS_KEY_FILE_WIN,
            "GCS_BUCKET": bucket,
            "GCS_PREFIX": os.getenv("GCS_PREFIX", "raw/1M"),
            "MT5_SYMBOL": os.getenv("MT5_SYMBOL", "XAUUSD"),
            "MT5_SYMBOL_FILE_PREFIX": os.getenv("MT5_SYMBOL_FILE_PREFIX", "XAUUSDm"),
            "MT5_LOOKBACK_DAYS": os.getenv("MT5_LOOKBACK_DAYS", "7"),
        }
    )

    cmd = [
        WINDOWS_PYTHON_EXE,
        SCRIPT_WIN,
        "--symbol",
        env["MT5_SYMBOL"],
        "--symbol-file-prefix",
        env["MT5_SYMBOL_FILE_PREFIX"],
        "--data-dir",
        DATA_DIR_WIN,
        "--gcs-bucket",
        bucket,
        "--gcs-prefix",
        env["GCS_PREFIX"],
        "--gcs-key-file",
        GCS_KEY_FILE_WIN,
        "--lookback-days",
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
    tags=["mt5", "xauusd", "gcs", "rawdata"],
) as dag:
    ingest_rawdata_to_gcs = PythonOperator(
        task_id="ingest_rawdata_to_gcs",
        python_callable=_run_windows_mt5_ingestion,
    )
