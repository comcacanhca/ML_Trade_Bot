$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\research\rf_mlflow;$PWD\vendor\scj"
python pipelines/kubeflow/pipeline.py
