param(
    [string]$DataPath = "D:\RunningSCJ999\data\1M",
    [string]$Variant = "products",
    [int]$TopN = 10,
    [int]$Trials = 50,
    [int]$MaxTrainRows = 1000000,
    [int]$Seed = 92101
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path "$PSScriptRoot\.."
$env:ML_TRADE_DATA_DIR = $DataPath
$env:MLFLOW_TRACKING_URI = "sqlite:///$($repo.Path.Replace('\','/'))/mlflow.db"
$env:PYTHONPATH = "$($repo.Path)\research\rf_mlflow;$($repo.Path)\vendor\scj"

Push-Location "$($repo.Path)\research\rf_mlflow"
try {
    python research_h4edge_feature_transforms_optuna.py `
        --variants $Variant `
        --top-n $TopN `
        --n-trials $Trials `
        --max-train-rows $MaxTrainRows `
        --seed $Seed `
        --enable-mlflow
}
finally {
    Pop-Location
}
