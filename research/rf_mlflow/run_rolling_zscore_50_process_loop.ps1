param(
    [int]$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds(),
    [int]$StartIndex = 1,
    [int]$EndIndex = 50,
    [int]$MaxTrainRows = 60000,
    [bool]$EnableMlflow = $true,
    [bool]$SavePredictions = $true,
    [bool]$LogDiagnostics = $true,
    [int]$ShapRows = 0,
    [string]$ParentRunId = "",
    [string]$FeaturePool = "",
    [string]$CacheFamily = "initial_non_bb_candidates"
)

$ErrorActionPreference = "Continue"
$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $WorkDir "research_rolling_zscore_50_h4h1.py"
$LogPath = Join-Path $WorkDir "rolling_zscore_50_h4h1_process_loop.log"
$ErrPath = Join-Path $WorkDir "rolling_zscore_50_h4h1_process_loop.err.log"

Set-Location $WorkDir
"run_id=$RunId start_index=$StartIndex end_index=$EndIndex max_train_rows=$MaxTrainRows" | Out-File $LogPath -Encoding utf8

if ($EnableMlflow -and [string]::IsNullOrWhiteSpace($ParentRunId)) {
    $env:RF_RESEARCH_RUN_ID = "$RunId"
    $env:RF_RESEARCH_RUNNER = "run_rolling_zscore_50_process_loop.ps1"
    $ParentRunId = python -c "import os; from research_random50_initial_features import _mlflow; m=_mlflow(); r=m.start_run(run_name='rolling_zscore_50_h4h1_' + os.environ['RF_RESEARCH_RUN_ID'] + '_exploration_parent'); print(r.info.run_id); m.log_param('research_run_id', os.environ['RF_RESEARCH_RUN_ID']); m.log_param('runner', os.environ['RF_RESEARCH_RUNNER']); m.log_param('run_scope', 'exploration'); m.end_run()"
    "mlflow_parent_run_id=$ParentRunId" | Tee-Object -FilePath $LogPath -Append
}

foreach ($Index in $StartIndex..$EndIndex) {
    $StartStamp = Get-Date -Format o
    "$StartStamp start model_index=$Index" | Tee-Object -FilePath $LogPath -Append

    $ArgsList = @(
        $ScriptPath,
        "--n-models", "50",
        "--only-index", "$Index",
        "--run-id", "$RunId",
        "--max-train-rows", "$MaxTrainRows",
        "--cache-family", "$CacheFamily"
    )
    if ($EnableMlflow) { $ArgsList += "--enable-mlflow" }
    if ($SavePredictions) { $ArgsList += "--save-predictions" }
    if ($LogDiagnostics) { $ArgsList += "--log-diagnostics" }
    if ($ShapRows -gt 0) { $ArgsList += @("--shap-rows", "$ShapRows") }
    if (-not [string]::IsNullOrWhiteSpace($ParentRunId)) { $ArgsList += @("--parent-run-id", "$ParentRunId") }
    if (-not [string]::IsNullOrWhiteSpace($FeaturePool)) { $ArgsList += @("--feature-pool", "$FeaturePool") }

    python @ArgsList 1>> $LogPath 2>> $ErrPath
    $Code = $LASTEXITCODE

    $EndStamp = Get-Date -Format o
    "$EndStamp end model_index=$Index exit_code=$Code" | Tee-Object -FilePath $LogPath -Append
    [GC]::Collect()

    if ($Code -ne 0) {
        "failed at model_index=$Index" | Tee-Object -FilePath $LogPath -Append
        exit $Code
    }
}

"completed run_id=$RunId" | Tee-Object -FilePath $LogPath -Append
