param(
    [int]$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds(),
    [int]$StartIndex = 1,
    [int]$EndIndex = 50,
    [int]$ChunkSize = 10,
    [int]$MaxTrainRows = 60000,
    [bool]$EnableMlflow = $true,
    [bool]$SavePredictions = $true,
    [bool]$LogDiagnostics = $true,
    [int]$ShapRows = 0,
    [int]$PauseSecondsBetweenChunks = 10,
    [string]$ParentRunId = "",
    [string]$FeaturePool = "",
    [string]$CacheFamily = "initial_non_bb_candidates"
)

$ErrorActionPreference = "Continue"
$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InnerRunner = Join-Path $WorkDir "run_rolling_zscore_50_process_loop.ps1"
$LogPath = Join-Path $WorkDir "rolling_zscore_50_h4h1_chunked.log"
$ErrPath = Join-Path $WorkDir "rolling_zscore_50_h4h1_chunked.err.log"

Set-Location $WorkDir

if ($ChunkSize -lt 1) {
    throw "ChunkSize must be >= 1"
}
if ($StartIndex -lt 1 -or $EndIndex -gt 50 -or $StartIndex -gt $EndIndex) {
    throw "Require 1 <= StartIndex <= EndIndex <= 50"
}

"run_id=$RunId start_index=$StartIndex end_index=$EndIndex chunk_size=$ChunkSize max_train_rows=$MaxTrainRows shap_rows=$ShapRows" | Out-File $LogPath -Encoding utf8
"" | Out-File $ErrPath -Encoding utf8

if ($EnableMlflow -and [string]::IsNullOrWhiteSpace($ParentRunId)) {
    $env:RF_RESEARCH_RUN_ID = "$RunId"
    $env:RF_RESEARCH_RUNNER = "run_rolling_zscore_50_chunked.ps1"
    $ParentRunId = python -c "import os; from research_random50_initial_features import _mlflow; m=_mlflow(); r=m.start_run(run_name='rolling_zscore_50_h4h1_' + os.environ['RF_RESEARCH_RUN_ID'] + '_exploration_chunked_parent'); print(r.info.run_id); m.log_param('research_run_id', os.environ['RF_RESEARCH_RUN_ID']); m.log_param('runner', os.environ['RF_RESEARCH_RUNNER']); m.log_param('run_scope', 'exploration_chunked'); m.end_run()"
    "mlflow_parent_run_id=$ParentRunId" | Tee-Object -FilePath $LogPath -Append
}

$ChunkStart = $StartIndex
while ($ChunkStart -le $EndIndex) {
    $ChunkEnd = [Math]::Min($ChunkStart + $ChunkSize - 1, $EndIndex)
    $StartStamp = Get-Date -Format o
    "$StartStamp start_chunk=$ChunkStart-$ChunkEnd" | Tee-Object -FilePath $LogPath -Append

    & $InnerRunner `
        -RunId $RunId `
        -StartIndex $ChunkStart `
        -EndIndex $ChunkEnd `
        -MaxTrainRows $MaxTrainRows `
        -EnableMlflow $EnableMlflow `
        -SavePredictions $SavePredictions `
        -LogDiagnostics $LogDiagnostics `
        -ShapRows $ShapRows `
        -ParentRunId $ParentRunId `
        -FeaturePool $FeaturePool `
        -CacheFamily $CacheFamily `
        1>> $LogPath 2>> $ErrPath

    $Code = $LASTEXITCODE
    $EndStamp = Get-Date -Format o
    "$EndStamp end_chunk=$ChunkStart-$ChunkEnd exit_code=$Code" | Tee-Object -FilePath $LogPath -Append

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()

    $SummaryPath = Join-Path $WorkDir "outputs\rolling_zscore_50_h4h1_$RunId\summary.csv"
    if (Test-Path -LiteralPath $SummaryPath) {
        $Rows = (Import-Csv $SummaryPath | Measure-Object).Count
        "checkpoint_summary=$SummaryPath rows=$Rows" | Tee-Object -FilePath $LogPath -Append
    }

    if ($Code -ne 0) {
        "failed at chunk=$ChunkStart-$ChunkEnd" | Tee-Object -FilePath $LogPath -Append
        exit $Code
    }

    if ($PauseSecondsBetweenChunks -gt 0 -and $ChunkEnd -lt $EndIndex) {
        "pause_seconds=$PauseSecondsBetweenChunks" | Tee-Object -FilePath $LogPath -Append
        Start-Sleep -Seconds $PauseSecondsBetweenChunks
    }

    $ChunkStart = $ChunkEnd + 1
}

"completed run_id=$RunId" | Tee-Object -FilePath $LogPath -Append
