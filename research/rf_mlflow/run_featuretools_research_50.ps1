param(
    [int]$RunId = [int][DateTimeOffset]::Now.ToUnixTimeSeconds(),
    [string]$CacheFamily = "featuretools_candidates",
    [int]$BuildMaxRowsPerYear = 0,
    [int]$FeatureSelectionRows = 120000,
    [int]$FeaturewizRows = 50000,
    [int]$FinalTopN = 45,
    [int]$StartIndex = 1,
    [int]$EndIndex = 50,
    [int]$ChunkSize = 10,
    [int]$MaxTrainRows = 60000,
    [int]$ShapRows = 0,
    [int]$SavePredictions = 0,
    [int]$LogDiagnostics = 1,
    [int]$EnableMlflow = 1
)

$ErrorActionPreference = "Continue"
$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogPath = Join-Path $WorkDir "featuretools_research_50.log"
$ErrPath = Join-Path $WorkDir "featuretools_research_50.err.log"

Set-Location $WorkDir

$SavePredictionsBool = [bool]$SavePredictions
$LogDiagnosticsBool = [bool]$LogDiagnostics
$EnableMlflowBool = [bool]$EnableMlflow

"run_id=$RunId cache_family=$CacheFamily build_max_rows_per_year=$BuildMaxRowsPerYear feature_selection_rows=$FeatureSelectionRows max_train_rows=$MaxTrainRows save_predictions=$SavePredictionsBool" | Out-File $LogPath -Encoding utf8
"" | Out-File $ErrPath -Encoding utf8

$BuildArgs = @(
    "build_featuretools_cache.py",
    "--cache-family", "$CacheFamily"
)
if ($BuildMaxRowsPerYear -gt 0) {
    $BuildArgs += @("--max-rows-per-year", "$BuildMaxRowsPerYear")
}

"phase=build_featuretools_cache start" | Tee-Object -FilePath $LogPath -Append
python @BuildArgs 1>> $LogPath 2>> $ErrPath
if ($LASTEXITCODE -ne 0) {
    "phase=build_featuretools_cache failed exit_code=$LASTEXITCODE" | Tee-Object -FilePath $LogPath -Append
    exit $LASTEXITCODE
}
[GC]::Collect()
[GC]::WaitForPendingFinalizers()
"phase=build_featuretools_cache done" | Tee-Object -FilePath $LogPath -Append

$FsRunId = $RunId
$FeatureSelectionArgs = @(
    "research_feature_selection_rolling_zscore.py",
    "--run-id", "$FsRunId",
    "--cache-family", "$CacheFamily",
    "--max-train-rows", "$FeatureSelectionRows",
    "--method", "featurewiz",
    "--corr-threshold", "0.95",
    "--max-corr-rows", "50000",
    "--featurewiz-corr-limit", "0.95",
    "--featurewiz-rows", "$FeaturewizRows",
    "--final-top-n", "$FinalTopN"
)
if ($EnableMlflowBool) {
    $FeatureSelectionArgs += "--enable-mlflow"
}

"phase=feature_selection start fs_run_id=$FsRunId" | Tee-Object -FilePath $LogPath -Append
python @FeatureSelectionArgs 1>> $LogPath 2>> $ErrPath
if ($LASTEXITCODE -ne 0) {
    "phase=feature_selection failed exit_code=$LASTEXITCODE" | Tee-Object -FilePath $LogPath -Append
    exit $LASTEXITCODE
}
[GC]::Collect()
[GC]::WaitForPendingFinalizers()
"phase=feature_selection done" | Tee-Object -FilePath $LogPath -Append

$FeaturePool = Join-Path $WorkDir "outputs\feature_selection_rolling_zscore_$FsRunId\selected_features.json"
if (-not (Test-Path -LiteralPath $FeaturePool)) {
    "missing_feature_pool=$FeaturePool" | Tee-Object -FilePath $LogPath -Append
    exit 2
}
"feature_pool=$FeaturePool" | Tee-Object -FilePath $LogPath -Append

"phase=train_50_chunked start" | Tee-Object -FilePath $LogPath -Append
& (Join-Path $WorkDir "run_rolling_zscore_50_chunked.ps1") `
    -RunId $RunId `
    -StartIndex $StartIndex `
    -EndIndex $EndIndex `
    -ChunkSize $ChunkSize `
    -MaxTrainRows $MaxTrainRows `
    -EnableMlflow $EnableMlflowBool `
    -SavePredictions $SavePredictionsBool `
    -LogDiagnostics $LogDiagnosticsBool `
    -ShapRows $ShapRows `
    -FeaturePool $FeaturePool `
    -CacheFamily $CacheFamily `
    -PauseSecondsBetweenChunks 10 `
    1>> $LogPath 2>> $ErrPath

if ($LASTEXITCODE -ne 0) {
    "phase=train_50_chunked failed exit_code=$LASTEXITCODE" | Tee-Object -FilePath $LogPath -Append
    exit $LASTEXITCODE
}
"phase=train_50_chunked done" | Tee-Object -FilePath $LogPath -Append
"completed run_id=$RunId" | Tee-Object -FilePath $LogPath -Append
