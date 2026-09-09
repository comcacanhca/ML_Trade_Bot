param(
    [string]$DataPath = "D:\RunningSCJ999\data\1M",
    [string]$RemoteName = "",
    [string]$RemoteUrl = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    throw "DVC is not installed or not in PATH. Run: pip install dvc"
}

if (-not (Test-Path ".git")) {
    git init
}

if (-not (Test-Path ".dvc")) {
    dvc init
}

if (Test-Path $DataPath) {
    New-Item -ItemType Directory -Force -Path "data/raw" | Out-Null
    if (-not (Test-Path "data/raw/1M")) {
        New-Item -ItemType Junction -Path "data/raw/1M" -Target $DataPath | Out-Null
    }
    dvc add data/raw/1M
}

if ($RemoteName -and $RemoteUrl) {
    dvc remote add -d $RemoteName $RemoteUrl
}

dvc status
