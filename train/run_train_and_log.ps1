<#
Simple PowerShell helper to run training and save logs.

Usage:
  .\run_train_and_log.ps1            # runs python train.py and saves train.log
  .\run_train_and_log.ps1 -InstallDeps  # creates venv (if missing) and installs requirements

This script assumes PowerShell execution policy allows running local scripts.
#>

param(
    [switch]$InstallDeps
)

Write-Host "Running training and logging output to train.log"

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

if ($InstallDeps) {
    if (!(Test-Path ".venv")) {
        python -m venv .venv
    }
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    if (Test-Path requirements.txt) {
        pip install -r requirements.txt
    }
}

# Run training and capture stdout/stderr
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "train.log"

Write-Host "Starting training at $(Get-Date)"
python train.py *>&1 | Tee-Object -FilePath $logFile
Write-Host "Training finished at $(Get-Date). Logs saved to $logFile"
