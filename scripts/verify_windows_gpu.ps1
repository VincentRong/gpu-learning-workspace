$ErrorActionPreference = "Stop"

function Invoke-Check {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "== $Label ==" -ForegroundColor Cyan
    try {
        & $Action
    }
    catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

Invoke-Check "Host Python" { python --version }
Invoke-Check "NVIDIA Driver" { nvidia-smi }
Invoke-Check "WSL Status" { wsl.exe --status }
Invoke-Check "WSL Distributions" { wsl.exe --list --verbose }

Write-Host ""
Write-Host "If Ubuntu is on WSL2 and the GPU is visible, continue in Ubuntu with scripts/verify_wsl_gpu.sh." -ForegroundColor Green

