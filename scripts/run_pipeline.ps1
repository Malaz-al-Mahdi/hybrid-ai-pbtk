###############################################################################
# run_pipeline.ps1
# ---------------------------------------------------------------------------
# Orchestrates the three-step pilot workflow:
#   Step 1 (R)      – Extract data from httk
#   Step 2 (Python) – Train RF, predict Clint, evaluate LOO-CV
#   Step 3 (R)      – Run PBTK simulations with native vs. imputed params
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
#
# Prerequisites:
#   - R with httk package installed
#   - Python 3.8+ with packages in requirements.txt
###############################################################################

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$env:R_LIBS_USER = Join-Path $HOME "Documents\R\win-library\4.5"

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "  PBTK Pilot Pipeline" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan
Write-Host "  R user library: $env:R_LIBS_USER`n" -ForegroundColor DarkCyan

# --- Step 1: Extract httk data ---
Write-Host "[1/3] Extracting httk data (R) ..." -ForegroundColor Yellow
Set-Location "$ScriptDir"
Rscript "01_extract_httk_data.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 1 failed. Check R / httk installation."
    exit 1
}
Write-Host "[1/3] Done.`n" -ForegroundColor Green

# --- Step 2: RF prediction ---
Write-Host "[2/3] Training Random Forest & LOO-CV (Python) ..." -ForegroundColor Yellow
python "02_rf_predict_clint.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 2 failed. Check Python / scikit-learn installation."
    exit 1
}
Write-Host "[2/3] Done.`n" -ForegroundColor Green

# --- Step 3: PBTK simulation ---
Write-Host "[3/3] Running PBTK simulations (R + httk) ..." -ForegroundColor Yellow
Rscript "03_httk_pbtk_simulation.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 3 failed. Check R / httk parameterize_pbtk."
    exit 1
}
Write-Host "[3/3] Done.`n" -ForegroundColor Green

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Pipeline complete!" -ForegroundColor Cyan
Write-Host "  Results in: $ProjectRoot\results\" -ForegroundColor Cyan
Write-Host "  Data in:    $ProjectRoot\data\" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan
