###############################################################################
# run_pipeline.ps1
# ---------------------------------------------------------------------------
# Orchestrates the three-step pilot workflow:
#   Step 1 (R)      – Extract data from httk
#   Step 2 (Python) – Train RF, predict Clint, evaluate LOO-CV
#   Step 3 (R)      – Run PBTK simulations with native vs. imputed params
#   Step 4 (R)      – Reverse dosimetry: ToxCast AC50 -> AED (Monte Carlo)
#   Step 4b (Python) – AED visualization and summary report
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
Write-Host "[1/6] Extracting httk data (R) ..." -ForegroundColor Yellow
Set-Location "$ScriptDir"
Rscript "01_extract_httk_data.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 1 failed. Check R / httk installation."
    exit 1
}
Write-Host "[1/6] Done.`n" -ForegroundColor Green

# --- Step 2: RF prediction ---
Write-Host "[2/6] Training Random Forest & LOO-CV (Python) ..." -ForegroundColor Yellow
python "02_rf_predict_clint.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 2 failed. Check Python / scikit-learn installation."
    exit 1
}
Write-Host "[2/6] Done.`n" -ForegroundColor Green

# --- Step 3: PBTK simulation ---
Write-Host "[3/6] Running PBTK simulations (R + httk) ..." -ForegroundColor Yellow
Rscript "03_httk_pbtk_simulation.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 3 failed. Check R / httk parameterize_pbtk."
    exit 1
}
Write-Host "[3/6] Done.`n" -ForegroundColor Green

# --- Step 4: Reverse dosimetry (Monte Carlo AED) ---
Write-Host "[4/6] Reverse dosimetry: ToxCast AC50 -> AED (R + httk MC) ..." -ForegroundColor Yellow
Rscript "04_reverse_dosimetry.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 4 failed. Check httk calc_mc_oral_equiv / ToxCast data."
    exit 1
}
Write-Host "[4/6] Done.`n" -ForegroundColor Green

# --- Step 4b: AED visualization ---
Write-Host "[5/6] AED analysis & visualization (Python) ..." -ForegroundColor Yellow
python "04b_aed_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 4b failed. Check Python / matplotlib."
    exit 1
}
Write-Host "[5/6] Done.`n" -ForegroundColor Green

# --- Step 5: Full RTK + AED + BER for all 777 chemicals ---
Write-Host "[6/6] Full RTK pipeline: 777 chemicals -> AED -> BER (R) ..." -ForegroundColor Yellow
Write-Host "       (This step takes 30-60 min for 777 chemicals)" -ForegroundColor DarkYellow
Rscript "05_full_rtk_aed_ber.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 5 failed. Check results/aed_ber_full.csv."
    exit 1
}
Write-Host "[6/6] Done.`n" -ForegroundColor Green

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Pipeline complete!" -ForegroundColor Cyan
Write-Host "  Results in: $ProjectRoot\results\" -ForegroundColor Cyan
Write-Host "  Data in:    $ProjectRoot\data\" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan
