###############################################################################
# run_pipeline.ps1
# ---------------------------------------------------------------------------
# Orchestrates the complete toxicokinetic research pipeline:
#
#   Step 1  (R)      – Extract data from httk
#   Step 2  (Python) – Train RF, predict Clint, evaluate LOO-CV
#   Step 3  (R)      – Run PBTK simulations with native vs. imputed params
#   Step 4  (R)      – Reverse dosimetry: ToxCast AC50 -> AED (Monte Carlo)
#   Step 4b (Python) – AED visualization and summary report
#   Step 5  (R)      – Full RTK pipeline: 777 chemicals -> AED -> BER
#   Step 6  (Python) – Neural ODE for continuous C(t) TK modeling
#   Step 7  (Python) – Explainable AI: SHAP for Clint and BER
#   Step 8  (Python) – Bayesian BER: MC Dropout uncertainty analysis
#   Step 9  (R)      – In-vivo validation vs. Wetmore2012 literature data
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
#
# Prerequisites:
#   - R with httk package installed
#   - Python 3.8+ with packages in requirements.txt (torch + shap required for steps 6-8)
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
Write-Host "[1/10] Extracting httk data (R) ..." -ForegroundColor Yellow
Set-Location "$ScriptDir"
Rscript "01_extract_httk_data.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 1 failed. Check R / httk installation."
    exit 1
}
Write-Host "[1/10] Done.`n" -ForegroundColor Green

# --- Step 2: RF prediction ---
Write-Host "[2/10] Training Random Forest & LOO-CV (Python) ..." -ForegroundColor Yellow
python "02_rf_predict_clint.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 2 failed. Check Python / scikit-learn installation."
    exit 1
}
Write-Host "[2/10] Done.`n" -ForegroundColor Green

# --- Step 3: PBTK simulation ---
Write-Host "[3/10] Running PBTK simulations (R + httk) ..." -ForegroundColor Yellow
Rscript "03_httk_pbtk_simulation.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 3 failed. Check R / httk parameterize_pbtk."
    exit 1
}
Write-Host "[3/10] Done.`n" -ForegroundColor Green

# --- Step 4: Reverse dosimetry (Monte Carlo AED) ---
Write-Host "[4/10] Reverse dosimetry: ToxCast AC50 -> AED (R + httk MC) ..." -ForegroundColor Yellow
Rscript "04_reverse_dosimetry.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 4 failed. Check httk calc_mc_oral_equiv / ToxCast data."
    exit 1
}
Write-Host "[4/10] Done.`n" -ForegroundColor Green

# --- Step 4b: AED visualization ---
Write-Host "[5/10] AED analysis & visualization (Python) ..." -ForegroundColor Yellow
python "04b_aed_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 4b failed. Check Python / matplotlib."
    exit 1
}
Write-Host "[5/10] Done.`n" -ForegroundColor Green

# --- Step 5: Full RTK + AED + BER for all 777 chemicals ---
Write-Host "[6/10] Full RTK pipeline: 777 chemicals -> AED -> BER (R) ..." -ForegroundColor Yellow
Write-Host "       (This step takes 30-60 min for 777 chemicals)" -ForegroundColor DarkYellow
Rscript "05_full_rtk_aed_ber.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Step 5 failed. Check results/aed_ber_full.csv."
    exit 1
}
Write-Host "[6/10] Done.`n" -ForegroundColor Green

# --- Step 6: Neural ODE for continuous TK modeling ---
Write-Host "[7/10] Neural ODE: continuous C(t) toxicokinetic modeling (Python) ..." -ForegroundColor Yellow
Write-Host "       (LOO-CV over 20 chemicals; ~5-10 min)" -ForegroundColor DarkYellow
python "06_neural_ode_tk.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Step 6 (Neural ODE) failed.  Check PyTorch installation: pip install torch"
    Write-Host "  Continuing pipeline without Neural ODE results.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[7/10] Done.`n" -ForegroundColor Green
}

# --- Step 7: Explainable AI (SHAP) ---
Write-Host "[8/10] Explainable AI: SHAP analysis for Clint and BER (Python) ..." -ForegroundColor Yellow
python "07_xai_shap_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Step 7 (XAI/SHAP) failed.  Check shap installation: pip install shap"
    Write-Host "  Continuing pipeline without SHAP results.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[8/10] Done.`n" -ForegroundColor Green
}

# --- Step 8: Bayesian BER ---
Write-Host "[9/10] Bayesian BER: MC Dropout uncertainty analysis (Python) ..." -ForegroundColor Yellow
python "08_bayesian_ber.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Step 8 (Bayesian BER) failed.  Check PyTorch installation: pip install torch"
    Write-Host "  Continuing pipeline without Bayesian BER results.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[9/10] Done.`n" -ForegroundColor Green
}

# --- Step 9: In-vivo validation ---
Write-Host "[10/10] In-vivo validation vs. Wetmore2012 literature data (R) ..." -ForegroundColor Yellow
Rscript "09_invivo_validation.R"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Step 9 (in-vivo validation) failed.  Check httk / R installation."
    Write-Host "  Continuing pipeline without validation results.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[10/10] Done.`n" -ForegroundColor Green
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Pipeline complete!" -ForegroundColor Cyan
Write-Host "  Results in: $ProjectRoot\results\" -ForegroundColor Cyan
Write-Host "  Data in:    $ProjectRoot\data\" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan

Write-Host "Key outputs:" -ForegroundColor White
Write-Host "  results/aed_ber_full.csv               - AED + BER for all chemicals"        -ForegroundColor Gray
Write-Host "  results/ber_ranking_plot.png            - BER waterfall ranking"               -ForegroundColor Gray
Write-Host "  results/neural_ode_curves.png           - Neural ODE C(t) predictions"        -ForegroundColor Gray
Write-Host "  results/neural_ode_sparse_demo.png      - Sparse-data interpolation demo"     -ForegroundColor Gray
Write-Host "  results/shap_rf_beeswarm.png            - SHAP feature importance (RF Clint)" -ForegroundColor Gray
Write-Host "  results/shap_ber_beeswarm.png           - SHAP BER explainability"            -ForegroundColor Gray
Write-Host "  results/ber_credible_intervals.png      - Bayesian BER 90% credible bands"    -ForegroundColor Gray
Write-Host "  results/invivo_validation_scatter.png   - In-vivo Css validation scatter"     -ForegroundColor Gray
Write-Host "  results/invivo_validation_metrics.csv   - R2, RMSE, GMR, fold-error"          -ForegroundColor Gray
Write-Host ""
