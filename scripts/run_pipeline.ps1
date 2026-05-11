###############################################################################
# run_pipeline.ps1
# ---------------------------------------------------------------------------
# Vollstaendige toxikokinetische Forschungspipeline:
#
#   Stufe 1  (R)      – Datenextraktion aus httk
#   Stufe 2  (Python) – RF/GB Clint-Vorhersage + LOO-CV
#   Stufe 3  (R)      – PBTK-Simulationen (nativ vs. RF-imputiert)
#   Stufe 4  (R)      – Reverse Dosimetry: ToxCast AC50 -> AED (Monte Carlo)
#   Stufe 4b (Python) – AED-Visualisierung
#   Stufe 5  (R)      – Vollstaendige RTK-Pipeline: 777 Chemikalien -> AED -> BER
#   Stufe 6  (Python) – Neural ODE fuer kontinuierliche C(t)-Kurven
#   Stufe 7  (Python) – Explainable AI (SHAP global + Ausreisseranalyse)
#   Stufe 8  (Python) – Bayesianische BER-Unsicherheitsanalyse (MC Dropout)
#   Stufe 9  (R)      – In-vivo-Validierung (Wetmore 2012)
#   Stufe 11 (Python) – GCN LOO-CV auf Pilotchemikalien
#   Stufe 13 (Python) – GCN + RF/GB auf allen 777 Chemikalien + BER-Vergleich
#
# Zusammengefasste Stufen (gegenueber frueherer 14-Stufen-Version):
#   02 + 10 -> Stufe 2  (Clint LOO-CV + externe Validierung)
#   07 + 12 -> Stufe 7  (SHAP global + Ausreisser-Waterfall)
#   13 + 14 -> Stufe 13 (GCN/RF Vorhersagen + BER-Berechnung)
#
# Ausfuehrung:
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
#
# Voraussetzungen:
#   - R mit httk-Paket (install.packages("httk"))
#   - Python 3.10+ mit: pip install -r requirements.txt
###############################################################################

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$env:R_LIBS_USER = Join-Path $HOME "Documents\R\win-library\4.5"

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "  TK-Hybrid Pipeline (11 Stufen)" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan
Write-Host "  R user library: $env:R_LIBS_USER`n" -ForegroundColor DarkCyan

# ── Stufe 1: Datenextraktion ──────────────────────────────────────────────────
Write-Host "[1/11] Datenextraktion aus httk (R) ..." -ForegroundColor Yellow
Set-Location "$ScriptDir"
Rscript "01_extract_httk_data.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stufe 1 fehlgeschlagen. R / httk-Installation pruefen."
    exit 1
}
Write-Host "[1/11] Fertig.`n" -ForegroundColor Green

# ── Stufe 2: RF/GB LOO-CV + Externe Validierung ───────────────────────────────
Write-Host "[2/11] RF/GB Clint-Vorhersage + LOO-CV + Validierung (Python) ..." -ForegroundColor Yellow
Write-Host "       (Trainiert auf 19 Piloten; validiert auf alle 777 httk-Chemikalien)" -ForegroundColor DarkYellow
python "02_rf_predict_clint.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stufe 2 fehlgeschlagen. Python / scikit-learn pruefen."
    exit 1
}
Write-Host "[2/11] Fertig.`n" -ForegroundColor Green

# ── Stufe 3: PBTK-Simulation ──────────────────────────────────────────────────
Write-Host "[3/11] PBTK-Simulationen (R + httk) ..." -ForegroundColor Yellow
Rscript "03_httk_pbtk_simulation.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stufe 3 fehlgeschlagen. R / httk parameterize_pbtk pruefen."
    exit 1
}
Write-Host "[3/11] Fertig.`n" -ForegroundColor Green

# ── Stufe 4: Reverse Dosimetry (Monte Carlo AED) ─────────────────────────────
Write-Host "[4/11] Reverse Dosimetry: ToxCast AC50 -> AED (R + httk MC) ..." -ForegroundColor Yellow
Rscript "04_reverse_dosimetry.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stufe 4 fehlgeschlagen. httk calc_mc_oral_equiv / ToxCast-Daten pruefen."
    exit 1
}
python "04b_aed_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "AED-Visualisierung (04b) fehlgeschlagen."
}
Write-Host "[4/11] Fertig.`n" -ForegroundColor Green

# ── Stufe 5: Vollstaendige RTK-Pipeline (777 Chemikalien) ────────────────────
Write-Host "[5/11] Vollstaendige RTK-Pipeline: 777 Chemikalien -> AED -> BER (R) ..." -ForegroundColor Yellow
Write-Host "       (Dauert 30-60 Min. fuer 777 Chemikalien)" -ForegroundColor DarkYellow
Rscript "05_full_rtk_aed_ber.R"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Stufe 5 fehlgeschlagen. results/aed_ber_full.csv pruefen."
    exit 1
}
Write-Host "[5/11] Fertig.`n" -ForegroundColor Green

# ── Stufe 6: Neural ODE ───────────────────────────────────────────────────────
Write-Host "[6/11] Neural ODE: kontinuierliche C(t)-TK-Modellierung (Python) ..." -ForegroundColor Yellow
Write-Host "       (LOO-CV ueber 20 Chemikalien; ~5-10 Min.)" -ForegroundColor DarkYellow
python "06_neural_ode_tk.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stufe 6 (Neural ODE) fehlgeschlagen. PyTorch pruefen: pip install torch"
    Write-Host "  Pipeline wird ohne Neural-ODE-Ergebnisse fortgesetzt.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[6/11] Fertig.`n" -ForegroundColor Green
}

# ── Stufe 7: Explainable AI (SHAP global + Ausreisseranalyse) ────────────────
Write-Host "[7/11] SHAP: globale Feature-Importance + Ausreisseranalyse (Python) ..." -ForegroundColor Yellow
Write-Host "       (Section A: RF global | Section B: BER | Section C: Tacrine/Phenylparaben)" -ForegroundColor DarkYellow
python "07_xai_shap_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stufe 7 (XAI/SHAP) fehlgeschlagen. shap pruefen: pip install shap"
    Write-Host "  Pipeline wird ohne SHAP-Ergebnisse fortgesetzt.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[7/11] Fertig.`n" -ForegroundColor Green
}

# ── Stufe 8: Bayesianische BER-Unsicherheitsanalyse ──────────────────────────
Write-Host "[8/11] Bayesianische BER: MC Dropout Unsicherheitsanalyse (Python) ..." -ForegroundColor Yellow
python "08_bayesian_ber.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stufe 8 (Bayesian BER) fehlgeschlagen. PyTorch pruefen: pip install torch"
    Write-Host "  Pipeline wird ohne Bayesian-BER-Ergebnisse fortgesetzt.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[8/11] Fertig.`n" -ForegroundColor Green
}

# ── Stufe 9: In-vivo-Validierung ─────────────────────────────────────────────
Write-Host "[9/11] In-vivo-Validierung vs. Wetmore2012 (R) ..." -ForegroundColor Yellow
Rscript "09_invivo_validation.R"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stufe 9 (In-vivo-Validierung) fehlgeschlagen. httk / R pruefen."
    Write-Host "  Pipeline wird ohne Validierungsergebnisse fortgesetzt.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[9/11] Fertig.`n" -ForegroundColor Green
}

# ── Stufe 10: GCN LOO-CV auf Pilotchemikalien ────────────────────────────────
Write-Host "[10/11] GCN LOO-CV auf Pilotchemikalien (Python) ..." -ForegroundColor Yellow
Write-Host "        (Benoetigt: rdkit + torch; SMILES-Download ~2 Min.)" -ForegroundColor DarkYellow
python "10_gcn_clint.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stufe 10 (GCN LOO-CV) fehlgeschlagen. Pruefen: pip install rdkit torch"
    Write-Host "  Pipeline wird ohne GCN-LOO-CV-Ergebnisse fortgesetzt.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[10/11] Fertig.`n" -ForegroundColor Green
}

# ── Stufe 11: GCN + RF/GB auf allen 777 Chemikalien + BER ────────────────────
Write-Host "[11/11] GCN + RF/GB auf allen 777 Chemikalien + BER-Berechnung (Python) ..." -ForegroundColor Yellow
Write-Host "        (SMILES aus Cache; GCN-Training ~1 Min.; BER fuer alle 3 Clint-Quellen)" -ForegroundColor DarkYellow
python "11_gcn_all777.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stufe 11 (GCN + BER) fehlgeschlagen."
    Write-Host "  Pipeline wird ohne GCN-Ergebnisse fortgesetzt.`n" -ForegroundColor DarkYellow
} else {
    Write-Host "[11/11] Fertig.`n" -ForegroundColor Green
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Pipeline vollstaendig! (11 Stufen)" -ForegroundColor Cyan
Write-Host "  Ergebnisse: $ProjectRoot\results\" -ForegroundColor Cyan
Write-Host "  Daten:      $ProjectRoot\data\" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan

Write-Host "Wichtige Ausgaben:" -ForegroundColor White
Write-Host "  results/rf_loo_cv_metrics.txt           - RF/GB LOO-CV Metriken (Stufe 2)"          -ForegroundColor Gray
Write-Host "  results/clint_validation_scatter.png    - Externe Validierung 777 Chemikalien (Stufe 2)" -ForegroundColor Gray
Write-Host "  results/aed_ber_full.csv                - AED + BER httk-nativ (Stufe 5)"           -ForegroundColor Gray
Write-Host "  results/shap_rf_beeswarm.png            - SHAP Feature-Importance (Stufe 7)"        -ForegroundColor Gray
Write-Host "  results/shap_outlier_waterfall_*.png    - Ausreisser-Erklaerung (Stufe 7)"          -ForegroundColor Gray
Write-Host "  results/ber_credible_intervals.png      - Bayesianische BER 90%-KI (Stufe 8)"       -ForegroundColor Gray
Write-Host "  results/gcn_777_predictions.csv         - GCN + RF Vorhersagen 777 (Stufe 11)"      -ForegroundColor Gray
Write-Host "  results/ber_all777.csv                  - BER: GCN vs. RF vs. httk (Stufe 11)"      -ForegroundColor Gray
Write-Host "  results/ber_all777_waterfall.png        - BER-Wasserfall (Stufe 11)"                -ForegroundColor Gray
Write-Host "  results/neural_ode_curves.png           - Neural ODE C(t) (Stufe 6)"               -ForegroundColor Gray
Write-Host ""
