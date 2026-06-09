# Hybrides KI-System fuer Toxikokinetik

Dieses Projekt entwickelt eine toxikokinetische Data-Science-Pipeline zur
Vorhersage fehlender `Clint`-Werte, zur Ableitung von AED/BER-Risikometriken und
zur Modellinterpretation. Der finale Modellansatz basiert auf einem erweiterten
httk-Datensatz:

- 777 parameterisierbare httk/ToxCast-Chemikalien
- 544 Chemikalien mit gemessenem `Clint`
- 500 Chemikalien fuer das Modelltraining
- 44 Chemikalien als Hold-out-Testset
- 233 Chemikalien ohne gemessenen `Clint`, fuer die `Clint` vorhergesagt wird

---

## Projektstruktur

```text
.
|-- data/
|   |-- all_777_chemicals.csv           Alle 777 httk/ToxCast-Chemikalien
|   |-- clint_predicted_233.csv         RF/GB-Vorhersagen fuer Chemikalien ohne gemessenen Clint
|   |-- clint_all777_final.csv          Finaler Clint-Datensatz: gemessen + vorhergesagt
|   |-- clint_predicted_233_rdkit.csv   RDKit-basierte Erweiterung der Vorhersagen
|   `-- smiles_cache_777.csv            SMILES-Cache fuer alle Chemikalien
|
|-- scripts/
|   |-- utils.py                        Gemeinsame Hilfsfunktionen
|   |-- 01_extract_httk_data.R          Datenextraktion aus httk
|   |-- 02b_rf_train_full544.py         Finales RF/GB-Clint-Modell: 500 Train / 44 Test
|   |-- 03_httk_pbtk_simulation.R       PBTK-Simulationen
|   |-- 04_reverse_dosimetry.R          Reverse TK / Monte-Carlo-AED
|   |-- 04b_aed_analysis.py             AED-Visualisierung
|   |-- 05_full_rtk_aed_ber.R           RTK + AED + BER fuer alle Chemikalien
|   |-- 06_neural_ode_tk.py             Neural ODE fuer C(t)-Kurven
|   |-- 07_xai_shap_analysis.py         SHAP-Analyse
|   |-- 08_bayesian_ber.py              Bayesianische BER-Unsicherheit
|   |-- 09_invivo_validation.R          In-vivo-Validierung
|   |-- 10_gcn_clint.py                 Explorative GCN-Analyse
|   |-- 11_gcn_all777.py                GCN/RF-Vergleich auf allen Chemikalien
|   `-- run_pipeline.ps1                PowerShell-Pipeline
|
|-- results/
|   |-- full544_metrics.txt             Hauptmetriken des 500/44-Modells
|   |-- full544_test_scatter.png        Hold-out-Test: gemessen vs. vorhergesagt
|   |-- full544_clint_distribution.png  Clint-Verteilung: gemessen vs. vorhergesagt
|   |-- aed_ber_full.csv                AED + BER fuer alle simulierten Chemikalien
|   |-- bayesian_ber.csv                Posterior-BER mit Unsicherheit
|   |-- ber_credible_intervals.png      BER-Glaubwuerdigkeitsintervalle
|   |-- shap_rf_beeswarm.png            SHAP fuer das Clint-Modell
|   |-- shap_rf_summary_bar.png         Globale Feature Importance
|   `-- ...                             Weitere Plots und CSV-Dateien
|
|-- requirements.txt
`-- README.md
```

---

## Voraussetzungen

### R

```r
install.packages(c("httk", "dplyr", "ggplot2", "randomForest"))
```

### Python

```powershell
pip install -r requirements.txt
```

Fuer neuronale Netze, GCN und Unsicherheitsanalyse:

```powershell
pip install torch rdkit shap
```

---

## Ausfuehrung

### Finales Clint-Modell

```powershell
cd scripts
python 02b_rf_train_full544.py
```

Dieses Skript ist die zentrale Modellstufe. Es trainiert Random Forest und
Gradient Boosting auf 500 Chemikalien, testet auf 44 Hold-out-Chemikalien und
erstellt Vorhersagen fuer 233 Chemikalien ohne gemessenen `Clint`.

### Gesamte Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
```

Hinweis: Die PowerShell-Pipeline enthaelt noch optionale explorative Altstufen.
Fuer die finale Bewertung des Projekts ist `02b_rf_train_full544.py` die
massgebliche Clint-Modellierung.

---

## Workflow-Uebersicht

| Stufe | Skript | Zweck | Wichtige Ausgabe |
|-------|--------|-------|------------------|
| 1 | `01_extract_httk_data.R` | httk-Datenextraktion | `all_777_chemicals.csv` |
| 2 | `02b_rf_train_full544.py` | RF/GB-Clint-Modell mit 500/44-Split | `full544_metrics.txt`, `clint_all777_final.csv` |
| 3 | `05_full_rtk_aed_ber.R` | RTK, AED und BER fuer alle Chemikalien | `aed_ber_full.csv` |
| 4 | `07_xai_shap_analysis.py` | SHAP-Erklaerbarkeit | `shap_rf_*.png`, `shap_*.csv` |
| 5 | `08_bayesian_ber.py` | Bayesianische BER-Unsicherheit | `bayesian_ber.csv`, `ber_credible_intervals.png` |
| 6 | `09_invivo_validation.R` | Literatur-/In-vivo-Validierung | Validierungsplots und CSVs |
| 7 | `06_neural_ode_tk.py` | Optionale C(t)-Kurvenmodellierung | `neural_ode_curves.png` |
| 8 | `10_gcn_clint.py`, `11_gcn_all777.py` | Optionale GCN-Vergleiche | `gcn_777_predictions.csv` |

---

## Finales Clint-Modell

**Ziel:** Vorhersage fehlender `Clint`-Werte fuer Chemikalien ohne Messwert.

**Datengrundlage:**

- `all_777_chemicals.csv`
- 544 Chemikalien mit `Human.Clint > 0`
- 233 Chemikalien mit fehlendem oder nicht verwertbarem `Clint`

**Modellierung:**

- Zielvariable: `log10(Clint)`
- Features: `MW`, `logP`, `Fup`
- Feature Engineering: 9 Features, u. a. `log10_MW`, `logP^2`, Interaktionen
- Modelle: Random Forest und Gradient Boosting
- Split: 500 Training / 44 Hold-out-Test
- Auswahl des besten Modells anhand der Testleistung auf Log-Skala

**Aktuelle Hauptmetriken aus `results/full544_metrics.txt`:**

| Modell | Set | R2 log10 | RMSE log10 | Spearman | GMFE |
|--------|-----|----------|------------|----------|------|
| RF | Train, n=500 | 0.7679 | 0.3391 | 0.9084 | 1.78x |
| GB | Train, n=500 | 0.7165 | 0.3748 | 0.8466 | 1.98x |
| RF | Test, n=44 | 0.2010 | 0.5928 | 0.4994 | 2.77x |
| GB | Test, n=44 | 0.0958 | 0.6306 | 0.4350 | 2.87x |

Der Random Forest ist damit das staerkere finale Modell auf dem Hold-out-Testset.

---

## Toxikokinetisches Modell

**Clint zu hepatischer Clearance:**

```text
CL_h = Q_H * Fup * Clint_L / (Q_H + Fup * Clint_L)
```

**IVIVE / Reverse Dosimetry:**

```text
AED = AC50 [mg/L] / Css pro Dosis
BER = AED / Exposition
```

**Risikopriorisierung:**

- Niedriger `BER` bedeutet hoehere Prioritaet.
- `BER < 1`: potenziell hohe Besorgnis.
- `1 <= BER < 10`: moderate Besorgnis.
- Hoehere BER-Werte sprechen fuer groesseren Abstand zwischen Bioaktivitaet und Exposition.

---

## Methodische Einordnung

Der finale Beitrag des Projekts ist eine hybride Pipeline auf Basis des
erweiterten httk-Datensatzes:

1. `Clint`-Modellierung auf 500 Trainingschemikalien
2. Hold-out-Validierung auf 44 Chemikalien
3. Imputation fuer 233 Chemikalien ohne gemessenen `Clint`
4. RTK/AED/BER-Risikopriorisierung
5. SHAP-Erklaerbarkeit und Bayesianische Unsicherheitsanalyse

---

## Referenzen

- [httk R-Paket (CRAN)](https://cran.r-project.org/package=httk)
- Wetmore, B. A. et al. (2012). Integration of dosimetry, exposure, and high-throughput screening data. *Toxicological Sciences*.
- Wambaugh, J. F. et al. (2019). Evaluating in vitro-in vivo extrapolation. *Toxicological Sciences*.
- Chen, R. T. Q. et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
- Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NIPS*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
