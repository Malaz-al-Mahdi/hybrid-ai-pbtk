# Hybrides KI-System fuer Toxikokinetik

Vorhersage fehlender toxikokinetischer Parameter (Clint) mittels Random Forest,
PBTK-Simulation, Neural ODEs, Graph Convolutional Networks (GCN), Explainable AI (SHAP),
Bayesianischer Risikoanalyse (BER) und Validierung gegen Literaturwerte —
integriert ueber das R-Paket [httk](https://cran.r-project.org/package=httk).

---

## Projektstruktur

```
.
├── data/
│   ├── pilot_chemicals_full.csv        19 Pilotchemikalien mit vollstaendigen httk-Daten
│   ├── pilot_chemicals_masked.csv      Gleiche Daten, Clint = NA (fuer Blind-Tests)
│   ├── pilot_chemicals_imputed.csv     RF-imputierter Clint
│   ├── pilot_chemicals_gcn.csv         SMILES + GCN-Vorhersagen fuer Pilotchemikalien
│   ├── rf_clint_predictions.csv        LOO-CV Vorhersagen vs. Wahrheit
│   ├── toxcast_ac50_pilot.csv          ToxCast AC50-Zusammenfassung (Pilot)
│   ├── all_777_chemicals.csv           Alle 777 parameterisierbaren httk-Chemikalien
│   └── smiles_cache_777.csv            SMILES-Cache fuer alle 777 (wird automatisch erstellt)
│
├── scripts/
│   ├── utils.py                        Gemeinsame Hilfsfunktionen (PK, GCN, Metriken, SMILES)
│   ├── 01_extract_httk_data.R          Stufe 1:  Datenextraktion aus httk
│   ├── 02_rf_predict_clint.py          Stufe 2:  RF/GB LOO-CV + externe Validierung (777 Chem.)
│   ├── 03_httk_pbtk_simulation.R       Stufe 3:  PBTK nativ vs. RF-imputiert
│   ├── 04_reverse_dosimetry.R          Stufe 4:  Reverse TK (Monte-Carlo-AED)
│   ├── 04b_aed_analysis.py             Stufe 4b: AED-Visualisierung
│   ├── 05_full_rtk_aed_ber.R           Stufe 5:  RTK + AED + BER (777 Chemikalien)
│   ├── 06_neural_ode_tk.py             Stufe 6:  Neural ODE fuer C(t)-Kurven
│   ├── 07_xai_shap_analysis.py         Stufe 7:  SHAP global + BER + Ausreisseranalyse
│   ├── 08_bayesian_ber.py              Stufe 8:  Bayesianische BER-Unsicherheit (MC Dropout)
│   ├── 09_invivo_validation.R          Stufe 9:  In-vivo-Validierung (Wetmore 2012)
│   ├── 10_gcn_clint.py                 Stufe 10: GCN LOO-CV auf Pilotchemikalien
│   ├── 11_gcn_all777.py                Stufe 11: GCN + RF/GB auf allen 777 + BER-Vergleich
│   └── run_pipeline.ps1                Gesamte Pipeline (PowerShell, 11 Stufen)
│
├── results/                            Automatisch generierte Ausgaben
│   ├── rf_loo_cv_metrics.txt           RF/GB LOO-CV Metriken
│   ├── rf_loo_cv_scatter.png           Beobachtet vs. Vorhergesagt
│   ├── aed_ber_full.csv                AED + BER fuer alle Chemikalien (httk-nativ)
│   ├── gcn_777_predictions.csv         GCN + RF Vorhersagen (alle 777)
│   ├── gcn_777_metrics.txt             GCN vs. RF Metriken
│   ├── ber_all777.csv                  BER: GCN vs. RF vs. httk
│   ├── ber_all777_waterfall.png        BER-Rangfolge
│   ├── bayesian_ber.csv                Posteriori-BER: Median + 90%-KI
│   ├── ber_credible_intervals.png      Bayesianische BER-Glaubwuerdigkeitsbander
│   ├── neural_ode_curves.png           Neural ODE C(t): Vorhersage vs. Wahrheit
│   ├── shap_rf_beeswarm.png            SHAP Beeswarm (RF Clint)
│   ├── shap_rf_summary_bar.png         Globale Feature Importance
│   ├── clint_validation_scatter.png    Clint: ML-Vorhersage vs. Literatur (544 Chem.)
│   └── ...                             (weitere Plots und CSV-Dateien)
│
├── requirements.txt                    Python-Abhaengigkeiten
└── README.md
```

---

## Voraussetzungen

### R (>= 4.2)

```r
install.packages(c("httk", "dplyr", "ggplot2"))
```

Oder aus PowerShell:
```powershell
Rscript -e "install.packages('httk', repos='https://cloud.r-project.org')"
```

### Python (>= 3.10)

```powershell
pip install -r requirements.txt
```

Fuer neuronale Netze (Stufen 6, 8, 11, 13):
```powershell
pip install torch rdkit
```

---

## Ausfuehrung

### Gesamte Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
```

### Einzelne Stufen

```powershell
cd scripts

# Stufe 1: Daten aus httk extrahieren
Rscript 01_extract_httk_data.R

# Stufe 2: RF/GB Clint-Vorhersage + LOO-CV + externe Validierung (alle 777)
python 02_rf_predict_clint.py

# Stufe 3: PBTK-Simulationen
Rscript 03_httk_pbtk_simulation.R

# Stufe 4: Reverse Dosimetry (Monte Carlo AED) + Visualisierung
Rscript 04_reverse_dosimetry.R
python 04b_aed_analysis.py

# Stufe 5: Vollstaendige RTK-Pipeline (777 Chemikalien -> AED -> BER)
Rscript 05_full_rtk_aed_ber.R

# Stufe 6: Neural ODE fuer C(t)-Kurven
python 06_neural_ode_tk.py

# Stufe 7: SHAP global + BER-Erklaerbarkeit + Ausreisseranalyse (Tacrin, Phenylparaben)
python 07_xai_shap_analysis.py

# Stufe 8: Bayesianische BER-Unsicherheitsanalyse
python 08_bayesian_ber.py

# Stufe 9: In-vivo-Validierung
Rscript 09_invivo_validation.R

# Stufe 10: GCN LOO-CV auf Pilotchemikalien
python 10_gcn_clint.py

# Stufe 11: GCN + RF/GB auf allen 777 Chemikalien + BER-Vergleich (httk/GCN/RF)
python 11_gcn_all777.py
```

---

## Workflow-Uebersicht (11 Stufen)

| Stufe | Skript | Methode | Ausgabe |
|-------|--------|---------|---------|
| 1 | `01_extract_httk_data.R` | httk-Datenextraktion | `pilot_chemicals_full.csv` |
| 2 | `02_rf_predict_clint.py` | RF + GB, LOO-CV + externe Validierung | `rf_loo_cv_metrics.txt`, `clint_validation_scatter.png` |
| 3 | `03_httk_pbtk_simulation.R` | PBTK-Simulation | `pbtk_comparison.csv` |
| 4 | `04_reverse_dosimetry.R` + `04b_aed_analysis.py` | Monte-Carlo-IVIVE + Visualisierung | `aed_monte_carlo.csv` |
| 5 | `05_full_rtk_aed_ber.R` | RTK + BER (777) | `aed_ber_full.csv` |
| 6 | `06_neural_ode_tk.py` | Neural ODE | `neural_ode_curves.png` |
| 7 | `07_xai_shap_analysis.py` | SHAP global + BER + Ausreisser-Waterfall | `shap_rf_beeswarm.png`, `shap_outlier_waterfall_*.png` |
| 8 | `08_bayesian_ber.py` | BNN MC Dropout | `bayesian_ber.csv` |
| 9 | `09_invivo_validation.R` | In-vivo-Vergleich | `invivo_validation.csv` |
| 10 | `10_gcn_clint.py` | GCN LOO-CV (Piloten) | `gcn_loo_cv_metrics.txt` |
| 11 | `11_gcn_all777.py` | GCN + RF (777) + BER-Vergleich | `gcn_777_predictions.csv`, `ber_all777.csv` |

---

## Gemeinsame Hilfsfunktionen (`utils.py`)

Alle Python-Skripte importieren geteilten Code aus `scripts/utils.py`:

| Modul | Inhalt |
|-------|--------|
| **Pfade** | `ROOT`, `DATA`, `RESULTS`, Standard-Dateinamen |
| **PK-Funktionen** | `clint_uL_to_cl_h`, `calc_aed`, `calc_ber`, `concern_label` |
| **Feature Engineering** | `engineer_features` (9 Features: log10_MW, logP, logP², ...) |
| **Metriken** | `compute_metrics`, `print_metrics` (RMSE, R², GMFE, Fold-Errors) |
| **SMILES** | `pubchem_smiles`, `cir_smiles`, `load_smiles_cache` |
| **Molekuelgraphen** | `mol_to_graph`, `N_ATOM_FEAT` |
| **GCN-Modell** | `GCNLayer`, `MolGCN`, `train_gcn`, `predict_gcn` |

---

## Pilotchemikalien (19)

| CAS | Substanz | Clint [µL/min/10⁶] |
|---|---|---|
| 80-05-7 | Bisphenol A | 19.9 |
| 34256-82-1 | Acetochlor | 84.71 |
| 99-71-8 | 4-sec-Butylphenol | 19.03 |
| 58-08-2 | Coffein | 0.286 |
| 298-46-4 | Carbamazepin | 2.375 |
| 2921-88-2 | Chlorpyrifos | 2.60 |
| 138261-41-3 | Imidacloprid | 2.807 |
| 87-86-5 | Pentachlorphenol | 8.764 |
| 62-44-2 | Phenacetin | 9.346 |
| 57-41-0 | Phenytoin | 0.818 |
| 94-75-7 | 2,4-D | 0.0 |
| 1912-24-9 | Atrazin | 0.0 |
| 330-54-1 | Diuron | 12.15 |
| 15307-79-6 | Diclofenac-Natrium | 38.84 |
| 137-26-8 | Thiram | 816.0 |
| 52-68-6 | Trichlorfon | 31.7 |
| 2104-64-5 | EPN | 8.16 |
| 62-73-7 | Dichlorvos | 86.4 |
| 56-72-4 | Coumaphos | 31.7 |

---

## Toxikokinetisches Modell

**Clint-Vorhersage:**
- Deskriptoren: MW, logP, Fup (3 Rohdeskriptoren → 9 Feature-Engineering-Features)
- Modelle: Random Forest (1000 Baeume) + Gradient Boosting
- Evaluation: Leave-One-Out Cross-Validation

**IVIVE (In Vitro → In Vivo):**
- Well-Stirred-Modell: CL_h = Q_H × Fup × Clint_L / (Q_H + Fup × Clint_L)
- AED = AC50 [mg/L] / Css_pro_Dosis
- BER = AED / SEEM3-Exposition

**GCN:**
- Architektur: 40 Atom-Features → 128 → 64 → 32 → Mean Pooling → MLP → Clint
- Training: 19 Piloten; Vorhersage: 748 Chemikalien mit SMILES

---

## Referenzen

- [httk R-Paket (CRAN)](https://cran.r-project.org/package=httk)
- Wetmore, B. A. et al. (2012) — Integration of dosimetry, exposure, and high-throughput screening data. *Toxicol. Sci.*
- Wambaugh, J. F. et al. (2019) — Evaluating in vitro–in vivo extrapolation. *Toxicol. Sci.*
- Breen et al. (2021) — httk Review
- Chen, R. T. Q. et al. (2018) — Neural Ordinary Differential Equations. *NeurIPS*
- Lundberg, S. M. & Lee, S.-I. (2017) — A Unified Approach to Interpreting Model Predictions. *NIPS*
- Gal, Y. & Ghahramani, Z. (2016) — Dropout as a Bayesian Approximation. *ICML*
- Kipf, T. N. & Welling, M. (2017) — Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*
