# Hybrides KI-System fur Toxikokinetik

Vorhersage fehlender toxikokinetischer Parameter (Clint) mittels Random Forest,
PBTK-Simulation, Neural ODEs, Explainable AI (SHAP), Bayesianischer
Risikoanalyse (BER) und Validierung gegen In-vivo-Daten -- alles integriert
uber das R-Paket [httk](https://cran.r-project.org/package=httk).

## Projektstruktur

```
.
├── data/
│   ├── pilot_chemicals_full.csv        # 20 Chemikalien, komplette httk-Daten
│   ├── pilot_chemicals_masked.csv      # Gleiche Daten, Clint = NA
│   ├── rf_clint_predictions.csv        # LOO-CV Vorhersagen vs. Wahrheit
│   ├── pilot_chemicals_imputed.csv     # RF-imputierter Clint
│   ├── toxcast_ac50_pilot.csv          # ToxCast AC50-Zusammenfassung
│   └── all_777_chemicals.csv           # Alle 777 parameterisierbaren Chemikalien
├── scripts/
│   ├── 01_extract_httk_data.R          # Schritt 1: Datenextraktion aus httk
│   ├── 02_rf_predict_clint.py          # Schritt 2: RF-Training + LOO-CV
│   ├── 03_httk_pbtk_simulation.R       # Schritt 3: PBTK nativ vs. RF-imputed
│   ├── 04_reverse_dosimetry.R          # Schritt 4: Reverse TK (MC-AED)
│   ├── 04b_aed_analysis.py             # Schritt 4b: AED-Visualisierung
│   ├── 05_full_rtk_aed_ber.R           # Schritt 5/6: RTK + AED + BER (777 Chem.)
│   ├── 06_neural_ode_tk.py             # Schritt 6: Neural ODE fur C(t)
│   ├── 07_xai_shap_analysis.py         # Schritt 7: Explainable AI (SHAP)
│   ├── 08_bayesian_ber.py              # Schritt 8: Bayesianische BER-Unsicherheit
│   ├── 09_invivo_validation.R          # Schritt 9: In-vivo-Validierung
│   └── run_pipeline.ps1                # Gesamte Pipeline (10 Schritte)
├── results/
│   ├── rf_loo_cv_metrics.txt
│   ├── rf_loo_cv_scatter.png
│   ├── pbtk_comparison.csv / .png
│   ├── pbtk_curves/                    # C(t)-Kurven pro Chemikalie
│   ├── aed_monte_carlo.csv / aed_mc_samples.csv
│   ├── aed_distributions.png / aed_paired_comparison.png
│   ├── aed_ber_full.csv                # AED + BER alle Chemikalien
│   ├── aed_ber_summary.csv             # Top-30 Hochrisikosubstanzen
│   ├── ber_ranking_plot.png            # BER-Wasserfall
│   ├── neural_ode_curves.png           # Neural ODE C(t): true vs. predicted
│   ├── neural_ode_sparse_demo.png      # Sparse-Daten-Demo
│   ├── neural_ode_metrics.csv          # LOO-CV: MAE, RMSE, R²
│   ├── shap_rf_beeswarm.png            # SHAP Beeswarm (RF Clint)
│   ├── shap_rf_summary_bar.png         # Globale Feature Importance
│   ├── shap_rf_dependence_*.png        # Dependence Plots
│   ├── shap_ber_beeswarm.png           # SHAP BER-Erklarbarkeit
│   ├── shap_rf_values.csv              # SHAP-Werte pro Chemikalie (RF)
│   ├── shap_ber_values.csv             # SHAP-Werte pro Chemikalie (BER)
│   ├── bayesian_ber.csv                # Posteriori-BER: Median + 90%-KI
│   ├── ber_credible_intervals.png      # Wasserfall + Kredibilitatsbander
│   ├── ber_posterior_top5.png          # Posteriori-Dichten Top-5
│   ├── clint_posterior_uncertainty.png # BNN Clint-Unsicherheit
│   ├── invivo_validation.csv           # Vorhergesagt vs. Literatur
│   ├── invivo_validation_metrics.csv   # R², RMSE, GMR, Fold-Error
│   ├── invivo_validation_scatter.png   # Log-Log Korrelationsplot
│   └── invivo_validation_residuals.png # Residuenanalyse
├── requirements.txt                    # Python-Abhangigkeiten
└── README.md
```

## Voraussetzungen

### R (>= 4.2)

**Wichtig:** `install.packages()` ist ein R-Befehl. Entweder in der R-Konsole ausfuhren *oder* aus PowerShell via `Rscript -e`.

In der R-Konsole:

```r
install.packages("httk")
```

Oder aus PowerShell:

```powershell
Rscript -e "install.packages('httk', repos='https://cloud.r-project.org')"
```

### Python (>= 3.8)

**Hinweis (Windows):** Falls `pip` nicht gefunden wird, nutze den Python Launcher:

```powershell
py -m pip install -r requirements.txt
```

Fur Schritte 6 und 8 (Neural ODE, Bayesian BER) wird zusatzlich PyTorch benotigt:

```powershell
py -m pip install torch
```

Fur Schritt 7 (Explainable AI / SHAP):

```powershell
py -m pip install shap
```

## Ausfuhrung

### Gesamte Pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
```

### Einzelschritte

```powershell
cd scripts

# Schritt 1: Daten aus httk extrahieren
Rscript 01_extract_httk_data.R

# Schritt 2: Random Forest trainieren + evaluieren
python 02_rf_predict_clint.py

# Schritt 3: PBTK-Simulationen durchfuhren
Rscript 03_httk_pbtk_simulation.R

# Schritt 4: Reverse Dosimetry (Monte Carlo AED)
Rscript 04_reverse_dosimetry.R

# Schritt 4b: AED-Visualisierung
python 04b_aed_analysis.py

# Schritt 5/6: Vollstandige RTK-Pipeline (777 Chemikalien -> AED -> BER)
Rscript 05_full_rtk_aed_ber.R

# Schritt 6: Neural ODE fur kontinuierliche C(t)-Kurven
python 06_neural_ode_tk.py

# Schritt 7: Explainable AI (SHAP)
python 07_xai_shap_analysis.py

# Schritt 8: Bayesianische BER-Unsicherheitsanalyse
python 08_bayesian_ber.py

# Schritt 9: In-vivo-Validierung
Rscript 09_invivo_validation.R
```

## Workflow-Beschreibung

### Schritt 1 -- Datenextraktion (`01_extract_httk_data.R`)

- Wahlt 20 Pilot-Chemikalien aus (inkl. Bisphenol A, CAS 80-05-7)
- Extrahiert MW, logP, Fup, Clint, Rblood2plasma aus httk
- Speichert vollstandige + maskierte (Clint=NA) Versionen

### Schritt 2 -- ML-Vorhersage (`02_rf_predict_clint.py`)

- Trainiert Random Forest auf log10(Clint) mit Features: MW, logP, Fup, Rblood2plasma
- Leave-One-Out Cross-Validation fur robuste Evaluation bei kleiner Stichprobe
- Berechnet RMSE, R-squared, Spearman-rho
- Exportiert imputierte Datentabelle mit Clint_RF und Clint_source

### Schritt 3 -- PBTK-Simulation (`03_httk_pbtk_simulation.R`)

- Ladt die imputierte Tabelle
- Fur jede Chemikalie: zwei PBTK-Laufe (28 Tage, 1 mg/kg/Tag)
  - **Track A**: httk-native Clint (Goldstandard)
  - **Track B**: RF-imputierter Clint
- Vergleicht Cmax, AUC, Css zwischen beiden Tracks
- Berechnet Fold-Change-Zusammenfassung
- Erstellt Konzentrations-Zeit-Kurven pro Chemikalie

### Schritt 4 -- Reverse Dosimetry (`04_reverse_dosimetry.R`)

- Implementiert **In-Vitro-to-In-Vivo-Extrapolation (IVIVE)** via Reverse Toxicokinetics
- Fur jede Pilot-Chemikalie: ToxCast-Daten aus `example.toxcast` (aktive Assays, `hitc == 1`)
- Konservative Zielkonzentration: **5. Perzentil der AC50-Verteilung** (sensibelster Endpunkt)
- `calc_mc_oral_equiv()` mit **1000 Monte-Carlo-Samples** pro Chemikalie
  - Propagiert Populationsvariabilitat in Clint, Fup, Korpergewicht, etc.
- Zwei Tracks: **httk-native** vs. **RF-imputed** (Clint-Skalierung)
- Ergebnisse: AED-Quantile (Median, 5., 95. Perzentil) in mg/kg/Tag

### Schritt 4b -- AED-Analyse (`04b_aed_analysis.py`)

- Paarvergleich native vs. RF-imputed AED (Scatter mit Fehlerbalken, 3-fold-Envelope)
- Populationsvariabilitats-Fanchart (5.--95. Perzentil-Band)
- Kumulative AED-Verteilung uber alle Pilot-Chemikalien
- Zusammenfassungsbericht als CSV

### Schritt 5/6 -- Vollstandige RTK + BER-Pipeline (`05_full_rtk_aed_ber.R`)

- Alle ~777 parameterisierbaren Chemikalien in httk
- Random-Forest-Imputation fehlender Clint-Werte
- AED via `calc_mc_oral_equiv()` mit 5. Perzentil AC50 als konservativem Bioaktivitatsziel
- Expositionsdaten: priorisiert NHANES (wambaugh2019.nhanes) → SEEM3 → example.seem
- BER = AED / Exposition; Klassifizierung HIGH/MODERATE/LOW/MINIMAL
- Output: `aed_ber_full.csv`, `aed_ber_summary.csv`, `ber_ranking_plot.png`

### Schritt 6 -- Neural ODE (`06_neural_ode_tk.py`)

- Modelliert kontinuierliche Plasma-Konzentrationsverlarufe C(t) mit einer gelernten ODE
- Architektur: ChemEncoder (MW, logP, Fup, Clint → Latent-Einbettung z) +
  ODEFunc (MLP-Vektorfeld dC/dt = f(C; z)) + differenzierbarer RK4-Integrator
- Training: Analytische 1-Kompartiment-Losungen als Ground Truth
- **Sparse-Data-Demo**: trainiert auf 5 unregelmasig verteilten Messpunkten,
  sagt die vollstandige 0--48-h-Kurve korrekt voraus
- Evaluation: Leave-One-Out-CV uber alle 20 Pilot-Chemikalien (MAE, RMSE, R²)

### Schritt 7 -- Explainable AI (`07_xai_shap_analysis.py`)

- **Teil A -- RF Clint**: SHAP TreeExplainer fur den Random-Forest-Clint-Pradiktor
  - Globale Bedeutung (Bar Chart), Beeswarm, Dependence Plots (logP, Fup, MW)
  - Identifiziert Wechselwirkungen zwischen Deskriptoren
- **Teil B -- BER**: Gradient-Boosted Regressor fur log10(BER), SHAP-Beeswarm
  - Lokale Erklarungen fur die Top-3 Hochrisikosubstanzen (Waterfall-artige Bar Charts)
  - Macht das BER-Ranking fur Regulierungsbehorden biologisch nachvollziehbar

### Schritt 8 -- Bayesianische BER-Analyse (`08_bayesian_ber.py`)

- Bayesianisches Neuronales Netz mit **Monte-Carlo-Dropout** (2000 Samples/Chemikalie)
- Propagiert Clint-Unsicherheit → AED → BER
- Liefert echte **Posteriori-Verteilungen** statt Punktschatzungen:
  - 90%-Kredibilitatsintervalle fur BER
  - Dichteplots fur Top-5-Hochrisikosubstanzen
- Output: `bayesian_ber.csv`, `ber_credible_intervals.png`, `ber_posterior_top5.png`

### Schritt 9 -- In-vivo-Validierung (`09_invivo_validation.R`)

- Vergleich von HTTK-Modellvorhersagen (3compartmentss + PBTK) mit publizierten
  In-vivo-Css-Werten aus Wetmore et al. (2012) und verwandten httk-Datensatzen
- Metriken: Pearson-R², RMSE(log10), Geometrischer Mittlerer Quotient (GMR),
  Anteil innerhalb 2-/3-/10-facher Abweichung vom Messwert
- Log-Log-Streudiagramm mit farbkodierter Fold-Error-Klassifikation
- Residuenanalyse (Histogramm + Normal-Q-Q-Plot)

## Pilot-Chemikalien (20)

| CAS         | Substanz           |
|-------------|--------------------|
| 80-05-7     | Bisphenol A        |
| 34256-82-1  | Acetochlor         |
| 99-71-8     | 4-sec-Butylphenol  |
| 58-08-2     | Caffeine           |
| 298-46-4    | Carbamazepine      |
| 2921-88-2   | Chlorpyrifos       |
| 138261-41-3 | Imidacloprid       |
| 87-86-5     | Pentachlorophenol  |
| 62-44-2     | Phenacetin         |
| 57-41-0     | Phenytoin          |
| 94-75-7     | 2,4-D              |
| 1912-24-9   | Atrazine           |
| 330-54-1    | Diuron             |
| 1071-83-6   | Glyphosate         |
| 15307-79-6  | Diclofenac sodium  |
| 137-26-8    | Thiram             |
| 52-68-6     | Trichlorfon        |
| 2104-64-5   | EPN                |
| 62-73-7     | Dichlorvos         |
| 56-72-4     | Coumaphos          |

## Referenzen

- [httk R-Paket (CRAN)](https://cran.r-project.org/package=httk)
- [httk GitHub Repository](https://github.com/USEPA/CompTox-ExpoCast-httk)
- [TAME Toolkit -- Toxicokinetic Modeling](https://uncsrp.github.io/Data-Analysis-Training-Modules/toxicokinetic-modeling.html)
- Breen et al. (2021) -- httk Review
- [PMC9887681 -- Hybrid PBTK/ML approaches](https://pmc.ncbi.nlm.nih.gov/articles/PMC9887681/)
- Chen, R. T. Q. et al. (2018) -- Neural Ordinary Differential Equations (NeurIPS)
- Lundberg, S. M. & Lee, S.-I. (2017) -- A Unified Approach to Interpreting Model Predictions (SHAP)
- Gal, Y. & Ghahramani, Z. (2016) -- Dropout as a Bayesian Approximation (ICML)
- Wetmore, B. A. et al. (2012) -- Integration of dosimetry, exposure, and high-throughput screening data in chemical toxicity assessment (Toxicol. Sci.)
- Wambaugh, J. F. et al. (2019) -- Evaluating in vitro-in vivo extrapolation of toxicokinetics (Toxicol. Sci.)
