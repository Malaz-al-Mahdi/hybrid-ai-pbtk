# Hybrides KI-System fur Toxikokinetik -- Pilotstudie

Pilotstudie zur Vorhersage fehlender toxikokinetischer Parameter (Clint) mit
Random Forest und anschliessender PBTK-Simulation uber das R-Paket
[httk](https://cran.r-project.org/package=httk).

## Projektstruktur

```
.
├── data/                            # Generierte Datendateien (CSV)
│   ├── pilot_chemicals_full.csv     # 20 Chemikalien, komplette httk-Daten
│   ├── pilot_chemicals_masked.csv   # Gleiche Daten, Clint = NA
│   ├── rf_clint_predictions.csv     # LOO-CV Vorhersagen vs. Wahrheit
│   └── pilot_chemicals_imputed.csv  # Tabelle mit RF-imputiertem Clint
├── scripts/
│   ├── 01_extract_httk_data.R       # Schritt 1: Datenextraktion aus httk
│   ├── 02_rf_predict_clint.py       # Schritt 2: RF-Training + LOO-CV
│   ├── 03_httk_pbtk_simulation.R    # Schritt 3: PBTK nativ vs. RF-imputed
│   └── run_pipeline.ps1             # Gesamte Pipeline starten
├── results/                         # Ergebnisse, Plots, Metriken
│   ├── rf_loo_cv_metrics.txt
│   ├── rf_loo_cv_scatter.png
│   ├── pbtk_comparison.csv
│   ├── pbtk_comparison.png
│   └── pbtk_curves/                 # Konzentrations-Zeit-Kurven pro Chemikalie
├── requirements.txt                 # Python-Abhangigkeiten
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
