"""
12_shap_outlier_analysis.py
----------------------------
SHAP-Analyse: Warum liegt das Modell bei Tacrin und Phenylparaben so stark daneben?

Wissenschaftliche Fragestellung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das RF/GB-Modell (Step 2) sagt fuer einige Chemikalien Clint um Faktoren >1000
falsch vorher. Die Top-Ausreisser aus der externen Validierung (Step 10) sind:
  - Tacrine          Lit: 1000 uL/min  |  GMFE: ~97000x
  - Phenylparaben    Lit:  865 uL/min  |  GMFE: ~79000x
  - Acibenzolar      Lit:  863 uL/min  |  GMFE: ~141000x

SHAP erklaert per-Chemikalie, WELCHE Features das Modell in die falsche Richtung
gedraengt haben und WARUM diese Features irreführend sind.

Vorgehen
~~~~~~~~
  1. Trainiere RF/GB auf allen 19 Pilotchemikalien (voller Datensatz)
  2. Wende Modell auf alle 777 httk-Chemikalien an
  3. SHAP TreeExplainer -> lokale Erklaerungen fuer jede Chemikalie
  4. Visualisierung:
     A) Global: Beeswarm + Bar-Chart ueber alle 544 Chemikalien
     B) Lokal: Waterfall-Plots fuer die 5 schlechtesten Vorhergesagten
     C) Vergleich: Tacrine/Phenylparaben vs. gut vorhergesagte Chemikalien
     D) Feature-Scatter: Warum irreführen MW/logP/Fup fuer reaktive Molekuele?

Outputs
~~~~~~~
  results/shap_outlier_global_bar.png       Globale Feature-Importance (alle 544)
  results/shap_outlier_beeswarm.png         Beeswarm ueber alle Chemikalien
  results/shap_outlier_waterfall_*.png      Waterfall fuer Top-5-Ausreisser
  results/shap_outlier_comparison.png       Ausreisser vs. gut vorhergesagte
  results/shap_outlier_values.csv           Alle SHAP-Werte + Metadaten
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import shap
except ImportError:
    sys.exit("ERROR: shap fehlt.  pip install shap")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy.stats as stats

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

PILOT_CSV = DATA / "pilot_chemicals_full.csv"
FULL_CSV  = DATA / "all_777_chemicals.csv"
LOO_CSV   = DATA / "rf_clint_predictions.csv"

EPSILON = 1e-3

print("=" * 65)
print("Step 12 - SHAP Ausreisser-Analyse: Tacrin & Phenylparaben")
print("=" * 65)


# ── 1. Feature Engineering (identisch zu Step 2) ─────────────────────────────

FEATURE_NAMES = [
    "log10_MW", "logP", "logP^2", "log10_Fup", "sqrt_Fup",
    "MW_x_logP", "MW_x_Fup", "logP_x_Fup", "MW",
]

def engineer_features(df_in: pd.DataFrame) -> np.ndarray:
    mw   = np.clip(pd.to_numeric(df_in["MW"],   errors="coerce").fillna(300).values, 1.0, None)
    logp = pd.to_numeric(df_in["logP"], errors="coerce").fillna(2.0).values
    fup  = np.clip(pd.to_numeric(df_in["Fup"],  errors="coerce").fillna(0.1).values, 1e-6, 1.0)
    return np.column_stack([
        np.log10(mw),
        logp,
        logp ** 2,
        np.log10(fup + 1e-6),
        np.sqrt(fup),
        mw * logp,
        mw * fup,
        logp * fup,
        mw,
    ])


# ── 2. Trainingsdaten (19 Pilotchemikalien) ───────────────────────────────────

pilot    = pd.read_csv(PILOT_CSV)
df_train = pilot.dropna(subset=["Clint"]).copy()
df_train["Fup"] = df_train["Fup"].clip(lower=1e-6)
X_train  = engineer_features(df_train)
y_train  = np.log10(df_train["Clint"].values + EPSILON)

print(f"\nTrainingsdaten: {len(df_train)} Pilotchemikalien")
print(f"Clint-Bereich: {df_train['Clint'].min():.1f} - {df_train['Clint'].max():.1f}")


# ── 3. Bestes Modell trainieren ───────────────────────────────────────────────

best_model_name = "GradientBoosting"
if LOO_CSV.exists():
    loo_df = pd.read_csv(LOO_CSV)
    if "model" in loo_df.columns:
        best_model_name = loo_df["model"].iloc[0]

print(f"Modell: {best_model_name}")

# Fuer SHAP brauchen wir das Rohmodell OHNE Pipeline-Wrapper (TreeExplainer)
imputer  = SimpleImputer(strategy="median")
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(imputer.fit_transform(X_train))

if best_model_name == "GradientBoosting":
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=2,
        subsample=0.8, min_samples_leaf=2, random_state=42,
    )
else:
    model = RandomForestRegressor(
        n_estimators=1000, max_features=None,
        min_samples_leaf=1, random_state=42, n_jobs=-1,
    )

model.fit(X_tr_sc, y_train)
y_pred_train = model.predict(X_tr_sc)
print(f"Training R^2 (log10): {r2_score(y_train, y_pred_train):.4f}")


# ── 4. Alle 777 Chemikalien vorbereiten ───────────────────────────────────────

full = pd.read_csv(FULL_CSV)
full = full.rename(columns={
    "Human.Clint":           "Clint",
    "Human.Funbound.plasma": "Fup",
})
full["Clint"] = pd.to_numeric(full["Clint"], errors="coerce")
full["Fup"]   = pd.to_numeric(full["Fup"],   errors="coerce").clip(lower=1e-6)
full["MW"]    = pd.to_numeric(full["MW"],     errors="coerce")
full["logP"]  = pd.to_numeric(full["logP"],   errors="coerce")

val = full.dropna(subset=["Clint", "MW", "logP", "Fup"]).copy()
val = val[val["Clint"] > 0].reset_index(drop=True)

X_val_raw = engineer_features(val)
X_val_sc  = scaler.transform(imputer.transform(X_val_raw))

pred_log = model.predict(X_val_sc)
val["Clint_pred"]  = 10 ** pred_log - EPSILON
val["log10_lit"]   = np.log10(val["Clint"] + EPSILON)
val["log10_pred"]  = pred_log
val["fold_error"]  = 10 ** np.abs(val["log10_lit"] - val["log10_pred"])

pilot_cas = set(df_train["CAS"].astype(str).str.strip())
val["in_pilot"] = val["CAS"].astype(str).str.strip().isin(pilot_cas)

print(f"\nValidierungsset: {len(val)} Chemikalien mit gemessenem Clint")
print(f"Worst-case Ausreisser (Top 10 nach Fold-Error):")
worst10 = val.nlargest(10, "fold_error")[["Compound","Clint","Clint_pred","fold_error","logP","MW","Fup"]]
print(worst10.to_string(index=False))


# ── 5. SHAP TreeExplainer ────────────────────────────────────────────────────

print("\nBerechne SHAP-Werte fuer alle Validierungschemikalien ...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_val_sc)   # shap.Explanation (n_val x n_features)

# SHAP-Werte als DataFrame
shap_df = pd.DataFrame(
    shap_values.values,
    columns=[f"SHAP_{f}" for f in FEATURE_NAMES],
)
shap_df.insert(0, "CAS",          val["CAS"].values)
shap_df.insert(1, "Compound",     val["Compound"].values)
shap_df.insert(2, "log10_lit",    val["log10_lit"].values)
shap_df.insert(3, "log10_pred",   val["log10_pred"].values)
shap_df.insert(4, "fold_error",   val["fold_error"].values)
shap_df.insert(5, "in_pilot",     val["in_pilot"].values)
for f in ["MW", "logP", "Fup", "Clint"]:
    shap_df[f] = val[f].values

shap_df.to_csv(RESULTS / "shap_outlier_values.csv", index=False)
print(f"SHAP-Werte gespeichert: results/shap_outlier_values.csv")

base_value = float(shap_values.base_values[0])
print(f"SHAP Basiswert (Trainingsdurchschnitt log10 Clint): {base_value:.3f}")


# ── 6. Ziel-Chemikalien identifizieren ───────────────────────────────────────

# Spezifische Ausreisser
TARGET_CHEMS = ["Tacrine", "Phenylparaben", "Acibenzolar"]
# Gut vorhergesagte Chemikalien (fold_error <= 1.5, externe Chemikalien)
good = val[~val["in_pilot"] & (val["fold_error"] <= 1.5)].nsmallest(5, "fold_error")

print(f"\nZiel-Ausreisser:")
for name in TARGET_CHEMS:
    row = val[val["Compound"].str.contains(name, case=False, na=False)]
    if len(row):
        r = row.iloc[0]
        print(f"  {r['Compound'][:35]:<35} Clint_lit={r['Clint']:.0f}  "
              f"Clint_pred={r['Clint_pred']:.2f}  FE={r['fold_error']:.0f}x  "
              f"logP={r['logP']:.2f}  MW={r['MW']:.0f}  Fup={r['Fup']:.4f}")
    else:
        print(f"  {name}: NICHT in Validierungsset gefunden")

print(f"\nGut vorhergesagte Referenzchemikalien:")
print(good[["Compound","Clint","Clint_pred","fold_error","logP","MW","Fup"]].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════════════════════════════════════

# ── 7A: Globale Feature-Importance (Bar) ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
order         = np.argsort(mean_abs_shap)

ax = axes[0]
colors = plt.cm.RdBu_r(np.linspace(0.15, 0.85, len(FEATURE_NAMES)))
bars = ax.barh(
    [FEATURE_NAMES[i] for i in order],
    [mean_abs_shap[i] for i in order],
    color=[colors[k] for k in range(len(order))],
    edgecolor="k", linewidth=0.4,
)
ax.set_xlabel("Mean |SHAP-Wert|  (Einfluss auf log10 Clint)", fontsize=10)
ax.set_title(f"Globale Feature-Importance\n{best_model_name} auf {len(val)} Chemikalien", fontsize=10)
ax.grid(axis="x", alpha=0.3)

# ── 7B: Beeswarm ─────────────────────────────────────────────────────────────
ax = axes[1]
# Manueller Beeswarm: jeder Punkt ist eine Chemikalie
for k, feat_idx in enumerate(order):
    shap_vals  = shap_values.values[:, feat_idx]
    feat_vals  = X_val_sc[:, feat_idx]
    # Normalise Farbe nach Feature-Wert
    norm_fv    = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-9)
    y_jitter   = np.random.default_rng(feat_idx).uniform(-0.35, 0.35, len(shap_vals))
    sc = ax.scatter(
        shap_vals, np.full(len(shap_vals), k) + y_jitter,
        c=norm_fv, cmap="RdBu_r", s=8, alpha=0.5, linewidths=0,
    )

ax.set_yticks(range(len(FEATURE_NAMES)))
ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=9)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("SHAP-Wert (Auswirkung auf log10 Clint-Vorhersage)", fontsize=10)
ax.set_title("Beeswarm: Jeder Punkt = 1 Chemikalie\nRot=hoher Feature-Wert, Blau=niedriger", fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.colorbar(sc, ax=ax, label="Normierter Feature-Wert", fraction=0.03, pad=0.04)

plt.tight_layout()
plt.savefig(RESULTS / "shap_outlier_global_bar.png", dpi=150)
plt.close()
print("\nSaved: results/shap_outlier_global_bar.png")


# ── 7C: Waterfall-Plots fuer Top-5-Ausreisser ────────────────────────────────
top5_out = val[~val["in_pilot"]].nlargest(5, "fold_error")

for rank, (_, row_val) in enumerate(top5_out.iterrows()):
    chem_idx = val.index.get_loc(row_val.name)
    shap_chem = shap_values.values[chem_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(np.abs(shap_chem))[::-1]

    colors_wf = ["#d73027" if v > 0 else "#4575b4" for v in shap_chem[sorted_idx]]
    bars = ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        shap_chem[sorted_idx],
        color=colors_wf, edgecolor="k", linewidth=0.4,
    )
    ax.axvline(0, color="k", lw=1.2)

    # Feature-Werte annotieren
    raw_feats = X_val_raw[chem_idx]
    for bar_idx, feat_idx in enumerate(sorted_idx):
        val_raw = raw_feats[feat_idx]
        shap_v  = shap_chem[feat_idx]
        x_pos   = shap_v + (0.01 if shap_v >= 0 else -0.01)
        ax.text(x_pos, bar_idx,
                f"  {FEATURE_NAMES[feat_idx]}={val_raw:.3g}",
                va="center", fontsize=8,
                ha="left" if shap_v >= 0 else "right")

    compound  = str(row_val["Compound"])
    clint_lit = float(row_val["Clint"])
    clint_pred= float(row_val["Clint_pred"])
    fe        = float(row_val["fold_error"])
    pred_log_v= float(row_val["log10_pred"])

    ax.set_xlabel("SHAP-Wert (Beitrag zur log10 Clint-Vorhersage)", fontsize=10)
    ax.set_title(
        f"SHAP Waterfall: {compound}\n"
        f"Literatur Clint={clint_lit:.0f}  |  Vorhergesagt={clint_pred:.2f}  |  "
        f"Fold-Error={fe:.0f}x\n"
        f"Basiswert={base_value:.2f}  +  SHAP-Summe={shap_chem.sum():.2f}  "
        f"=  Vorhersage={pred_log_v:.2f}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)

    # Erklaerungsbox
    mw_v   = float(row_val["MW"])
    logp_v = float(row_val["logP"])
    fup_v  = float(row_val["Fup"])
    textbox = (
        f"MW={mw_v:.0f}  logP={logp_v:.2f}  Fup={fup_v:.4f}\n"
        f"Warum falsch: Das Modell kennt keine\n"
        f"reaktiven Gruppen / Enzymspezifitaet"
    )
    ax.text(0.98, 0.02, textbox, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.9))

    safe_name = compound.replace(" ", "_").replace("/", "_")[:25]
    out_path  = RESULTS / f"shap_outlier_waterfall_{rank+1}_{safe_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path.name}")


# ── 7D: Vergleichsplot Ausreisser vs. gut vorhergesagte Chems ─────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

COMPARE_CHEMS = {
    "Ausreisser": [],
    "Gut":        [],
}

for name in TARGET_CHEMS:
    row = val[val["Compound"].str.contains(name, case=False, na=False)]
    if len(row):
        idx = val.index.get_loc(row.index[0])
        COMPARE_CHEMS["Ausreisser"].append((row.iloc[0]["Compound"], idx))

for _, row_g in good.iterrows():
    idx = val.index.get_loc(row_g.name)
    COMPARE_CHEMS["Gut"].append((row_g["Compound"], idx))

all_compare = COMPARE_CHEMS["Ausreisser"] + COMPARE_CHEMS["Gut"]
n_compare   = len(all_compare)

for k, (chem_name, chem_idx) in enumerate(all_compare[:6]):
    row_data   = val.iloc[chem_idx]
    shap_c     = shap_values.values[chem_idx]
    raw_feats  = X_val_raw[chem_idx]
    fe         = float(row_data["fold_error"])
    is_outlier = k < len(COMPARE_CHEMS["Ausreisser"])

    ax = fig.add_subplot(gs[k // 3, k % 3])
    sorted_idx = np.argsort(np.abs(shap_c))[::-1]

    bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_c[sorted_idx]]
    ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        shap_c[sorted_idx],
        color=bar_colors, edgecolor="k", linewidth=0.3,
    )
    ax.axvline(0, color="k", lw=0.8)

    border_color = "#d73027" if is_outlier else "#2e7d32"
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.5)

    label  = "AUSREISSER" if is_outlier else "GUT VORHERGESAGT"
    title  = (f"[{label}] {str(chem_name)[:25]}\n"
              f"Lit={row_data['Clint']:.0f}  Pred={row_data['Clint_pred']:.1f}  "
              f"FE={fe:.1f}x")
    ax.set_title(title, fontsize=8, color=border_color, fontweight="bold")
    ax.set_xlabel("SHAP", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="both", labelsize=7)

fig.suptitle(
    "SHAP Vergleich: Ausreisser vs. gut vorhergesagte Chemikalien\n"
    "Rot=Beitrag erhoehend, Blau=Beitrag erniedrigend | Roter Rahmen=Ausreisser",
    fontsize=11, y=1.01,
)
plt.savefig(RESULTS / "shap_outlier_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/shap_outlier_comparison.png")


# ── 7E: Feature-Space-Plot: wo liegen die Ausreisser? ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

feature_pairs = [("logP", "log10_MW"), ("log10_Fup", "logP"), ("MW_x_logP", "log10_Fup")]

for ax, (fx, fy) in zip(axes, feature_pairs):
    xi = FEATURE_NAMES.index(fx)
    yi = FEATURE_NAMES.index(fy)

    # Alle externen Chemikalien (grau, Groesse ~ log fold error)
    ext_mask = ~val["in_pilot"].values
    fe_log   = np.log10(val["fold_error"].values + 1)

    sc = ax.scatter(
        X_val_sc[ext_mask, xi], X_val_sc[ext_mask, yi],
        c=fe_log[ext_mask], cmap="YlOrRd", s=20, alpha=0.5,
        vmin=0, vmax=4, linewidths=0, label="_nolegend_"
    )
    plt.colorbar(sc, ax=ax, label="log10(fold-error)", fraction=0.04)

    # Pilotchemikalien gelb markieren
    pilot_mask = val["in_pilot"].values
    ax.scatter(X_val_sc[pilot_mask, xi], X_val_sc[pilot_mask, yi],
               c="gold", s=80, edgecolors="k", linewidths=0.8,
               zorder=4, label="Pilot (Training)")

    # Ausreisser rot beschriften
    for name in TARGET_CHEMS:
        row = val[val["Compound"].str.contains(name, case=False, na=False)]
        if len(row):
            idx = val.index.get_loc(row.index[0])
            ax.scatter(X_val_sc[idx, xi], X_val_sc[idx, yi],
                       c="red", s=150, edgecolors="k", linewidths=1.2,
                       marker="*", zorder=5, label=str(row.iloc[0]["Compound"])[:15])
            ax.annotate(str(row.iloc[0]["Compound"])[:12],
                        (X_val_sc[idx, xi], X_val_sc[idx, yi]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points",
                        color="darkred", fontweight="bold")

    ax.set_xlabel(f"{fx} (skaliert)", fontsize=9)
    ax.set_ylabel(f"{fy} (skaliert)", fontsize=9)
    ax.set_title(f"Feature-Raum: {fx} vs. {fy}\nFarbe = Fold-Error", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)

plt.suptitle(
    "Feature-Raum der 544 Chemikalien\n"
    "Ausreisser (Tacrine, Phenylparaben) im Kontext des Trainingsraums (gelb)",
    fontsize=11, y=1.01,
)
plt.savefig(RESULTS / "shap_outlier_feature_space.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/shap_outlier_feature_space.png")


# ── 8. Erklaerungstext Tacrine / Phenylparaben ───────────────────────────────

print("\n" + "=" * 65)
print("MECHANISTISCHE ERKLAERUNG DER AUSREISSER")
print("=" * 65)

for name in TARGET_CHEMS:
    row = val[val["Compound"].str.contains(name, case=False, na=False)]
    if not len(row):
        continue
    r       = row.iloc[0]
    idx     = val.index.get_loc(row.index[0])
    shap_c  = shap_values.values[idx]
    rf_raw  = X_val_raw[idx]

    print(f"\n{r['Compound']}")
    print(f"  Gemessen:      Clint = {r['Clint']:.0f} uL/min/10^6 cells (sehr hoch)")
    print(f"  Vorhergesagt:  Clint = {r['Clint_pred']:.2f}  (Fold-Error = {r['fold_error']:.0f}x)")
    print(f"  Deskriptoren:  MW={r['MW']:.0f}  logP={r['logP']:.2f}  Fup={r['Fup']:.4f}")
    print(f"  SHAP-Haupttreiber:")
    order_c = np.argsort(np.abs(shap_c))[::-1]
    for feat_idx in order_c[:4]:
        direction = "erhoehend" if shap_c[feat_idx] > 0 else "erniedrigend"
        print(f"    {FEATURE_NAMES[feat_idx]:20s}: SHAP={shap_c[feat_idx]:+.3f}  "
              f"({direction})  Feature-Wert={rf_raw[feat_idx]:.3g}")
    print(f"  Mechanistische Erklaerung:")
    if "Tacrin" in name or "Tacrine" in name:
        print("    Tacrin (9-amino-1,2,3,4-tetrahydroacridin) hat ein reaktives")
        print("    aromatisches Amin. Hochaffines CYP1A2-Substrat. Das Modell")
        print("    hat KEIN Trainingsbeispiel mit aehnlicher Enzymspezifitaet.")
    elif "Phenylparaben" in name or "paraben" in name.lower():
        print("    Phenylparaben wird durch Esterasen (CES1/CES2) extrem schnell")
        print("    hydrolysiert -- nicht ueber CYP-Wege. Das Modell kennt nur")
        print("    MW/logP/Fup und hat KEIN Trainingsbeispiel fuer Esterase-Metabolismus.")
    else:
        print("    Reaktive funktionelle Gruppen erfordern spezifische Enzymwege,")
        print("    die durch MW/logP/Fup nicht erfasst werden koennen.")

print("\n" + "=" * 65)
print("ZUSAMMENFASSUNG")
print("=" * 65)
print(f"  Analysierte Chemikalien:   {len(val)}")
print(f"  SHAP-Basiswert:            {base_value:.3f} (= mean log10 Clint Trainingsset)")
print(f"  Wichtigstes Feature:       {FEATURE_NAMES[np.argmax(mean_abs_shap)]}")
print(f"\nOutputs gespeichert in results/:")
for f in sorted(RESULTS.glob("shap_outlier_*.png")):
    print(f"  {f.name}")
print(f"  shap_outlier_values.csv")
print("\nDone.")
