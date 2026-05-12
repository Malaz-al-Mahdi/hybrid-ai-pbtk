"""
13_gcn_all777.py
----------------
GCN + RF/GB Analyse aller 777 httk-Chemikalien + BER-Berechnung.

Vorgehen
~~~~~~~~
  1. SMILES-Abruf (multi-source):
       PubChem REST  -> cactus.nci.nih.gov (CIR) -> lokal gespeichert
     Resultat wird in data/smiles_cache_777.csv zwischengespeichert.
  2. Molekuelgraphen (RDKit) fuer alle Chemikalien mit gueltigen SMILES
  3. GCN auf 19 Pilotchemikalien trainieren, auf alle 777 mit SMILES anwenden
  4. RF/GB (MW/logP/Fup – kein SMILES noetig) auf ALLE 777 anwenden
  5. Metriken und Plots
  6. BER-Berechnung fuer alle 777 Chemikalien (httk / GCN / RF) –
     ehemals 14_ber_all777.py

Outputs
~~~~~~~
  data/smiles_cache_777.csv
  results/gcn_777_predictions.csv
  results/gcn_777_metrics.txt
  results/gcn_777_scatter.png
  results/gcn_777_clint_distribution.png
  results/ber_all777.csv
  results/ber_all777_metrics.txt
  results/ber_all777_waterfall.png
  results/ber_all777_aed_scatter.png
  results/ber_all777_comparison.png
  results/ber_all777_distribution.png
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    ROOT, DATA, RESULTS,
    PILOT_CSV, ALL_777_CSV as FULL_CSV, PILOT_GCN_CSV as PILOT_GCN,
    AED_BER_CSV,
    EPSILON,
    load_smiles_cache,
    mol_to_graph, N_ATOM_FEAT,
    train_gcn, predict_gcn, MolGCN,
    GCN_EPOCHS as EPOCHS,
    engineer_features as engineer,
    compute_metrics, print_metrics,
    clint_uL_to_cl_h, calc_aed, calc_ber, concern_label,
)

try:
    import torch
    from rdkit import Chem
except ImportError:
    sys.exit("ERROR: rdkit + torch fehlen.  pip install rdkit torch")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

torch.manual_seed(42); np.random.seed(42)

print("=" * 65)
print("Step 13 - GCN + RF/GB auf allen 777 httk-Chemikalien")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Daten laden
# ═══════════════════════════════════════════════════════════════════════════════

pilot = pd.read_csv(PILOT_CSV)
df_tr = pilot.dropna(subset=["Clint"]).copy()
df_tr["Fup"] = df_tr["Fup"].clip(lower=1e-6)

full  = pd.read_csv(FULL_CSV)
full  = full.rename(columns={"Human.Clint":"Clint","Human.Funbound.plasma":"Fup"})
for col in ["Clint","Fup","MW","logP"]:
    full[col] = pd.to_numeric(full[col], errors="coerce")
full["Fup"] = full["Fup"].clip(lower=1e-6)
full["CAS"] = full["CAS"].astype(str).str.strip()

print(f"\nPilot (Training): {len(df_tr)} Chemikalien")
print(f"Vollstaendiger httk-Datensatz: {len(full)} Chemikalien")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SMILES laden / abrufen (via utils.load_smiles_cache)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- SMILES-Abruf ---")
all_cas    = full["CAS"].tolist()
smiles_map = load_smiles_cache(all_cas)

full["SMILES"] = full["CAS"].map(smiles_map)
n_smiles = full["SMILES"].notna().sum()
print(f"Chemikalien mit SMILES: {n_smiles} / {len(full)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GCN trainieren (auf Pilot-Set)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- GCN Trainings-Graphen ---")
pilot_gcn = pd.read_csv(PILOT_GCN) if PILOT_GCN.exists() else pd.DataFrame()

train_graphs, train_y, train_names = [], [], []
for _, row in df_tr.iterrows():
    cas  = str(row["CAS"]).strip()
    smi  = smiles_map.get(cas)
    if not smi or smi == "nan":
        # Fallback aus pilot_chemicals_gcn.csv
        pg_row = pilot_gcn[pilot_gcn["CAS"].astype(str) == cas]
        if len(pg_row):
            smi = pg_row.iloc[0]["SMILES"]
    if not smi or not isinstance(smi, str):
        print(f"  WARNUNG: Kein SMILES fuer {row.get('Compound','?')} (CAS={cas})")
        continue
    g = mol_to_graph(smi)
    if g is None:
        continue
    train_graphs.append(g)
    train_y.append(np.log10(float(row["Clint"]) + EPSILON))
    train_names.append(row.get("Compound","?"))
    X_g, _ = g
    mol_tmp = Chem.MolFromSmiles(smi)
    n_bonds = mol_tmp.GetNumBonds() if mol_tmp else 0
    print(f"  {str(row.get('Compound',''))[:30]:<30}: {X_g.shape[0]:>3} Atome, "
          f"{n_bonds:>3} Bindungen")

y_train = np.array(train_y)
print(f"\nGCN Trainingsdaten: {len(train_graphs)} Graphen")

gcn = None
if len(train_graphs) >= 5:
    print(f"\n--- GCN Training (max {EPOCHS} Epochen) ---")
    t0  = time.time()
    gcn = train_gcn(train_graphs, y_train)
    print(f"Training abgeschlossen in {time.time()-t0:.1f}s")
else:
    print("Zu wenige Graphen fuer GCN-Training – GCN wird uebersprungen")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RF/GB trainieren (auf Pilot-Set)
# ═══════════════════════════════════════════════════════════════════════════════

X_tr   = engineer(df_tr)
imp    = SimpleImputer(strategy="median")
sc_rf  = StandardScaler()
X_tr_s = sc_rf.fit_transform(imp.fit_transform(X_tr))
gb     = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=2,
    subsample=0.8, min_samples_leaf=2, random_state=42)
gb.fit(X_tr_s, y_train[:len(X_tr_s)])
print(f"RF/GB trainiert ({len(X_tr_s)} Chemikalien)")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Vorhersage fuer alle 777
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n--- Vorhersage fuer {len(full)} Chemikalien ---")
pilot_cas_set = set(df_tr["CAS"].astype(str).str.strip())

rows_out   = []
n_gcn_ok   = 0
n_gcn_skip = 0

for idx, row in full.iterrows():
    cas   = str(row["CAS"]).strip()
    name  = str(row.get("Compound",""))
    smi   = row.get("SMILES", None)
    clint = float(row["Clint"]) if pd.notna(row.get("Clint")) else np.nan

    # RF/GB Vorhersage (immer moeglich)
    rf_feats = engineer(row.to_frame().T)
    rf_log   = float(gb.predict(sc_rf.transform(imp.transform(rf_feats)))[0])
    rf_clint = max(10**rf_log - EPSILON, 0)

    # GCN Vorhersage (nur wenn SMILES verfuegbar)
    gcn_log = gcn_clint = np.nan
    g = None
    if gcn and pd.notna(smi) and str(smi) != "nan":
        g = mol_to_graph(str(smi))
    if gcn and g is not None:
        X_g, A_g = g
        gcn_log   = predict_gcn(gcn, X_g, A_g)
        gcn_clint = max(10**gcn_log - EPSILON, 0)
        n_gcn_ok += 1
    else:
        n_gcn_skip += 1

    # Fold-Errors
    fe_gcn = fe_rf = np.nan
    if pd.notna(clint) and clint > 0:
        lit_log = np.log10(clint + EPSILON)
        fe_rf   = round(10**abs(lit_log - rf_log), 3)
        if pd.notna(gcn_log):
            fe_gcn = round(10**abs(lit_log - gcn_log), 3)

    rows_out.append({
        "CAS": cas, "Compound": name,
        "MW": row.get("MW"), "logP": row.get("logP"), "Fup": row.get("Fup"),
        "Clint_lit": clint,
        "GCN_log10_pred": round(gcn_log, 4) if pd.notna(gcn_log) else np.nan,
        "GCN_Clint_pred": round(gcn_clint, 4) if pd.notna(gcn_clint) else np.nan,
        "RF_log10_pred":  round(rf_log,  4),
        "RF_Clint_pred":  round(rf_clint, 4),
        "fold_error_GCN": fe_gcn,
        "fold_error_RF":  fe_rf,
        "in_pilot": cas in pilot_cas_set,
        "has_smiles": pd.notna(smi) and str(smi) != "nan",
    })

    if (idx + 1) % 100 == 0:
        print(f"  {idx+1:>4}/{len(full)} verarbeitet (GCN ok: {n_gcn_ok})", flush=True)

result_df = pd.DataFrame(rows_out)
result_df.to_csv(RESULTS / "gcn_777_predictions.csv", index=False)
print(f"\nErgebnisse gespeichert: results/gcn_777_predictions.csv")
print(f"  GCN-Vorhersagen: {n_gcn_ok}")
print(f"  Nur RF/GB (kein SMILES): {n_gcn_skip}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Metriken
# ═══════════════════════════════════════════════════════════════════════════════

def metrics_report(log_lit, log_pred, label):
    m = compute_metrics(log_lit, log_pred)
    print_metrics(m, label=label, n=len(log_lit))
    return dict(Modell=label, N=len(log_lit),
                R2=round(m["r2"],4), RMSE_log=round(m["rmse"],4),
                Spearman=round(m["spearman_rho"],4), GMFE=round(m["gmfe"],2),
                Pct_2fold=round(m["pct_2fold"],1), Pct_3fold=round(m["pct_3fold"],1),
                Pct_10fold=round(m["pct_10fold"],1))

print("\n" + "="*65)
print("METRIKEN (Vergleich mit Literatur-Clint)")
print("="*65)

has_lit = result_df.dropna(subset=["Clint_lit","RF_log10_pred"])
has_lit = has_lit[has_lit["Clint_lit"] > 0].copy()
has_lit["log10_lit"] = np.log10(has_lit["Clint_lit"] + EPSILON)
ext     = has_lit[~has_lit["in_pilot"]]

all_metrics = []
# RF auf alle / extern
all_metrics.append(metrics_report(has_lit["log10_lit"].values,
    has_lit["RF_log10_pred"].values, "RF/GB -- alle mit Lit-Clint"))
if len(ext):
    all_metrics.append(metrics_report(ext["log10_lit"].values,
        ext["RF_log10_pred"].values, "RF/GB -- extern"))

# GCN (falls vorhanden)
gcn_sub = has_lit.dropna(subset=["GCN_log10_pred"])
if len(gcn_sub) >= 5:
    all_metrics.append(metrics_report(gcn_sub["log10_lit"].values,
        gcn_sub["GCN_log10_pred"].values, "GCN  -- alle mit Lit-Clint + SMILES"))
    gcn_ext = gcn_sub[~gcn_sub["in_pilot"]]
    if len(gcn_ext) >= 5:
        all_metrics.append(metrics_report(gcn_ext["log10_lit"].values,
            gcn_ext["GCN_log10_pred"].values, "GCN  -- extern"))

metrics_df = pd.DataFrame(all_metrics)
with open(RESULTS / "gcn_777_metrics.txt", "w") as f:
    f.write("GCN + RF/GB auf 777 httk-Chemikalien\n")
    f.write("=" * 52 + "\n\n")
    f.write(f"Trainingsset: {len(train_graphs)} Pilotchemikalien\n")
    f.write(f"GCN-Vorhersagen: {n_gcn_ok}\n")
    f.write(f"RF/GB-Vorhersagen: {len(result_df)}\n\n")
    f.write(metrics_df.to_string(index=False))
print(f"\nMetriken -> results/gcn_777_metrics.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Plots
# ═══════════════════════════════════════════════════════════════════════════════

def fold_col(fe_arr):
    return ["#2196F3" if f<=2 else "#4CAF50" if f<=3 else "#FF9800" if f<=10
            else "#F44336" for f in fe_arr]

# ── Plot A: RF/GB Scatter (alle 777 mit Lit-Clint) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
fe_arr = has_lit["fold_error_RF"].fillna(999).values
cols   = fold_col(fe_arr)
ax.scatter(has_lit.loc[~has_lit["in_pilot"],"log10_lit"],
           has_lit.loc[~has_lit["in_pilot"],"RF_log10_pred"],
           c=[cols[i] for i in range(len(has_lit)) if not has_lit["in_pilot"].values[i]],
           s=12, alpha=0.6, linewidths=0, label="Extern")
ax.scatter(has_lit.loc[has_lit["in_pilot"],"log10_lit"],
           has_lit.loc[has_lit["in_pilot"],"RF_log10_pred"],
           c="gold", s=80, edgecolors="k", lw=0.8, zorder=5, label="Pilot (Training)")
lims = [has_lit["log10_lit"].min()-0.5, has_lit["log10_lit"].max()+0.5]
ax.plot(lims, lims, "k--", lw=1.2)
ax.fill_between(lims, [v-np.log10(3) for v in lims],
                [v+np.log10(3) for v in lims], alpha=0.07, color="green")
m0 = all_metrics[0]
ax.set_title(f"RF/GB  (n={m0['N']})\nR²={m0['R2']:.3f}  GMFE={m0['GMFE']:.1f}x  "
             f"<=3-fold={m0['Pct_3fold']:.0f}%", fontsize=10)
ax.set_xlabel("log10(Clint Literatur)"); ax.set_ylabel("log10(Clint RF/GB)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# ── Plot B: GCN Scatter oder Hinweis ─────────────────────────────────────────
ax = axes[1]
if len(gcn_sub) >= 5:
    fe2   = gcn_sub["fold_error_GCN"].fillna(999).values
    c2    = fold_col(fe2)
    ax.scatter(gcn_sub.loc[~gcn_sub["in_pilot"],"log10_lit"],
               gcn_sub.loc[~gcn_sub["in_pilot"],"GCN_log10_pred"],
               c=[c2[i] for i in range(len(gcn_sub)) if not gcn_sub["in_pilot"].values[i]],
               s=12, alpha=0.6, linewidths=0, label="Extern")
    ax.scatter(gcn_sub.loc[gcn_sub["in_pilot"],"log10_lit"],
               gcn_sub.loc[gcn_sub["in_pilot"],"GCN_log10_pred"],
               c="gold", s=80, edgecolors="k", lw=0.8, zorder=5, label="Pilot (Training)")
    lims2 = [gcn_sub["log10_lit"].min()-0.5, gcn_sub["log10_lit"].max()+0.5]
    ax.plot(lims2, lims2, "k--", lw=1.2)
    m_gcn = [m for m in all_metrics if "GCN" in m["Modell"]]
    if m_gcn:
        mg = m_gcn[0]
        ax.set_title(f"GCN  (n={mg['N']})\nR²={mg['R2']:.3f}  GMFE={mg['GMFE']:.1f}x  "
                     f"<=3-fold={mg['Pct_3fold']:.0f}%", fontsize=10)
    ax.set_xlabel("log10(Clint Literatur)"); ax.set_ylabel("log10(Clint GCN)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
else:
    ax.text(0.5, 0.5,
            f"GCN: {n_gcn_ok} Chemikalien mit SMILES\n"
            "(PubChem API in dieser Umgebung\nnicht erreichbar – nur RF/GB verfuegbar)",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", fc="#FFF9C4", ec="#F9A825"))
    ax.set_title("GCN – SMILES-Abruf nicht moeglich", fontsize=10)
    ax.axis("off")

from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#2196F3", label="<=2-fold"),
              Patch(facecolor="#4CAF50", label="<=3-fold"),
              Patch(facecolor="#FF9800", label="<=10-fold"),
              Patch(facecolor="#F44336", label=">10-fold")]
axes[0].legend(handles=legend_els+[plt.scatter([],[],c="gold",s=50,
               edgecolors="k",label="Pilot (Training)")], fontsize=7, loc="upper left")
plt.suptitle("Clint-Vorhersage: 777 httk-Chemikalien | Trainiert auf 19 Piloten",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(RESULTS / "gcn_777_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/gcn_777_scatter.png")

# ── Plot B: Vorhersageverteilung + Ausreisser ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ok_df = result_df
ax.hist(ok_df["RF_log10_pred"].dropna(), bins=40,
        color="#2196F3", alpha=0.7, label="RF/GB (n=777)", edgecolor="white")
if len(gcn_sub):
    ax.hist(gcn_sub["GCN_log10_pred"].dropna(), bins=30,
            color="#E91E63", alpha=0.6, label=f"GCN (n={n_gcn_ok})", edgecolor="white")
if has_lit["log10_lit"].notna().sum():
    ax.hist(has_lit["log10_lit"], bins=40, color="#4CAF50", alpha=0.45,
            label="Literatur (gemessen)", edgecolor="white")
ax.set_xlabel("log10(Clint)"); ax.set_ylabel("Anzahl Chemikalien")
ax.set_title("Vorhersageverteilung (alle 777)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(has_lit["RF_log10_pred"], has_lit["fold_error_RF"].apply(np.log10),
           s=8, alpha=0.4, c="#2196F3", linewidths=0)
ax.axhline(np.log10(3),  color="orange", lw=1.5, ls="--", label="3-fold")
ax.axhline(np.log10(10), color="red",    lw=1.5, ls=":",  label="10-fold")
ax.axhline(0,            color="green",  lw=1.2, ls="-")
ax.set_xlabel("RF/GB Vorhersage log10(Clint)"); ax.set_ylabel("log10(Fold-Error)")
ax.set_title(f"RF/GB Fehlerplot (n={len(has_lit)})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
top20 = has_lit.nlargest(20, "fold_error_RF")[["Compound","fold_error_RF"]].copy()
top20["Compound"] = top20["Compound"].str[:24]
ax.barh(top20["Compound"][::-1],
        np.log10(top20["fold_error_RF"][::-1]+1),
        color="#F44336", edgecolor="k", lw=0.3)
ax.axvline(np.log10(3),  color="orange", lw=1.5, ls="--", label="3-fold")
ax.axvline(np.log10(10), color="red",    lw=1.5, ls=":",  label="10-fold")
ax.set_xlabel("log10(Fold-Error + 1)"); ax.set_title("Top-20 RF/GB Ausreisser")
ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / "gcn_777_clint_distribution.png", dpi=150)
plt.close()
print("Saved: results/gcn_777_clint_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Abschlusszusammenfassung
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ABSCHLUSSZUSAMMENFASSUNG GCN + RF/GB")
print("="*65)
print(f"\n  Gesamtdatensatz       : {len(full)} Chemikalien")
print(f"  Mit Lit-Clint         : {len(has_lit)}")
print(f"  SMILES verfuegbar     : {n_smiles} ({n_smiles/len(full)*100:.0f}%)")
print(f"  GCN-Vorhersagen       : {n_gcn_ok}")
print(f"  RF/GB-Vorhersagen     : {len(result_df)}")
print()
print(metrics_df[["Modell","N","R2","RMSE_log","GMFE",
                   "Pct_3fold","Pct_10fold"]].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 13. BER-Berechnung fuer alle 777 Chemikalien (ehemals 14_ber_all777.py)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("Step 13b - BER fuer alle 777 httk-Chemikalien")
print("="*65)

# ── Daten zusammenfuehren ────────────────────────────────────────────────────
# full (already loaded above) hat die SEEM/AC50-Spalten aus all_777_chemicals.csv
# result_df (already computed above) hat GCN/RF-Vorhersagen

# SEEM-Spalten sichern, falls vorhanden
seem_cols = [c for c in ["SEEM_mg_kg_day","SEEM_l95","SEEM_u95","SEEM_pathway","has_SEEM"]
             if c in full.columns]

merge_cols = ["DTXSID","Compound","CAS","MW","logP","Fup","Clint"] + seem_cols
merge_cols = [c for c in merge_cols if c in full.columns]

df_ber = full[merge_cols].copy()
df_ber = df_ber.rename(columns={"Clint": "Clint_httk"})
df_ber["CAS"] = df_ber["CAS"].astype(str).str.strip()

# GCN / RF Vorhersagen einfuegen
df_ber = df_ber.merge(
    result_df[["CAS","GCN_Clint_pred","RF_Clint_pred",
               "GCN_log10_pred","RF_log10_pred","has_smiles"]],
    on="CAS", how="left",
)

# AC50 aus aed_ber_full.csv (Step 05)
if AED_BER_CSV.exists():
    ber_ref = pd.read_csv(AED_BER_CSV)
    ber_ref["CAS"] = ber_ref["CAS"].astype(str).str.strip()
    df_ber = df_ber.merge(
        ber_ref[["CAS","AC50_5pct_uM","AED_median","BER"]].rename(
            columns={"AC50_5pct_uM": "AC50_10pct_uM", "AED_median": "AED_httk_ref", "BER": "BER_httk_ref"}),
        on="CAS", how="left",
    )
else:
    df_ber["AC50_10pct_uM"] = np.nan
    df_ber["AED_httk_ref"]  = np.nan
    df_ber["BER_httk_ref"]  = np.nan

print(f"\n  Merge: {len(df_ber)} Zeilen")
print(f"  mit AC50         : {df_ber['AC50_10pct_uM'].notna().sum()}")
seem_col = "SEEM_mg_kg_day" if "SEEM_mg_kg_day" in df_ber.columns else None
if seem_col:
    print(f"  mit SEEM         : {df_ber[seem_col].notna().sum()}")
print(f"  mit GCN-Clint    : {df_ber['GCN_Clint_pred'].notna().sum()}")
print(f"  mit httk-Clint   : {df_ber['Clint_httk'].notna().sum()}")

# ── AED und BER berechnen ────────────────────────────────────────────────────
print("\nBerechne AED/BER fuer alle 777 Chemikalien ...")

rows_ber = []
for _, row in df_ber.iterrows():
    mw    = float(row["MW"])   if pd.notna(row.get("MW"))   else 300.0
    fup   = float(row["Fup"])  if pd.notna(row.get("Fup"))  else 0.1
    ac50  = float(row["AC50_10pct_uM"]) if pd.notna(row.get("AC50_10pct_uM")) else np.nan
    seem  = float(row[seem_col]) if seem_col and pd.notna(row.get(seem_col)) else np.nan

    c_httk = float(row["Clint_httk"])    if pd.notna(row.get("Clint_httk"))    else np.nan
    c_gcn  = float(row["GCN_Clint_pred"]) if pd.notna(row.get("GCN_Clint_pred")) else np.nan
    c_rf   = float(row["RF_Clint_pred"]) if pd.notna(row.get("RF_Clint_pred")) else np.nan

    aed_httk = calc_aed(ac50, mw, c_httk, fup) if pd.notna(ac50) and pd.notna(c_httk) else np.nan
    aed_gcn  = calc_aed(ac50, mw, c_gcn,  fup) if pd.notna(ac50) and pd.notna(c_gcn)  else np.nan
    aed_rf   = calc_aed(ac50, mw, c_rf,   fup) if pd.notna(ac50) and pd.notna(c_rf)   else np.nan

    ber_httk = calc_ber(aed_httk, seem)
    ber_gcn  = calc_ber(aed_gcn,  seem)
    ber_rf   = calc_ber(aed_rf,   seem)

    rows_ber.append({
        "DTXSID":         row.get("DTXSID"),
        "CAS":            row["CAS"],
        "Compound":       row.get("Compound"),
        "MW":             mw,
        "logP":           row.get("logP"),
        "Fup":            fup,
        "AC50_10pct_uM":  ac50,
        "SEEM_mg_kg_day": seem,
        "Clint_httk":     round(c_httk, 4) if pd.notna(c_httk) else np.nan,
        "Clint_GCN":      round(c_gcn,  4) if pd.notna(c_gcn)  else np.nan,
        "Clint_RF":       round(c_rf,   4) if pd.notna(c_rf)   else np.nan,
        "CL_httk":        round(clint_uL_to_cl_h(c_httk, fup), 6) if pd.notna(c_httk) else np.nan,
        "CL_GCN":         round(clint_uL_to_cl_h(c_gcn,  fup), 6) if pd.notna(c_gcn)  else np.nan,
        "CL_RF":          round(clint_uL_to_cl_h(c_rf,   fup), 6) if pd.notna(c_rf)   else np.nan,
        "AED_httk":       round(aed_httk, 6) if pd.notna(aed_httk) else np.nan,
        "AED_GCN":        round(aed_gcn,  6) if pd.notna(aed_gcn)  else np.nan,
        "AED_RF":         round(aed_rf,   6) if pd.notna(aed_rf)   else np.nan,
        "BER_httk":       round(ber_httk, 4) if pd.notna(ber_httk) else np.nan,
        "BER_GCN":        round(ber_gcn,  4) if pd.notna(ber_gcn)  else np.nan,
        "BER_RF":         round(ber_rf,   4) if pd.notna(ber_rf)   else np.nan,
        "BER_httk_ref":   row.get("BER_httk_ref", np.nan),
    })

result_ber = pd.DataFrame(rows_ber)
result_ber["concern_httk"] = result_ber["BER_httk"].apply(concern_label)
result_ber["concern_GCN"]  = result_ber["BER_GCN"].apply(concern_label)
result_ber["concern_RF"]   = result_ber["BER_RF"].apply(concern_label)

result_ber.to_csv(RESULTS / "ber_all777.csv", index=False)
print(f"Gespeichert: results/ber_all777.csv")

# ── Metriken ─────────────────────────────────────────────────────────────────
ber_ok  = result_ber.dropna(subset=["BER_httk"])
aed_ok  = result_ber.dropna(subset=["AED_httk","AED_GCN","AED_RF"])

print(f"\n  Chemikalien gesamt          : {len(result_ber)}")
print(f"  mit AC50 (ToxCast)          : {result_ber['AC50_10pct_uM'].notna().sum()}")
print(f"  mit SEEM-Exposition         : {result_ber['SEEM_mg_kg_day'].notna().sum()}")
print(f"  AED berechenbar (alle 3)    : {len(aed_ok)}")
print(f"  BER berechenbar             : {ber_ok['BER_httk'].notna().sum()}")

for lbl, col in [("httk","AED_httk"),("GCN","AED_GCN"),("RF","AED_RF")]:
    sub = result_ber[col].dropna()
    if len(sub):
        print(f"  AED_{lbl:<5}: n={len(sub):>4}  median={sub.median():.3e}  "
              f"IQR=[{sub.quantile(0.25):.2e}, {sub.quantile(0.75):.2e}]")

if len(ber_ok):
    print("\n--- Concern-Klassifikation (BER_httk) ---")
    for cat, cnt in ber_ok["concern_httk"].value_counts().items():
        print(f"  {cat:<30}: {cnt}")

    if len(aed_ok) > 1:
        log_httk = np.log10(aed_ok["AED_httk"].clip(1e-10))
        log_gcn  = np.log10(aed_ok["AED_GCN"].clip(1e-10))
        log_rf   = np.log10(aed_ok["AED_RF"].clip(1e-10))
        rho_gcn, p_gcn = spearmanr(log_httk, log_gcn)
        rho_rf,  p_rf  = spearmanr(log_httk, log_rf)
        fe_gcn = 10**np.abs(log_httk - log_gcn)
        fe_rf  = 10**np.abs(log_httk - log_rf)
        print(f"\n--- AED Vergleich vs. httk (n={len(aed_ok)}) ---")
        print(f"  GCN:  rho={rho_gcn:.3f} (p={p_gcn:.3e})  "
              f"GMFE={np.exp(np.mean(np.log(fe_gcn))):.2f}x  "
              f"<=2-fold={np.mean(fe_gcn<=2)*100:.0f}%")
        print(f"  RF:   rho={rho_rf:.3f}  (p={p_rf:.3e})  "
              f"GMFE={np.exp(np.mean(np.log(fe_rf))):.2f}x  "
              f"<=2-fold={np.mean(fe_rf<=2)*100:.0f}%")

with open(RESULTS / "ber_all777_metrics.txt", "w") as f:
    f.write("BER Analyse - 777 httk-Chemikalien\n")
    f.write("="*52 + "\n\n")
    f.write(f"Gesamt       : {len(result_ber)}\n")
    f.write(f"mit AC50     : {result_ber['AC50_10pct_uM'].notna().sum()}\n")
    f.write(f"mit SEEM     : {result_ber['SEEM_mg_kg_day'].notna().sum()}\n")
    f.write(f"BER ok       : {ber_ok['BER_httk'].notna().sum()}\n\n")
    if len(ber_ok):
        f.write(ber_ok[["Compound","BER_httk","BER_GCN","BER_RF",
                          "SEEM_mg_kg_day","concern_httk"]].to_string(index=False))
print("Metriken -> results/ber_all777_metrics.txt")

# ── BER Plots ────────────────────────────────────────────────────────────────
ber_plot = result_ber.dropna(subset=["BER_httk"]).copy()
ber_plot = ber_plot.sort_values("BER_httk").reset_index(drop=True)

if len(ber_plot):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ber_plot))
    ax.bar(x, np.log10(ber_plot["BER_httk"].clip(1e-4)),
           color="#B0BEC5", alpha=0.9, label="httk (Referenz)", zorder=2)
    if ber_plot["BER_GCN"].notna().sum():
        ax.scatter(x, np.log10(ber_plot["BER_GCN"].clip(1e-4)),
                   color="#1565C0", s=60, zorder=5, marker="D", label="GCN")
    if ber_plot["BER_RF"].notna().sum():
        ax.scatter(x, np.log10(ber_plot["BER_RF"].clip(1e-4)),
                   color="#E65100", s=40, zorder=4, marker="o", label="RF/GB")
    ax.axhline(0, color="red", lw=2, ls="--", label="BER=1 (Grenzwert)")
    ax.axhline(1, color="orange", lw=1.5, ls=":", label="BER=10")
    ax.axhline(2, color="green",  lw=1.0, ls=":", label="BER=100")
    ax.set_xticks(x)
    ax.set_xticklabels(ber_plot["Compound"].str[:20], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("log10(BER)  [niedriger = hoehere Besorgnis]", fontsize=11)
    ax.set_title(f"BER-Ranking: httk vs. GCN vs. RF/GB Clint\n"
                 f"n={len(ber_plot)} Chemikalien mit SEEM-Expositionsdaten", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "ber_all777_waterfall.png", dpi=150)
    plt.close()
    print("Saved: results/ber_all777_waterfall.png")

# AED Scatter GCN/RF vs httk
both_aed = result_ber.dropna(subset=["AED_httk","AED_GCN","AED_RF"])
if len(both_aed) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (gcn_col, title, clr) in zip(axes, [
        ("AED_GCN", "GCN vs. httk",   "#1565C0"),
        ("AED_RF",  "RF/GB vs. httk", "#E65100"),
    ]):
        log_x = np.log10(both_aed["AED_httk"].clip(1e-10))
        log_y = np.log10(both_aed[gcn_col].clip(1e-10))
        has_s = both_aed["SEEM_mg_kg_day"].notna()
        ax.scatter(log_x[~has_s], log_y[~has_s], c="#90A4AE", s=10, alpha=0.4, linewidths=0)
        ax.scatter(log_x[has_s],  log_y[has_s],  c=clr, s=60, alpha=0.8,
                   linewidths=0.5, edgecolors="k", zorder=5, label="mit SEEM")
        lims = [min(log_x.min(),log_y.min())-0.5, max(log_x.max(),log_y.max())+0.5]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.fill_between(lims, [v-1 for v in lims], [v+1 for v in lims],
                        alpha=0.06, color="orange")
        rho, pv = spearmanr(log_x, log_y)
        fe = 10**np.abs(log_x - log_y)
        gmfe = np.exp(np.mean(np.log(fe)))
        ax.set_xlabel("log10(AED httk)  [Referenz]", fontsize=10)
        ax.set_ylabel(f"log10(AED {title.split()[0]})", fontsize=10)
        ax.set_title(f"AED: {title}  (n={len(both_aed)})\n"
                     f"Spearman rho={rho:.3f}  GMFE={gmfe:.2f}x", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.suptitle("AED-Vergleich: ML-Vorhersage vs. httk-Literatur", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS / "ber_all777_aed_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/ber_all777_aed_scatter.png")

# AED/BER Verteilung
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
aed_all = result_ber.dropna(subset=["AED_httk","AED_GCN","AED_RF"])
bins = np.linspace(-5, 5, 40)
ax.hist(np.log10(aed_all["AED_httk"].clip(1e-5)), bins=bins,
        color="#607D8B", alpha=0.75, label=f"httk (n={aed_all['AED_httk'].notna().sum()})")
ax.hist(np.log10(aed_all["AED_GCN"].clip(1e-5)), bins=bins,
        color="#1565C0", alpha=0.55, label=f"GCN  (n={aed_all['AED_GCN'].notna().sum()})")
ax.hist(np.log10(aed_all["AED_RF"].clip(1e-5)),  bins=bins,
        color="#E65100", alpha=0.40, label=f"RF   (n={aed_all['AED_RF'].notna().sum()})")
ax.set_xlabel("log10(AED [mg/kg/day])", fontsize=10)
ax.set_ylabel("Anzahl Chemikalien", fontsize=10)
ax.set_title("AED-Verteilung: alle 777 Chemikalien", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
if len(ber_plot):
    colors_ber = {"HIGH  (BER<1)": "#F44336", "MEDIUM (BER 1-10)": "#FF9800",
                  "LOW   (BER 10-100)": "#8BC34A", "NEGLIGIBLE (BER>100)": "#4CAF50",
                  "no_data": "#B0BEC5"}
    cat_counts = ber_plot["concern_httk"].value_counts()
    cats  = list(cat_counts.index)
    vals  = [cat_counts[c] for c in cats]
    clrs  = [colors_ber.get(c, "#9E9E9E") for c in cats]
    ax.bar(cats, vals, color=clrs, edgecolor="k", linewidth=0.4)
    ax.set_ylabel("Anzahl Chemikalien")
    ax.set_title("BER-Concern-Klassifikation (httk)")
    ax.set_xticklabels([c[:22] for c in cats], rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v+0.1, str(v), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS / "ber_all777_distribution.png", dpi=150)
plt.close()
print("Saved: results/ber_all777_distribution.png")


print("\n" + "="*65)
print("ABSCHLUSSZUSAMMENFASSUNG GESAMT")
print("="*65)
print()
print("GCN + RF/GB:")
print("  data/smiles_cache_777.csv               -- SMILES-Cache")
print("  results/gcn_777_predictions.csv         -- Vorhersagen")
print("  results/gcn_777_metrics.txt             -- Metriken")
print("  results/gcn_777_scatter.png             -- Scatter-Plot")
print("  results/gcn_777_clint_distribution.png  -- Verteilung + Ausreisser")
print()
print("BER-Analyse:")
print("  results/ber_all777.csv                  -- vollstaendige Tabelle")
print("  results/ber_all777_metrics.txt          -- Metriken")
print("  results/ber_all777_waterfall.png        -- BER-Wasserfall")
print("  results/ber_all777_aed_scatter.png      -- AED: GCN/RF vs. httk")
print("  results/ber_all777_distribution.png     -- AED/BER Verteilung")
print("\nDone.")
