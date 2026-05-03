"""
14_ber_all777.py
----------------
BER (Bioactivity Exposure Ratio) fuer alle 777 httk-Chemikalien.

Methodik
~~~~~~~~
  BER = AED / Exposure

  AED (Activity Equivalent Dose):
    AED = AC50 [mg/L] / Css_1mg [mg/L pro mg/kg/day]
    Css_1mg = Fup / (CL_hepatic * 24)   [Well-Stirred Modell]

    CL_hepatic = Q_H * Fup * Clint_invivo / (Q_H + Fup * Clint_invivo)
    Clint_invivo = Clint_uL * 1e-6 * 60 * 110e6_cells * 26e-3_kg_liver

  Drei Clint-Quellen werden verglichen:
    1. httk-Literatur  (Human.Clint aus Wetmore 2012)
    2. RF/GB-Vorhersage (Script 02/13)
    3. GCN-Vorhersage   (Script 13)

  Exposure: SEEM3-Schaetzwerte (SEEM_mg_kg_day aus all_777_chemicals.csv)
  AC50:     ToxCast / aed_ber_full.csv (AC50_10pct_uM)

Outputs
~~~~~~~
  results/ber_all777.csv              Vollstaendige BER-Tabelle
  results/ber_all777_comparison.png   GCN vs RF vs httk BER Vergleich
  results/ber_all777_waterfall.png    BER-Wasserfall (Rangfolge nach httk-BER)
  results/ber_all777_aed_scatter.png  AED Scatter: GCN vs httk
  results/ber_all777_metrics.txt      Metriken
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

# ── PK-Konstanten (identisch zu Script 08) ────────────────────────────────────
Q_H     = 1.5        # hepatische Blutflussrate [L/h/kg]
F_LIVER = 26e-3      # Leberanteil am Koerpergewicht [kg/kg]
HEPATO  = 110e6      # Hepatozyten pro g Leber

def clint_uL_to_cl_h(clint_uL: float, fup: float) -> float:
    """
    Well-Stirred Modell: hepatische Clearance [L/h/kg]
    aus Clint (µL/min/Mio Zellen) und Fup.
    """
    fup = max(float(fup), 1e-4)
    c   = max(float(clint_uL), 0.0)
    clint_L = c * 1e-6 * 60.0 * HEPATO * F_LIVER
    return Q_H * fup * clint_L / (Q_H + fup * clint_L)

def css_per_dose(cl_h: float, fup: float) -> float:
    """
    Steady-State Konzentration pro Dosis [mg/L pro mg/kg/day]
    Css = Dose * F_abs / (CL_h * 24)
    F_abs = 1 (vollstaendige orale Absorption, konservativ)
    """
    cl_h = max(cl_h, 1e-9)
    return fup / (cl_h * 24.0)

def ac50_uM_to_mg_L(ac50_uM: float, mw: float) -> float:
    """AC50 von µM in mg/L umrechnen."""
    return float(ac50_uM) * float(mw) / 1000.0

def calc_aed(ac50_uM: float, mw: float, clint_uL: float, fup: float) -> float:
    """
    AED (mg/kg/day) nach Well-Stirred IVIVE:
    AED = AC50 [mg/L] / Css_1mg [mg/L / (mg/kg/day)]
    """
    ac50_mg = ac50_uM_to_mg_L(ac50_uM, mw)
    cl_h    = clint_uL_to_cl_h(clint_uL, fup)
    c1mg    = css_per_dose(cl_h, fup)
    if c1mg <= 0:
        return np.nan
    return ac50_mg / c1mg

def calc_ber(aed: float, exposure: float) -> float:
    """BER = AED / Exposure."""
    if pd.isna(aed) or pd.isna(exposure) or exposure <= 0 or aed <= 0:
        return np.nan
    return aed / exposure


print("=" * 65)
print("Step 14 - BER fuer alle 777 httk-Chemikalien")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Daten laden
# ═══════════════════════════════════════════════════════════════════════════════

full = pd.read_csv(DATA / "all_777_chemicals.csv")
full = full.rename(columns={
    "Human.Clint":             "Clint_httk",
    "Human.Funbound.plasma":   "Fup",
})
full["CAS"]        = full["CAS"].astype(str).str.strip()
full["Clint_httk"] = pd.to_numeric(full["Clint_httk"], errors="coerce")
full["Fup"]        = pd.to_numeric(full["Fup"],        errors="coerce").clip(lower=1e-4)
full["MW"]         = pd.to_numeric(full["MW"],         errors="coerce")
full["logP"]       = pd.to_numeric(full["logP"],       errors="coerce")

# GCN + RF Vorhersagen
gcn_df = pd.read_csv(RESULTS / "gcn_777_predictions.csv")
gcn_df["CAS"] = gcn_df["CAS"].astype(str).str.strip()

# AED/BER Referenz (httk-native, Step 05)
ber_ref = pd.read_csv(RESULTS / "aed_ber_full.csv")
ber_ref["CAS"] = ber_ref["CAS"].astype(str).str.strip()

print(f"  all_777:       {len(full)} Chemikalien")
print(f"  GCN-Predictions: {len(gcn_df)}")
print(f"  ber_ref (httk):  {len(ber_ref)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Daten zusammenfuehren
# ═══════════════════════════════════════════════════════════════════════════════

df = full[["DTXSID","Compound","CAS","MW","logP","Fup",
           "Clint_httk","SEEM_mg_kg_day","SEEM_l95","SEEM_u95",
           "SEEM_pathway","has_SEEM"]].copy()

# GCN / RF Vorhersagen
df = df.merge(
    gcn_df[["CAS","GCN_Clint_pred","RF_Clint_pred",
             "GCN_log10_pred","RF_log10_pred","has_smiles"]],
    on="CAS", how="left",
)

# AC50 aus ber_ref
df = df.merge(
    ber_ref[["CAS","AC50_10pct_uM","AED_median","BER"]].rename(
        columns={"AED_median":"AED_httk_ref","BER":"BER_httk_ref"}),
    on="CAS", how="left",
)

print(f"\nNach Merge: {len(df)} Zeilen")
print(f"  mit AC50         : {df['AC50_10pct_uM'].notna().sum()}")
print(f"  mit SEEM         : {df['SEEM_mg_kg_day'].notna().sum()}")
print(f"  mit GCN-Clint    : {df['GCN_Clint_pred'].notna().sum()}")
print(f"  mit httk-Clint   : {df['Clint_httk'].notna().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. AED berechnen (alle 3 Clint-Quellen)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBerechne AED fuer alle 777 Chemikalien ...")

rows = []
for _, row in df.iterrows():
    mw    = float(row["MW"])   if pd.notna(row["MW"])   else 300.0
    fup   = float(row["Fup"])  if pd.notna(row["Fup"])  else 0.1
    ac50  = float(row["AC50_10pct_uM"]) if pd.notna(row["AC50_10pct_uM"]) else np.nan
    seem  = float(row["SEEM_mg_kg_day"]) if pd.notna(row["SEEM_mg_kg_day"]) else np.nan

    # Clint aus drei Quellen
    c_httk = float(row["Clint_httk"])    if pd.notna(row["Clint_httk"])    else np.nan
    c_gcn  = float(row["GCN_Clint_pred"]) if pd.notna(row["GCN_Clint_pred"]) else np.nan
    c_rf   = float(row["RF_Clint_pred"]) if pd.notna(row["RF_Clint_pred"]) else np.nan

    # AED
    aed_httk = calc_aed(ac50, mw, c_httk, fup) if pd.notna(ac50) and pd.notna(c_httk) else np.nan
    aed_gcn  = calc_aed(ac50, mw, c_gcn,  fup) if pd.notna(ac50) and pd.notna(c_gcn)  else np.nan
    aed_rf   = calc_aed(ac50, mw, c_rf,   fup) if pd.notna(ac50) and pd.notna(c_rf)   else np.nan

    # BER
    ber_httk = calc_ber(aed_httk, seem)
    ber_gcn  = calc_ber(aed_gcn,  seem)
    ber_rf   = calc_ber(aed_rf,   seem)

    # Clearances
    cl_httk = clint_uL_to_cl_h(c_httk, fup) if pd.notna(c_httk) else np.nan
    cl_gcn  = clint_uL_to_cl_h(c_gcn,  fup) if pd.notna(c_gcn)  else np.nan
    cl_rf   = clint_uL_to_cl_h(c_rf,   fup) if pd.notna(c_rf)   else np.nan

    rows.append({
        "DTXSID":        row["DTXSID"],
        "CAS":           row["CAS"],
        "Compound":      row["Compound"],
        "MW":            mw,
        "logP":          row["logP"],
        "Fup":           fup,
        "AC50_10pct_uM": ac50,
        "SEEM_mg_kg_day": seem,
        "SEEM_pathway":  row.get("SEEM_pathway", np.nan),
        # Clint (alle Quellen)
        "Clint_httk":    round(c_httk, 4) if pd.notna(c_httk) else np.nan,
        "Clint_GCN":     round(c_gcn,  4) if pd.notna(c_gcn)  else np.nan,
        "Clint_RF":      round(c_rf,   4) if pd.notna(c_rf)   else np.nan,
        # Hepatische Clearance
        "CL_httk":       round(cl_httk, 6) if pd.notna(cl_httk) else np.nan,
        "CL_GCN":        round(cl_gcn,  6) if pd.notna(cl_gcn)  else np.nan,
        "CL_RF":         round(cl_rf,   6) if pd.notna(cl_rf)   else np.nan,
        # AED
        "AED_httk":      round(aed_httk, 6) if pd.notna(aed_httk) else np.nan,
        "AED_GCN":       round(aed_gcn,  6) if pd.notna(aed_gcn)  else np.nan,
        "AED_RF":        round(aed_rf,   6) if pd.notna(aed_rf)   else np.nan,
        # BER
        "BER_httk":      round(ber_httk, 4) if pd.notna(ber_httk) else np.nan,
        "BER_GCN":       round(ber_gcn,  4) if pd.notna(ber_gcn)  else np.nan,
        "BER_RF":        round(ber_rf,   4) if pd.notna(ber_rf)   else np.nan,
        "BER_httk_ref":  row.get("BER_httk_ref", np.nan),
    })

result = pd.DataFrame(rows)

# ── Concern-Klassifikation ────────────────────────────────────────────────────
def concern(ber):
    if pd.isna(ber):  return "no_data"
    if ber < 1:       return "HIGH  (BER<1)"
    if ber < 10:      return "MEDIUM (BER 1-10)"
    if ber < 100:     return "LOW   (BER 10-100)"
    return "NEGLIGIBLE (BER>100)"

result["concern_httk"] = result["BER_httk"].apply(concern)
result["concern_GCN"]  = result["BER_GCN"].apply(concern)
result["concern_RF"]   = result["BER_RF"].apply(concern)

result.to_csv(RESULTS / "ber_all777.csv", index=False)
print(f"Gespeichert: results/ber_all777.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Metriken
# ═══════════════════════════════════════════════════════════════════════════════

aed_ok = result.dropna(subset=["AED_httk","AED_GCN","AED_RF"])
ber_ok = result.dropna(subset=["BER_httk"])

print("\n" + "="*65)
print("ZUSAMMENFASSUNG")
print("="*65)
print(f"\n  Chemikalien gesamt          : {len(result)}")
print(f"  mit AC50 (ToxCast)          : {result['AC50_10pct_uM'].notna().sum()}")
print(f"  mit SEEM-Exposition         : {result['SEEM_mg_kg_day'].notna().sum()}")
print(f"  AED berechenbar (alle 3)    : {len(aed_ok)}")
print(f"  BER berechenbar             : {ber_ok['BER_httk'].notna().sum()}")

print("\n--- AED-Statistik (alle Chemikalien mit AC50) ---")
for lbl, col in [("httk","AED_httk"),("GCN","AED_GCN"),("RF","AED_RF")]:
    sub = result[col].dropna()
    if len(sub):
        print(f"  AED_{lbl:<5}: n={len(sub):>4}  median={sub.median():.3e}  "
              f"IQR=[{sub.quantile(0.25):.2e}, {sub.quantile(0.75):.2e}]")

if len(ber_ok):
    print("\n--- BER-Statistik (Chemikalien mit SEEM-Daten) ---")
    print(f"  {'Chemikalienname':<40} {'httk':>8} {'GCN':>8} {'RF':>8}  Exposition")
    print(f"  {'-'*80}")
    ber_disp = ber_ok.sort_values("BER_httk").head(30)
    for _, r in ber_disp.iterrows():
        print(f"  {str(r['Compound'])[:40]:<40} "
              f"{r['BER_httk']:>8.2f} "
              f"{r['BER_GCN'] if pd.notna(r['BER_GCN']) else float('nan'):>8.2f} "
              f"{r['BER_RF']  if pd.notna(r['BER_RF'])  else float('nan'):>8.2f}"
              f"  {r['SEEM_mg_kg_day']:.2e}")

    # Concern-Verteilung
    print("\n--- Concern-Klassifikation (BER_httk) ---")
    for cat, cnt in ber_ok["concern_httk"].value_counts().items():
        print(f"  {cat:<30}: {cnt}")

    # AED-Vergleich GCN vs httk
    both_aed = result.dropna(subset=["AED_httk","AED_GCN"])
    if len(both_aed) > 1:
        log_httk = np.log10(both_aed["AED_httk"].clip(1e-10))
        log_gcn  = np.log10(both_aed["AED_GCN"].clip(1e-10))
        log_rf   = np.log10(both_aed["AED_RF"].clip(1e-10))
        rho_gcn, p_gcn = spearmanr(log_httk, log_gcn)
        rho_rf,  p_rf  = spearmanr(log_httk, log_rf)
        fe_gcn = 10**np.abs(log_httk - log_gcn)
        fe_rf  = 10**np.abs(log_httk - log_rf)
        print(f"\n--- AED Vergleich vs. httk (n={len(both_aed)}) ---")
        print(f"  GCN:  Spearman rho={rho_gcn:.3f} (p={p_gcn:.3e})  "
              f"GMFE={np.exp(np.mean(np.log(fe_gcn))):.2f}x  "
              f"<=2-fold={np.mean(fe_gcn<=2)*100:.0f}%  "
              f"<=10-fold={np.mean(fe_gcn<=10)*100:.0f}%")
        print(f"  RF:   Spearman rho={rho_rf:.3f}  (p={p_rf:.3e})  "
              f"GMFE={np.exp(np.mean(np.log(fe_rf))):.2f}x  "
              f"<=2-fold={np.mean(fe_rf<=2)*100:.0f}%  "
              f"<=10-fold={np.mean(fe_rf<=10)*100:.0f}%")

# Metriken-Datei
with open(RESULTS / "ber_all777_metrics.txt", "w") as f:
    f.write("BER Analyse - 777 httk-Chemikalien\n")
    f.write("="*52 + "\n\n")
    f.write(f"Gesamt       : {len(result)}\n")
    f.write(f"mit AC50     : {result['AC50_10pct_uM'].notna().sum()}\n")
    f.write(f"mit SEEM     : {result['SEEM_mg_kg_day'].notna().sum()}\n")
    f.write(f"BER ok       : {ber_ok['BER_httk'].notna().sum()}\n\n")
    if len(ber_ok):
        f.write(ber_ok[["Compound","BER_httk","BER_GCN","BER_RF",
                          "SEEM_mg_kg_day","concern_httk"]].to_string(index=False))
print("\nMetriken -> results/ber_all777_metrics.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plots
# ═══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: BER-Wasserfall (Chemikalien mit SEEM-Daten) ───────────────────────
ber_plot = result.dropna(subset=["BER_httk"]).copy()
ber_plot = ber_plot.sort_values("BER_httk").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(ber_plot))

# httk (grau, Referenz)
ax.bar(x, np.log10(ber_plot["BER_httk"].clip(1e-4)),
       color="#B0BEC5", alpha=0.9, label="httk (Referenz)", zorder=2)
# GCN (blau)
if ber_plot["BER_GCN"].notna().sum():
    ax.scatter(x, np.log10(ber_plot["BER_GCN"].clip(1e-4)),
               color="#1565C0", s=60, zorder=5, marker="D", label="GCN")
# RF (orange)
if ber_plot["BER_RF"].notna().sum():
    ax.scatter(x, np.log10(ber_plot["BER_RF"].clip(1e-4)),
               color="#E65100", s=40, zorder=4, marker="o", label="RF/GB")

ax.axhline(0,    color="red",    lw=2,   ls="--", label="BER=1 (Grenzwert)")
ax.axhline(1,    color="orange", lw=1.5, ls=":",  label="BER=10")
ax.axhline(2,    color="green",  lw=1.0, ls=":",  label="BER=100")

ax.set_xticks(x)
ax.set_xticklabels(ber_plot["Compound"].str[:20], rotation=55, ha="right", fontsize=7)
ax.set_ylabel("log10(BER)  [niedriger = hoehere Besorgnis]", fontsize=11)
ax.set_title("BER-Ranking: httk vs. GCN vs. RF/GB Clint\n"
             f"n={len(ber_plot)} Chemikalien mit SEEM-Expositionsdaten", fontsize=11)
ax.legend(fontsize=9, loc="upper left"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "ber_all777_waterfall.png", dpi=150)
plt.close()
print("Saved: results/ber_all777_waterfall.png")


# ── Plot 2: AED Scatter GCN vs httk (alle Chemikalien) ──────────────────────
both_aed = result.dropna(subset=["AED_httk","AED_GCN","AED_RF"])

if len(both_aed) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (gcn_col, rf_col, title, clr) in zip(axes, [
        ("AED_GCN", "AED_httk", "GCN vs. httk", "#1565C0"),
        ("AED_RF",  "AED_httk", "RF/GB vs. httk", "#E65100"),
    ]):
        log_x = np.log10(both_aed["AED_httk"].clip(1e-10))
        log_y = np.log10(both_aed[gcn_col].clip(1e-10))
        # Farbkodierung nach SEEM verfuegbar
        has_seem = both_aed["SEEM_mg_kg_day"].notna()
        ax.scatter(log_x[~has_seem], log_y[~has_seem],
                   c="#90A4AE", s=10, alpha=0.4, linewidths=0, label="ohne SEEM")
        ax.scatter(log_x[has_seem],  log_y[has_seem],
                   c=clr, s=60, alpha=0.8, linewidths=0.5, edgecolors="k",
                   zorder=5, label="mit SEEM-Daten")
        lims = [min(log_x.min(), log_y.min())-0.5,
                max(log_x.max(), log_y.max())+0.5]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.fill_between(lims, [v-np.log10(10) for v in lims],
                        [v+np.log10(10) for v in lims],
                        alpha=0.06, color="orange", label="10-fold Toleranz")
        rho, pv = spearmanr(log_x, log_y)
        fe = 10**np.abs(log_x - log_y)
        gmfe = np.exp(np.mean(np.log(fe)))
        p3 = np.mean(fe <= 3)*100
        ax.set_xlabel("log10(AED httk)  [Referenz]", fontsize=10)
        ax.set_ylabel(f"log10(AED {title.split()[0]})", fontsize=10)
        ax.set_title(f"AED: {title}  (n={len(both_aed)})\n"
                     f"Spearman rho={rho:.3f}  GMFE={gmfe:.2f}x  <=3-fold={p3:.0f}%",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    plt.suptitle("AED-Vergleich: ML-Vorhersage vs. httk-Literatur\n"
                 "(Alle 777 Chemikalien, trainiert auf 19 Piloten)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS / "ber_all777_aed_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/ber_all777_aed_scatter.png")


# ── Plot 3: BER GCN vs httk vs RF (mit SEEM) ─────────────────────────────────
if len(ber_plot) > 1:
    ber3 = ber_plot.dropna(subset=["BER_httk","BER_GCN","BER_RF"])

    if len(ber3) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        for ax, (model_col, label, clr) in zip(axes, [
            ("BER_GCN", "GCN",   "#1565C0"),
            ("BER_RF",  "RF/GB", "#E65100"),
        ]):
            log_x = np.log10(ber3["BER_httk"].clip(1e-4))
            log_y = np.log10(ber3[model_col].clip(1e-4))
            fe    = 10**np.abs(log_x - log_y)
            ax.scatter(log_x, log_y, c=clr, s=80, edgecolors="k", lw=0.5, zorder=5)
            # Chemikaliennamen beschriften
            for _, r in ber3.iterrows():
                ax.annotate(str(r["Compound"])[:16],
                            (np.log10(max(r["BER_httk"],1e-4)),
                             np.log10(max(r[model_col],1e-4))),
                            fontsize=5.5, alpha=0.8)
            lims = [min(log_x.min(),log_y.min())-0.5,
                    max(log_x.max(),log_y.max())+0.5]
            ax.plot(lims, lims, "k--", lw=1.2, label="y=x (perfekte Uebereinstimmung)")
            ax.axhline(0, color="red",    lw=1.5, ls="--", alpha=0.5)
            ax.axvline(0, color="red",    lw=1.5, ls="--", alpha=0.5)
            rho, pv = spearmanr(log_x, log_y)
            gmfe    = np.exp(np.mean(np.log(fe)))
            ax.set_xlabel("log10(BER_httk) [Referenz]",     fontsize=10)
            ax.set_ylabel(f"log10(BER_{label})", fontsize=10)
            ax.set_title(f"BER: {label} vs. httk (n={len(ber3)})\n"
                         f"Spearman rho={rho:.3f}  GMFE={gmfe:.2f}x",
                         fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

        plt.suptitle("BER-Vergleich: GCN & RF/GB vs. httk-Referenz\n"
                     f"(n={len(ber3)} Chemikalien mit SEEM-Exposition)",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(RESULTS / "ber_all777_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: results/ber_all777_comparison.png")


# ── Plot 4: AED-Verteilung aller 777 Chemikalien ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
aed_all = result.dropna(subset=["AED_httk","AED_GCN","AED_RF"])
bins = np.linspace(-5, 5, 40)
ax.hist(np.log10(aed_all["AED_httk"].clip(1e-5)), bins=bins,
        color="#607D8B", alpha=0.75, label=f"httk (n={aed_all['AED_httk'].notna().sum()})")
ax.hist(np.log10(aed_all["AED_GCN"].clip(1e-5)),  bins=bins,
        color="#1565C0", alpha=0.55, label=f"GCN  (n={aed_all['AED_GCN'].notna().sum()})")
ax.hist(np.log10(aed_all["AED_RF"].clip(1e-5)),   bins=bins,
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
    ax.set_ylabel("Anzahl Chemikalien"); ax.set_title("BER-Concern-Klassifikation (httk)")
    ax.set_xticklabels([c[:22] for c in cats], rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v+0.1, str(v), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS / "ber_all777_distribution.png", dpi=150)
plt.close()
print("Saved: results/ber_all777_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Abschlusszusammenfassung
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ABSCHLUSSZUSAMMENFASSUNG BER")
print("="*65)
print(f"  Gesamtdatensatz          : {len(result)}")
print(f"  AED berechnet (httk)     : {result['AED_httk'].notna().sum()}")
print(f"  AED berechnet (GCN)      : {result['AED_GCN'].notna().sum()}")
print(f"  AED berechnet (RF)       : {result['AED_RF'].notna().sum()}")
print(f"  BER berechnet (httk)     : {result['BER_httk'].notna().sum()}")
print(f"  BER berechnet (GCN)      : {result['BER_GCN'].notna().sum()}")

if result["BER_httk"].notna().sum():
    print(f"\n  BER-Concern (httk-Clint):")
    for cat, cnt in result["concern_httk"].value_counts().items():
        print(f"    {cat:<30}: {cnt}")

print("\nAusgaben:")
print("  results/ber_all777.csv               -- vollstaendige Tabelle")
print("  results/ber_all777_metrics.txt       -- Metriken")
print("  results/ber_all777_waterfall.png     -- BER-Wasserfall")
print("  results/ber_all777_aed_scatter.png  -- AED: GCN/RF vs. httk")
print("  results/ber_all777_comparison.png   -- BER: GCN/RF vs. httk")
print("  results/ber_all777_distribution.png -- AED/BER Verteilung")
print("\nDone.")
