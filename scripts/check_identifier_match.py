"""
check_identifier_match.py
--------------------------
Prueft die Identifier-Uebereinstimmung (CAS / DTXSID) zwischen:
  - data/all_777_chemicals.csv      (Quelle: httk, mit SEEM-Expositionsdaten)
  - results/aed_ber_full.csv        (Ergebnis: RTK-Pipeline, AED + BER)
  - data/pilot_chemicals_imputed.csv (Pilotset, 19 Chemikalien)

Ausserdem:
  - Zeigt welche CAS-Nummern in den Expositionsdaten vorhanden sind
  - Prueft ob die Spaltennamen fuer Exposure konsistent sind
  - Speichert einen Abgleichsbericht: results/identifier_match_report.csv
"""

from pathlib import Path
import pandas as pd
import re

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

# ── 1. Dateien laden ──────────────────────────────────────────────────────────

print("=" * 65)
print("Identifier-Abgleich: 777 Chemikalien vs. Expositionsdaten")
print("=" * 65)

df777   = pd.read_csv(DATA    / "all_777_chemicals.csv")
ber_df  = pd.read_csv(RESULTS / "aed_ber_full.csv")
pilot   = pd.read_csv(DATA    / "pilot_chemicals_imputed.csv")

print(f"\nall_777_chemicals.csv : {len(df777)} Zeilen")
print(f"aed_ber_full.csv      : {len(ber_df)} Zeilen")
print(f"pilot_chemicals_imputed.csv: {len(pilot)} Zeilen")

# ── 2. CAS-Normalisierung ─────────────────────────────────────────────────────

def normalize_cas(s):
    """Entfernt fuehrende Nullen, Leerzeichen und wandelt in String um."""
    if pd.isna(s):
        return ""
    s = str(s).strip().strip('"')
    # Fuehrende Nullen in jedem Segment entfernen
    parts = s.split("-")
    try:
        parts = [str(int(p)) for p in parts]
    except Exception:
        pass
    return "-".join(parts)

df777["CAS_norm"]  = df777["CAS"].apply(normalize_cas)
ber_df["CAS_norm"] = ber_df["CAS"].apply(normalize_cas)
pilot["CAS_norm"]  = pilot["CAS"].apply(normalize_cas)

# ── 3. Spaltencheck (Expositionsdaten) ───────────────────────────────────────

print("\n\n--- Spaltennamen in all_777_chemicals.csv (Exposure-relevant) ---")
expo_cols_777 = [c for c in df777.columns if any(
    kw in c.lower() for kw in ["seem", "exposure", "nhanes", "mg_kg"])]
print("  " + ", ".join(expo_cols_777) if expo_cols_777 else "  (keine gefunden)")

print("\n--- Spaltennamen in aed_ber_full.csv (Exposure-relevant) ---")
expo_cols_ber = [c for c in ber_df.columns if any(
    kw in c.lower() for kw in ["seem", "exposure", "nhanes", "mg_kg"])]
print("  " + ", ".join(expo_cols_ber) if expo_cols_ber else "  (keine gefunden)")

# ── 4. CAS-Ueberlappung: 777 vs. BER-Tabelle ─────────────────────────────────

cas_777    = set(df777["CAS_norm"])
cas_ber    = set(ber_df["CAS_norm"])
cas_pilot  = set(pilot["CAS_norm"])

in_both         = cas_777 & cas_ber
only_in_777     = cas_777 - cas_ber
only_in_ber     = cas_ber - cas_777
pilot_in_ber    = cas_pilot & cas_ber
pilot_not_in_ber = cas_pilot - cas_ber

print(f"\n\n--- CAS-Abgleich: all_777 vs. aed_ber_full ---")
print(f"  In beiden Dateien         : {len(in_both)}")
print(f"  Nur in all_777            : {len(only_in_777)}")
print(f"  Nur in aed_ber_full       : {len(only_in_ber)}")

print(f"\n--- Pilotset ({len(cas_pilot)} Chemikalien) ---")
print(f"  Im BER-Ergebnis gefunden  : {len(pilot_in_ber)}")
print(f"  Im BER-Ergebnis NICHT gefunden: {len(pilot_not_in_ber)}")
if pilot_not_in_ber:
    missing_names = pilot[pilot["CAS_norm"].isin(pilot_not_in_ber)][["CAS", "Compound"]]
    print("  Fehlende Pilotchemikalien:")
    print(missing_names.to_string(index=False))

# ── 5. Expositionsdaten-Check in aed_ber_full ────────────────────────────────

print(f"\n--- Exposure-Vollstaendigkeit in aed_ber_full.csv ---")
if "SEEM_mg_kg_day" in ber_df.columns:
    n_with_exp = ber_df["SEEM_mg_kg_day"].notna().sum()
    print(f"  Zeilen mit SEEM_mg_kg_day (nicht-NA): {n_with_exp} / {len(ber_df)}")
    n_with_ber = ber_df["BER"].notna().sum()
    print(f"  Zeilen mit BER (nicht-NA)           : {n_with_ber} / {len(ber_df)}")
if "Exposure_median_mg_kg_day" in ber_df.columns:
    print(f"  Spalte 'Exposure_median_mg_kg_day' gefunden (OK fuer Schritt 8)")
else:
    print(f"\n  WARNUNG: Spalte 'Exposure_median_mg_kg_day' fehlt in aed_ber_full.csv!")
    print(f"  Schritt 8 (08_bayesian_ber.py) sucht nach dieser Spalte.")
    if "SEEM_mg_kg_day" in ber_df.columns:
        print(f"  Vorhandene Spalte: 'SEEM_mg_kg_day'  -> Spaltennamen-Mismatch!")
        print(f"  Loesung: Spalte umbenennen oder Skript anpassen.")

# ── 6. SEEM-Abdeckung in all_777 ─────────────────────────────────────────────

print(f"\n--- SEEM-Abdeckung in all_777_chemicals.csv ---")
if "has_SEEM" in df777.columns:
    n_seem = (df777["has_SEEM"] == True).sum()
    print(f"  Chemikalien mit has_SEEM=TRUE: {n_seem} / {len(df777)}")
if "SEEM_mg_kg_day" in df777.columns:
    n_seem_val = df777["SEEM_mg_kg_day"].notna().sum()
    print(f"  Chemikalien mit SEEM-Wert (nicht-NA): {n_seem_val} / {len(df777)}")

# ── 7. DTXSID-Check ──────────────────────────────────────────────────────────

print(f"\n--- DTXSID-Verfuegbarkeit ---")
if "DTXSID" in df777.columns:
    n_dtxsid = df777["DTXSID"].notna().sum()
    print(f"  all_777: {n_dtxsid} / {len(df777)} Chemikalien haben DTXSID")
if "DTXSID" in ber_df.columns:
    print(f"  aed_ber_full: DTXSID-Spalte vorhanden")
else:
    print(f"  aed_ber_full: DTXSID-Spalte NICHT vorhanden (nur CAS)")

# ── 8. Detailbericht: Chemikalien OHNE Expositionsdaten ──────────────────────

no_exp = ber_df[ber_df["SEEM_mg_kg_day"].isna() | (ber_df["SEEM_mg_kg_day"] == "NA")].copy() \
    if "SEEM_mg_kg_day" in ber_df.columns else pd.DataFrame()

print(f"\n--- Chemikalien OHNE Expositionsdaten in aed_ber_full.csv ---")
print(f"  Anzahl: {len(no_exp)} / {len(ber_df)} Zeilen")

# ── 9. Bericht speichern ─────────────────────────────────────────────────────

# Merge: 777 + BER + Expositionsstatus
merge = df777[["DTXSID", "CAS", "Compound", "CAS_norm",
               "has_SEEM", "SEEM_mg_kg_day"]].copy()
ber_sub = ber_df[ber_df["Track"] == "httk_native"][
    ["CAS_norm", "AED_median", "BER", "concern"]].copy()

report = merge.merge(ber_sub, on="CAS_norm", how="left")
report["in_aed_ber"] = report["CAS_norm"].isin(cas_ber)
report["has_BER"]    = report["BER"].notna()
report = report.drop(columns=["CAS_norm"])

report_path = RESULTS / "identifier_match_report.csv"
report.to_csv(report_path, index=False)
print(f"\n\nAbgleichsbericht gespeichert: {report_path}")
print(f"  Spalten: {list(report.columns)}")

# ── 10. Zusammenfassung ───────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("ZUSAMMENFASSUNG")
print(f"{'='*65}")
print(f"  777 Chemikalien in all_777_chemicals.csv    : {len(df777)}")
print(f"  Davon mit CAS-Match in aed_ber_full.csv     : {len(in_both)}")
print(f"  Davon mit SEEM-Expositionsdaten             : {report['has_BER'].sum()}")

if "SEEM_mg_kg_day" in ber_df.columns and "Exposure_median_mg_kg_day" not in ber_df.columns:
    print(f"\n  HANDLUNGSBEDARF:")
    print(f"  Spalte 'SEEM_mg_kg_day' muss in Schritt 8 als 'Exposure_median_mg_kg_day'")
    print(f"  gelesen werden, damit BER-Posterior-Schaetzungen moeglich sind.")

print(f"\nDone.")
