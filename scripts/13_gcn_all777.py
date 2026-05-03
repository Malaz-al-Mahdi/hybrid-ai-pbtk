"""
13_gcn_all777.py
----------------
GCN + RF/GB Analyse aller 777 httk-Chemikalien.

Vorgehen
~~~~~~~~
  1. SMILES-Abruf (multi-source):
       PubChem REST  -> cactus.nci.nih.gov (CIR) -> lokal gespeichert
     Resultat wird in data/smiles_cache_777.csv zwischengespeichert.
  2. Molekuelgraphen (RDKit) fuer alle Chemikalien mit gueltigen SMILES
  3. GCN auf 19 Pilotchemikalien trainieren, auf alle 777 mit SMILES anwenden
  4. RF/GB (MW/logP/Fup – kein SMILES noetig) auf ALLE 777 anwenden
  5. Metriken und Plots

Outputs
~~~~~~~
  data/smiles_cache_777.csv
  results/gcn_777_predictions.csv
  results/gcn_777_metrics.txt
  results/gcn_777_scatter.png
  results/gcn_777_clint_distribution.png
"""

import sys, time, json, warnings
import urllib.request, urllib.parse
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except ImportError:
    sys.exit("ERROR: rdkit fehlt.  pip install rdkit")

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PILOT_CSV    = DATA / "pilot_chemicals_full.csv"
FULL_CSV     = DATA / "all_777_chemicals.csv"
SMILES_CACHE = DATA / "smiles_cache_777.csv"
PILOT_GCN    = DATA / "pilot_chemicals_gcn.csv"   # hat bereits SMILES fuer Pilot

EPSILON  = 1e-3
torch.manual_seed(42); np.random.seed(42)

# ── GCN Hyperparameter ────────────────────────────────────────────────────────
HIDDEN1 = 128; HIDDEN2 = 64; HIDDEN3 = 32
DROPOUT_P = 0.30; EPOCHS = 500; LR = 5e-4; WD = 1e-4; PATIENCE = 80

print("=" * 65)
print("Step 13 - GCN + RF/GB auf allen 777 httk-Chemikalien")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SMILES-Abruf (multi-source, kurze Timeouts)
# ═══════════════════════════════════════════════════════════════════════════════

def _get(url: str, timeout: int = 5) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None

def pubchem_smiles(cas: str) -> str | None:
    raw = _get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.parse.quote(cas)}/property/IsomericSMILES/JSON",
        timeout=4,
    )
    if raw:
        try:
            return json.loads(raw)["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        except Exception:
            pass
    return None

def cir_smiles(cas: str) -> str | None:
    """NCI Chemical Identifier Resolver (cactus.nci.nih.gov)."""
    raw = _get(
        f"https://cactus.nci.nih.gov/chemical/structure/"
        f"{urllib.parse.quote(cas)}/smiles",
        timeout=6,
    )
    if raw:
        smi = raw.decode("utf-8", errors="ignore").strip().split()[0]
        if smi and Chem.MolFromSmiles(smi):
            return smi
    return None

def fetch_smiles_for(cas: str) -> str | None:
    smi = pubchem_smiles(cas)
    if smi:
        return smi
    smi = cir_smiles(cas)
    return smi


def load_or_fetch_smiles(all_cas: list[str]) -> dict[str, str]:
    """
    Gibt {cas -> smiles} zurueck.
    Laedt zuerst den Cache, fragt dann fuer fehlende CAS bei externen APIs.
    """
    # --- Cache laden ---
    if SMILES_CACHE.exists():
        cache_df = pd.read_csv(SMILES_CACHE, dtype=str)
        cache = dict(zip(cache_df["CAS"], cache_df["SMILES"]))
        cache = {k: v for k, v in cache.items() if pd.notna(v) and v != "nan"}
    else:
        cache = {}

    # --- Pilot-SMILES schon bekannt ---
    if PILOT_GCN.exists():
        pg = pd.read_csv(PILOT_GCN, dtype=str)
        for _, row in pg.iterrows():
            if pd.notna(row.get("SMILES")) and row["SMILES"] != "nan":
                cache[row["CAS"]] = row["SMILES"]

    missing = [c for c in all_cas if c not in cache]
    print(f"Im Cache: {len(cache)}  |  Fehlend: {len(missing)}")

    if missing:
        # Teste zuerst, ob irgendeine API erreichbar ist
        test_smi = pubchem_smiles("80-05-7")   # Bisphenol-A
        if not test_smi:
            test_smi = cir_smiles("80-05-7")
        api_ok = test_smi is not None
        print(f"Netzwerk-Test (Bisphenol-A): {'OK -> ' + test_smi[:30] if api_ok else 'KEIN ZUGRIFF'}")

        if api_ok:
            print(f"Starte Abruf fuer {len(missing)} CAS-Nummern ...")
            new_rows = []
            for i, cas in enumerate(missing):
                smi = fetch_smiles_for(cas)
                if smi:
                    cache[cas] = smi
                    new_rows.append({"CAS": cas, "SMILES": smi})
                time.sleep(0.15)
                if (i+1) % 50 == 0:
                    pct = (i+1) / len(missing) * 100
                    found = sum(1 for c in missing[:i+1] if c in cache)
                    print(f"  {i+1:>4}/{len(missing)}  gefunden={found}  ({pct:.0f}%)",
                          flush=True)
                    # Cache nach jeder 50er-Runde sichern
                    _save_cache(cache)
            print(f"Neu abgerufen: {len(new_rows)}")
        else:
            print("HINWEIS: Kein Netzwerkzugriff – verwende vorhandenen Cache")

        _save_cache(cache)

    print(f"SMILES verfuegbar: {len(cache)} / {len(all_cas)}")
    return cache

def _save_cache(cache: dict):
    df = pd.DataFrame(list(cache.items()), columns=["CAS","SMILES"])
    df.to_csv(SMILES_CACHE, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Molekuelgraph
# ═══════════════════════════════════════════════════════════════════════════════

ATOM_SYMBOLS  = ["C","N","O","S","P","Cl","F","Br","I","Si","OTHER"]
HYBRID_TYPES  = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
                 rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
                 rdchem.HybridizationType.SP3D2]

def one_hot(v, choices):
    e = [0.0] * (len(choices)+1)
    e[choices.index(v) if v in choices else len(choices)] = 1.0
    return e

def atom_features(a):
    f  = one_hot(a.GetSymbol(),       ATOM_SYMBOLS)
    f += one_hot(a.GetDegree(),        [0,1,2,3,4,5,6])
    f += one_hot(a.GetTotalNumHs(),    [0,1,2,3,4])
    f += one_hot(a.GetFormalCharge(),  [-2,-1,0,1,2])
    f += one_hot(a.GetHybridization(), HYBRID_TYPES)
    f.append(float(a.GetIsAromatic()))
    f.append(float(a.IsInRing()))
    return f

_N_FEAT = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))

def mol_to_graph(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    n   = mol.GetNumAtoms()
    X   = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    adj = np.eye(n, dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i,j] = adj[j,i] = 1.0
    deg = adj.sum(1)
    D   = np.diag(1./np.sqrt(np.maximum(deg, 1e-9)))
    A   = torch.tensor(D @ adj @ D, dtype=torch.float32)
    return X, A


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GCN-Modell
# ═══════════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.w = nn.Linear(i, o)
    def forward(self, A, H):
        return self.w(A @ H)

class MolGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1  = GCNLayer(_N_FEAT, HIDDEN1)
        self.g2  = GCNLayer(HIDDEN1,  HIDDEN2)
        self.g3  = GCNLayer(HIDDEN2,  HIDDEN3)
        self.drop = nn.Dropout(DROPOUT_P)
        self.mlp  = nn.Sequential(nn.Linear(HIDDEN3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.act  = nn.ReLU()
    def forward(self, A, X):
        h = self.drop(self.act(self.g1(A, X)))
        h = self.drop(self.act(self.g2(A, h)))
        h = self.act(self.g3(A, h))
        return self.mlp(h.mean(0)).squeeze()

def train_gcn(graphs, y_log):
    sc = StandardScaler()
    ys = sc.fit_transform(y_log.reshape(-1,1)).ravel()
    m  = MolGCN()
    op = optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    cr = nn.MSELoss()
    best, best_st, wait = float("inf"), {}, 0
    m.train()
    for ep in range(EPOCHS):
        loss_ep = 0.
        for i in np.random.permutation(len(graphs)):
            X, A = graphs[i]
            op.zero_grad()
            l = cr(m(A, X), torch.tensor(ys[i], dtype=torch.float32))
            if torch.isfinite(l):
                l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
                loss_ep += l.item()
        avg = loss_ep / max(len(graphs), 1)
        if avg < best - 1e-7:
            best, best_st, wait = avg, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            print(f"  Fruehzeitiger Stopp: Epoche {ep+1}")
            break
    m.load_state_dict(best_st)
    m._sc = sc
    return m

@torch.no_grad()
def predict_gcn(m, X, A):
    m.eval()
    raw = float(m(A, X).item())
    return float(m._sc.inverse_transform([[raw]])[0,0])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RF/GB Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def engineer(df_in):
    mw   = np.clip(pd.to_numeric(df_in["MW"],   errors="coerce").fillna(300).values, 1., None)
    lgp  = pd.to_numeric(df_in["logP"], errors="coerce").fillna(2.).values
    fup  = np.clip(pd.to_numeric(df_in["Fup"],  errors="coerce").fillna(.1).values, 1e-6, 1.)
    return np.column_stack([
        np.log10(mw), lgp, lgp**2, np.log10(fup+1e-6), np.sqrt(fup),
        mw*lgp, mw*fup, lgp*fup, mw,
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Daten laden
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
# 6. SMILES laden / abrufen
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- SMILES-Abruf ---")
all_cas    = full["CAS"].tolist()
smiles_map = load_or_fetch_smiles(all_cas)

full["SMILES"] = full["CAS"].map(smiles_map)
n_smiles = full["SMILES"].notna().sum()
print(f"Chemikalien mit SMILES: {n_smiles} / {len(full)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GCN trainieren (auf Pilot-Set)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- GCN Trainings-Graphen ---")
# Pilot-SMILES aus gespeicherter CSV
pilot_gcn = pd.read_csv(PILOT_GCN)

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
    fe   = 10**np.abs(log_lit - log_pred)
    r2   = r2_score(log_lit, log_pred)
    rmse = float(np.sqrt(mean_squared_error(log_lit, log_pred)))
    rho, p  = spearmanr(log_lit, log_pred)
    gmfe = float(np.exp(np.mean(np.log(fe))))
    p2   = float(np.mean(fe<=2.)*100); p3 = float(np.mean(fe<=3.)*100)
    p10  = float(np.mean(fe<=10.)*100)
    print(f"\n  [{label}]  n={len(log_lit)}")
    print(f"    R^2       : {r2:.4f}")
    print(f"    RMSE log10: {rmse:.4f}")
    print(f"    Spearman  : {rho:.4f}  (p={p:.3e})")
    print(f"    GMFE      : {gmfe:.2f}x")
    print(f"    <=2-fold  : {p2:.0f}%  |  <=3-fold: {p3:.0f}%  |  <=10-fold: {p10:.0f}%")
    return dict(Modell=label, N=len(log_lit), R2=round(r2,4), RMSE_log=round(rmse,4),
                Spearman=round(rho,4), GMFE=round(gmfe,2),
                Pct_2fold=round(p2,1), Pct_3fold=round(p3,1), Pct_10fold=round(p10,1))

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
print("ABSCHLUSSZUSAMMENFASSUNG")
print("="*65)
print(f"\n  Gesamtdatensatz       : {len(full)} Chemikalien")
print(f"  Mit Lit-Clint         : {len(has_lit)}")
print(f"  SMILES verfuegbar     : {n_smiles} ({n_smiles/len(full)*100:.0f}%)")
print(f"  GCN-Vorhersagen       : {n_gcn_ok}")
print(f"  RF/GB-Vorhersagen     : {len(result_df)}")
print()
print(metrics_df[["Modell","N","R2","RMSE_log","GMFE",
                   "Pct_3fold","Pct_10fold"]].to_string(index=False))
print()
print("Ausgaben:")
print("  data/smiles_cache_777.csv           -- SMILES-Cache")
print("  results/gcn_777_predictions.csv    -- Vorhersagen")
print("  results/gcn_777_metrics.txt        -- Metriken")
print("  results/gcn_777_scatter.png        -- Scatter-Plot")
print("  results/gcn_777_clint_distribution.png -- Verteilung + Ausreisser")
print("\nDone.")
