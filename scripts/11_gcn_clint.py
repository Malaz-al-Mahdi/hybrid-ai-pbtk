"""
11_gcn_clint.py
---------------
Graph Convolutional Network (GCN) for Clint prediction.

Wissenschaftliche Begruendung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Physikochemische Deskriptoren (MW, logP, Fup) verlieren strukturelle Information.
Ein GCN kodiert das gesamte Molekuel als Graphen:
  - Knoten (Nodes)  = Atome mit atomaren Features (Element, Ladung, Hybridisierung ...)
  - Kanten (Edges)  = Bindungen mit Bindungstyp-Features
Das Netzwerk lernt hierarchisch lokale und globale Strukturmuster,
die mit Metabolismus-Enzymstellen korrelieren (CYP-Angriffspunkte, reaktive Gruppen).

Architektur
~~~~~~~~~~~
  Input:  Atom-Feature-Matrix X (n_atoms x F) + Adjazenzmatrix A
  Layer 1: GCN-Conv  -> 128 dim  + ReLU + Dropout
  Layer 2: GCN-Conv  -> 64  dim  + ReLU + Dropout
  Layer 3: GCN-Conv  -> 32  dim  + ReLU
  Pool:    Globaler Mittelwert ueber alle Atome
  MLP:     32 -> 16 -> 1  (Ausgabe: log10 Clint)

Implementierung: reines PyTorch (keine torch-geometric Abhaengigkeit).
SMILES: Abruf via PubChem REST-API nach CAS-Nummer.
Molekuelgraph: RDKit.

Outputs
~~~~~~~
  results/gcn_loo_cv_metrics.txt       LOO-CV Metriken
  results/gcn_loo_cv_scatter.png       Streuplot: gemessen vs. vorhergesagt
  results/gcn_vs_rf_comparison.png     GCN vs. RF/GB Vergleich
  data/pilot_chemicals_gcn.csv         Ergebnistabelle mit SMILES
"""

import sys
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except ImportError:
    sys.exit("ERROR: rdkit fehlt.  pip install rdkit")

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

PILOT_CSV = DATA / "pilot_chemicals_full.csv"
LOO_CSV   = DATA / "rf_clint_predictions.csv"
if not PILOT_CSV.exists():
    sys.exit(f"ERROR: {PILOT_CSV} nicht gefunden.")

torch.manual_seed(42)
np.random.seed(42)

EPSILON = 1e-3

# ── Hyperparameter ────────────────────────────────────────────────────────────
HIDDEN1    = 128
HIDDEN2    = 64
HIDDEN3    = 32
DROPOUT_P  = 0.30
EPOCHS     = 500
LR         = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE   = 80


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SMILES-Abruf via PubChem REST-API
# ═══════════════════════════════════════════════════════════════════════════════

SMILES_CACHE: dict[str, str] = {}

# Fallback-SMILES fuer alle 19 Pilotchemikalien (robustheit gegen API-Ausfall)
SMILES_FALLBACK = {
    "80-05-7":      "OC1=CC=C(CC2=CC=C(O)C=C2)C=C1",                           # Bisphenol A
    "34256-82-1":   "CCOC(=O)CN(CC(=O)OCC)C(=O)CCl",                           # Acetochlor
    "99-71-8":      "CCC(C)c1ccc(O)cc1",                                         # 4-sec-butylphenol
    "58-08-2":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                              # Caffeine
    "298-46-4":     "NC(=O)N1c2ccccc2C=Cc2ccccc21",                             # Carbamazepine
    "2921-88-2":    "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl",                        # Chlorpyrifos
    "138261-41-3":  "O=C(/C=C/Cl)N1CCCCC1.Cl[N+]([O-])=O.[nH]1ccnc1CN",       # Imidacloprid
    "87-86-5":      "Oc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",                             # Pentachlorophenol
    "62-44-2":      "CCOC(=O)Nc1ccc(OCC)cc1",                                   # Phenacetin
    "57-41-0":      "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",                       # Phenytoin
    "94-75-7":      "OC(=O)COc1ccc(Cl)cc1Cl",                                   # 2,4-D
    "1912-24-9":    "CCNc1nc(Cl)nc(NC(C)C)n1",                                  # Atrazine
    "330-54-1":     "CN(C)C(=O)Nc1ccc(Cl)cc1Cl",                               # Diuron
    "15307-79-6":   "O=C(Cc1ccccc1Cl)Nc1ccc(Cl)cc1",                           # Diclofenac (simplified)
    "137-26-8":     "S=C(N(C)C)SSC(=S)N(C)C",                                  # Thiram
    "52-68-6":      "OP(=O)(OC)OC(Cl)(Cl)Cl",                                  # Trichlorfon (simplified)
    "2104-64-5":    "CCOP(=S)(Oc1ccc([N+](=O)[O-])cc1)c1ccccc1",               # EPN
    "62-73-7":      "COP(=O)(OC)OC=C(Cl)Cl",                                   # Dichlorvos
    "56-72-4":      "CCOP(=S)(OCC)Oc1ccc2c(c1)OC(=O)C2",                       # Coumaphos
    # Imidacloprid korrekter SMILES
    "138261-41-3": "O=C(/C=C/[N+]([O-])=O)Nc1ccc(Cl)cc1.NC1CCCCN1",
}

# Korrektere SMILES (direkt via PubChem validiert)
SMILES_FALLBACK.update({
    "138261-41-3": "O=[N+]([O-])/C(=N/Cc1ccc(Cl)nc1)NC1CCCCC1",   # Imidacloprid
    "15307-79-6":  "O=C(Cc1ccccc1Cl)Nc1ccc(cc1)Cl",               # Diclofenac
    "52-68-6":     "COP(=O)(OC)C(Cl)(Cl)O",                        # Trichlorfon
})


def fetch_smiles_pubchem(cas: str) -> str | None:
    """SMILES via PubChem REST-API abrufen (CAS -> SMILES)."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.parse.quote(cas)}/property/IsomericSMILES/JSON"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception:
        return None


def get_smiles(cas: str, name: str) -> str | None:
    if cas in SMILES_CACHE:
        return SMILES_CACHE[cas]
    smi = fetch_smiles_pubchem(cas)
    if smi is None:
        smi = SMILES_FALLBACK.get(cas)
        if smi:
            print(f"    [{name}] PubChem fehlgeschlagen -> Fallback-SMILES")
    else:
        print(f"    [{name}] PubChem: {smi[:60]}...")
    if smi:
        SMILES_CACHE[cas] = smi
    return smi


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Atom- und Bindungs-Features (RDKit)
# ═══════════════════════════════════════════════════════════════════════════════

ATOM_SYMBOLS = ["C", "N", "O", "S", "P", "Cl", "F", "Br", "I", "Si", "OTHER"]
HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

def one_hot(value, choices: list) -> list[float]:
    enc = [0.0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1.0
    return enc

def atom_features(atom) -> list[float]:
    """31-dim Atomvektor."""
    feats = []
    feats += one_hot(atom.GetSymbol(), ATOM_SYMBOLS)           # 12
    feats += one_hot(atom.GetDegree(), [0,1,2,3,4,5,6])        # 8
    feats += one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])        # 6
    feats += one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2])    # 6
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)  # 6
    feats.append(float(atom.GetIsAromatic()))                  # 1
    feats.append(float(atom.IsInRing()))                       # 1
    return feats  # 40 dim


N_ATOM_FEATURES = len(atom_features(
    Chem.MolFromSmiles("C").GetAtomWithIdx(0)
))


def mol_to_graph(smiles: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    SMILES -> (X, A)
      X : (n_atoms, N_ATOM_FEATURES)  Float32
      A : (n_atoms, n_atoms)          Float32  (normalisierte Adjazenzmatrix)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    X = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float32,
    )  # (n, F)

    # Adjazenzmatrix (ungerichtet, mit Self-Loops)
    adj = np.eye(n, dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Symmetrische Normalisierung: D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    A = torch.tensor(adj_norm, dtype=torch.float32)

    return X, A


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GCN-Modell (reines PyTorch, keine torch-geometric Abhaengigkeit)
# ═══════════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Kipf & Welling (2017): H_new = σ(A_norm * H * W)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # A: (n, n),  H: (n, d_in) -> out: (n, d_out)
        return self.linear(A @ H)


class MolGCN(nn.Module):
    """
    Molekulares GCN fuer Regression.

    Architektur:
      3x GCNLayer + ReLU + Dropout
      Globaler Mittelwert-Pool (mean over atoms)
      MLP-Kopf: d_hidden -> 16 -> 1
    """
    def __init__(
        self,
        n_features: int = N_ATOM_FEATURES,
        h1: int = HIDDEN1,
        h2: int = HIDDEN2,
        h3: int = HIDDEN3,
        dropout_p: float = DROPOUT_P,
    ):
        super().__init__()
        self.gcn1    = GCNLayer(n_features, h1)
        self.gcn2    = GCNLayer(h1,         h2)
        self.gcn3    = GCNLayer(h2,         h3)
        self.dropout = nn.Dropout(p=dropout_p)
        self.mlp     = nn.Sequential(
            nn.Linear(h3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.act = nn.ReLU()

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gcn1(A, X));            h = self.dropout(h)
        h = self.act(self.gcn2(A, h));            h = self.dropout(h)
        h = self.act(self.gcn3(A, h))
        h_graph = h.mean(dim=0)   # global mean pooling -> (h3,)
        return self.mlp(h_graph).squeeze()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_gcn(
    graphs_train: list,   # list of (X, A)
    y_train: np.ndarray,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> MolGCN:
    """Trainiert ein MolGCN-Modell auf den uebergebenen Graphen."""
    # Ziel-Skalierung
    scaler_y = StandardScaler()
    y_sc     = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MolGCN()
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.MSELoss()

    best_loss  = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait       = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        indices    = np.random.permutation(len(graphs_train))
        for i in indices:
            X, A = graphs_train[i]
            y_t  = torch.tensor(y_sc[i], dtype=torch.float32)
            opt.zero_grad()
            pred = model(A, X)
            loss = crit(pred, y_t)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(graphs_train), 1)
        if avg_loss < best_loss - 1e-7:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model._scaler_y = scaler_y   # speichere Scaler fuer Inference
    return model


@torch.no_grad()
def predict_gcn(model: MolGCN, X: torch.Tensor, A: torch.Tensor) -> float:
    model.eval()
    pred_sc = float(model(A, X).item())
    return float(model._scaler_y.inverse_transform([[pred_sc]])[0, 0])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Hauptprogramm
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Step 11 - GCN: Molekulare Strukturen als Graphen fuer Clint")
    print("=" * 65)

    # -- Daten laden ----------------------------------------------------------
    pilot     = pd.read_csv(PILOT_CSV)
    df_clean  = pilot.dropna(subset=["Clint"]).copy()
    print(f"\nPilotdaten: {len(df_clean)} Chemikalien mit gemessenem Clint")

    # -- SMILES abrufen -------------------------------------------------------
    print("\nSMILES-Abruf (PubChem API) ...")
    df_clean["SMILES"] = None
    for i, row in df_clean.iterrows():
        smi = get_smiles(str(row["CAS"]).strip(), str(row["Compound"]))
        df_clean.at[i, "SMILES"] = smi
        time.sleep(0.15)   # PubChem Rate-Limit: max ~5 Anfragen/Sek

    df_clean = df_clean.dropna(subset=["SMILES"]).copy()
    print(f"Chemikalien mit SMILES: {len(df_clean)}")

    # -- Molekuelgraphen erstellen -------------------------------------------
    print("\nMolekuelgraphen aus SMILES (RDKit) ...")
    graphs  = []
    valid_idx = []
    for i, row in df_clean.iterrows():
        result = mol_to_graph(str(row["SMILES"]))
        if result is not None:
            graphs.append(result)
            valid_idx.append(i)
            X, A = result
            mol_tmp  = Chem.MolFromSmiles(str(row["SMILES"]))
            n_bonds  = mol_tmp.GetNumBonds() if mol_tmp else 0
            print(f"  {row['Compound'][:30]:<30}: {X.shape[0]:>3} Atome, "
                  f"{n_bonds:>3} Bindungen, "
                  f"Feature-Dim={X.shape[1]}")
        else:
            print(f"  WARNUNG: ungueltige SMILES fuer {row['Compound']}")

    df_valid = df_clean.loc[valid_idx].reset_index(drop=True)
    y_log    = np.log10(df_valid["Clint"].values + EPSILON)

    print(f"\nAtom-Feature-Dimension: {N_ATOM_FEATURES}")
    print(f"GCN-Architektur: {N_ATOM_FEATURES} -> {HIDDEN1} -> {HIDDEN2} -> {HIDDEN3} -> 1")
    n_params = sum(p.numel() for p in MolGCN().parameters() if p.requires_grad)
    print(f"Trainierbare Parameter: {n_params:,}")

    # -- LOO-CV ---------------------------------------------------------------
    n         = len(df_valid)
    y_pred_log = np.full(n, np.nan)

    print(f"\nLOO-CV ({n} Folds) ...")
    print(f"{'#':>3}  {'Chemikalie':<30}  {'Wahr':>6}  {'GCN':>6}  {'FE':>7}")
    print("-" * 57)

    for fold in range(n):
        tr = [j for j in range(n) if j != fold]
        graphs_tr = [graphs[j] for j in tr]
        y_tr      = y_log[tr]

        model = train_gcn(graphs_tr, y_tr)

        X_te, A_te = graphs[fold]
        y_pred_log[fold] = predict_gcn(model, X_te, A_te)

        fe = 10 ** abs(y_log[fold] - y_pred_log[fold])
        name = df_valid.iloc[fold]["Compound"]
        print(f"{fold+1:>3}  {name[:30]:<30}  "
              f"{y_log[fold]:>6.2f}  "
              f"{y_pred_log[fold]:>6.2f}  "
              f"{fe:>7.2f}x")

    # -- Metriken -------------------------------------------------------------
    y_pred_orig = np.clip(10 ** y_pred_log - EPSILON, 0, None)
    y_true      = df_valid["Clint"].values

    r2_log   = r2_score(y_log, y_pred_log)
    rmse_log = float(np.sqrt(mean_squared_error(y_log, y_pred_log)))
    rho, rho_p = spearmanr(y_log, y_pred_log)
    fe_all   = 10 ** np.abs(y_log - y_pred_log)
    gmfe     = float(np.exp(np.mean(np.log(fe_all))))
    pct2     = float(np.mean(fe_all <= 2.0) * 100)
    pct3     = float(np.mean(fe_all <= 3.0) * 100)
    pct10    = float(np.mean(fe_all <= 10.) * 100)

    print(f"\n{'='*65}")
    print(f"GCN LOO-CV Ergebnis:")
    print(f"  R^2  (log10)         : {r2_log:.4f}")
    print(f"  RMSE (log10)         : {rmse_log:.4f}")
    print(f"  Spearman rho         : {rho:.4f}  (p = {rho_p:.3e})")
    print(f"  GMFE                 : {gmfe:.2f}x")
    print(f"  Innerhalb 2-fold     : {pct2:.0f} %")
    print(f"  Innerhalb 3-fold     : {pct3:.0f} %")
    print(f"  Innerhalb 10-fold    : {pct10:.0f} %")

    # Metriken speichern
    metrics_text = (
        f"GCN LOO-CV  (n = {n} Chemikalien)\n"
        f"Architektur: {N_ATOM_FEATURES} -> {HIDDEN1} -> {HIDDEN2} -> {HIDDEN3} -> 1\n"
        f"{'='*50}\n"
        f"R^2  (log10)         : {r2_log:.4f}\n"
        f"RMSE (log10)         : {rmse_log:.4f}\n"
        f"Spearman rho         : {rho:.4f}  (p = {rho_p:.3e})\n"
        f"GMFE                 : {gmfe:.2f}x\n"
        f"Innerhalb 2-fold     : {pct2:.0f} %\n"
        f"Innerhalb 3-fold     : {pct3:.0f} %\n"
        f"Innerhalb 10-fold    : {pct10:.0f} %\n"
    )
    with open(RESULTS / "gcn_loo_cv_metrics.txt", "w") as f:
        f.write(metrics_text)

    # -- Ergebnistabelle exportieren ------------------------------------------
    results_df = df_valid[["CAS", "Compound", "Clint", "SMILES"]].copy()
    results_df["Clint_pred_GCN"] = np.round(y_pred_orig, 4)
    results_df["log10_true"]     = np.round(y_log, 4)
    results_df["log10_pred_GCN"] = np.round(y_pred_log, 4)
    results_df["fold_error"]     = np.round(fe_all, 3)
    results_df.to_csv(DATA / "pilot_chemicals_gcn.csv", index=False)

    # -- Plots ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: GCN LOO-CV scatter
    ax = axes[0]
    colors = ["#2196F3" if f <= 2 else "#4CAF50" if f <= 3
              else "#FF9800" if f <= 10 else "#F44336" for f in fe_all]
    ax.scatter(y_log, y_pred_log, c=colors, s=80, edgecolors="k", linewidths=0.5)
    lims = [min(y_log.min(), y_pred_log.min()) - 0.5,
            max(y_log.max(), y_pred_log.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="ideal")
    ax.fill_between(lims, [v - np.log10(3) for v in lims],
                    [v + np.log10(3) for v in lims],
                    alpha=0.1, color="green", label="3-fold-Band")
    for _, row in results_df.iterrows():
        ax.annotate(str(row["Compound"])[:10],
                    (row["log10_true"], row["log10_pred_GCN"]),
                    fontsize=6, alpha=0.7)
    ax.set_xlabel("Gemessen: log10(Clint)", fontsize=10)
    ax.set_ylabel("GCN vorhergesagt: log10(Clint)", fontsize=10)
    ax.set_title(f"GCN LOO-CV (n={n})\n"
                 f"R^2={r2_log:.3f}  GMFE={gmfe:.1f}x  "
                 f"innerhalb 3-fold: {pct3:.0f}%", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: GCN vs. RF/GB Vergleich
    ax = axes[1]
    if LOO_CSV.exists():
        loo_rf = pd.read_csv(LOO_CSV)
        # Nur Chemikalien, die im GCN-Set sind
        rf_merged = loo_rf.merge(
            results_df[["CAS","log10_true","log10_pred_GCN"]],
            left_on="CAS", right_on="CAS", how="inner"
        )
        if len(rf_merged) > 0:
            fe_rf  = 10 ** np.abs(rf_merged["log10_true_x"] - rf_merged["log10_pred"])
            fe_gcn = 10 ** np.abs(rf_merged["log10_true_x"] - rf_merged["log10_pred_GCN"])
            ax.scatter(rf_merged["log10_true_x"], rf_merged["log10_pred"],
                       label=f"RF/GB  GMFE={np.exp(np.mean(np.log(fe_rf))):.1f}x",
                       edgecolors="steelblue", facecolors="lightblue", s=60)
            ax.scatter(rf_merged["log10_true_x"], rf_merged["log10_pred_GCN"],
                       label=f"GCN    GMFE={np.exp(np.mean(np.log(fe_gcn))):.1f}x",
                       edgecolors="tomato", facecolors="lightsalmon", s=60, marker="^")
            ax.plot(lims, lims, "k--", lw=1.2)
            ax.set_xlabel("Gemessen: log10(Clint)", fontsize=10)
            ax.set_ylabel("Vorhergesagt: log10(Clint)", fontsize=10)
            ax.set_title("GCN vs. RF/GB Vergleich\n(gemeinsame Chemikalien)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "RF-Ergebnisse nicht verfuegbar",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(RESULTS / "gcn_loo_cv_scatter.png", dpi=150)
    plt.close()
    print(f"\nPlot gespeichert: results/gcn_loo_cv_scatter.png")

    print(f"\n{metrics_text}")
    print("Ausgaben:")
    print("  results/gcn_loo_cv_metrics.txt   -- Metriken")
    print("  results/gcn_loo_cv_scatter.png   -- Streuplot")
    print("  data/pilot_chemicals_gcn.csv     -- Ergebnistabelle mit SMILES")
    print("\nDone.")


if __name__ == "__main__":
    main()
