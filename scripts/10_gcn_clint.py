import sys
import time
import json
import warnings
import urllib.request
import urllib.parse
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from rdkit import Chem
    from rdkit.Chem import rdchem
    _GCN_AVAILABLE = True
except ImportError:
    _GCN_AVAILABLE = False
    print("WARNUNG: rdkit/torch nicht verfuegbar - GCN wird uebersprungen, RF/GB wird verwendet.")

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

ALL_777_CSV = DATA / "all_777_chemicals.csv"
if not ALL_777_CSV.exists():
    sys.exit(f"ERROR: {ALL_777_CSV} nicht gefunden. Erst 01_extract_httk_data.R ausfuehren.")

TRAIN_SIZE = 500
TEST_SIZE  = 44

if _GCN_AVAILABLE:
    torch.manual_seed(42)
np.random.seed(42)

EPSILON = 1e-3

HIDDEN1    = 128
HIDDEN2    = 64
HIDDEN3    = 32
DROPOUT_P  = 0.30
EPOCHS     = 500
LR         = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE   = 80

SMILES_CACHE: dict[str, str] = {}

SMILES_FALLBACK = {
    "80-05-7":      "OC1=CC=C(CC2=CC=C(O)C=C2)C=C1",
    "34256-82-1":   "CCOC(=O)CN(CC(=O)OCC)C(=O)CCl",
    "99-71-8":      "CCC(C)c1ccc(O)cc1",
    "58-08-2":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "298-46-4":     "NC(=O)N1c2ccccc2C=Cc2ccccc21",
    "2921-88-2":    "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl",
    "138261-41-3":  "O=C(/C=C/Cl)N1CCCCC1.Cl[N+]([O-])=O.[nH]1ccnc1CN",
    "87-86-5":      "Oc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",
    "62-44-2":      "CCOC(=O)Nc1ccc(OCC)cc1",
    "57-41-0":      "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "94-75-7":      "OC(=O)COc1ccc(Cl)cc1Cl",
    "1912-24-9":    "CCNc1nc(Cl)nc(NC(C)C)n1",
    "330-54-1":     "CN(C)C(=O)Nc1ccc(Cl)cc1Cl",
    "15307-79-6":   "O=C(Cc1ccccc1Cl)Nc1ccc(Cl)cc1",
    "137-26-8":     "S=C(N(C)C)SSC(=S)N(C)C",
    "52-68-6":      "OP(=O)(OC)OC(Cl)(Cl)Cl",
    "2104-64-5":    "CCOP(=S)(Oc1ccc([N+](=O)[O-])cc1)c1ccccc1",
    "62-73-7":      "COP(=O)(OC)OC=C(Cl)Cl",
    "56-72-4":      "CCOP(=S)(OCC)Oc1ccc2c(c1)OC(=O)C2",
    "138261-41-3": "O=C(/C=C/[N+]([O-])=O)Nc1ccc(Cl)cc1.NC1CCCCN1",
}

SMILES_FALLBACK.update({
    "138261-41-3": "O=[N+]([O-])/C(=N/Cc1ccc(Cl)nc1)NC1CCCCC1",
    "15307-79-6":  "O=C(Cc1ccccc1Cl)Nc1ccc(cc1)Cl",
    "52-68-6":     "COP(=O)(OC)C(Cl)(Cl)O",
})

def fetch_smiles_pubchem(cas: str) -> str | None:
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

if _GCN_AVAILABLE:
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
        feats = []
        feats += one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        feats += one_hot(atom.GetDegree(), [0,1,2,3,4,5,6])
        feats += one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])
        feats += one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2])
        feats += one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)
        feats.append(float(atom.GetIsAromatic()))
        feats.append(float(atom.IsInRing()))
        return feats

    N_ATOM_FEATURES = len(atom_features(
        Chem.MolFromSmiles("C").GetAtomWithIdx(0)
    ))

if _GCN_AVAILABLE:
    def mol_to_graph(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        n = mol.GetNumAtoms()
        X = torch.tensor(
            [atom_features(a) for a in mol.GetAtoms()],
            dtype=torch.float32,
        )

        adj = np.eye(n, dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        deg = adj.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
        A = torch.tensor(adj_norm, dtype=torch.float32)

        return X, A

if _GCN_AVAILABLE:
    class GCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim, bias=True)

        def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
            return self.linear(A @ H)

    class MolGCN(nn.Module):
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
            h_graph = h.mean(dim=0)
            return self.mlp(h_graph).squeeze()

    def train_gcn(
        graphs_train: list,
        y_train: np.ndarray,
        epochs: int = EPOCHS,
        patience: int = PATIENCE,
    ):
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
        model._scaler_y = scaler_y
        return model

    @torch.no_grad()
    def predict_gcn(model, X: torch.Tensor, A: torch.Tensor) -> float:
        model.eval()
        pred_sc = float(model(A, X).item())
        return float(model._scaler_y.inverse_transform([[pred_sc]])[0, 0])

def main():
    print("=" * 65)
    print("Step 11 - GCN: Molekulare Strukturen als Graphen fuer Clint")
    print("=" * 65)

    full = pd.read_csv(ALL_777_CSV)
    full = full.rename(columns={"Human.Clint": "Clint",
                                "Human.Funbound.plasma": "Fup"})
    for col in ["Clint", "Fup", "MW", "logP"]:
        full[col] = pd.to_numeric(full[col], errors="coerce")
    full["Fup"] = full["Fup"].clip(lower=1e-6)
    full["CAS"] = full["CAS"].astype(str).str.strip()

    measured = full[full["Clint"] > 0].copy().reset_index(drop=True)
    measured["log10_Clint"] = np.log10(measured["Clint"] + EPSILON)
    measured["strat_bin"]   = pd.cut(measured["log10_Clint"], bins=6, labels=False)

    train_df, test_df = train_test_split(
        measured,
        test_size=TEST_SIZE,
        stratify=measured["strat_bin"],
        random_state=42,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"\nGesamtdatensatz: {len(measured)} Chemikalien mit gemessenem Clint")
    print(f"Training : {len(train_df)} Chemikalien")
    print(f"Test     : {len(test_df)} Chemikalien")

    if not _GCN_AVAILABLE:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import engineer_features
        X_tr = engineer_features(train_df)
        X_te = engineer_features(test_df)
        y_tr = np.log10(train_df["Clint"].values + EPSILON)
        y_te = np.log10(test_df["Clint"].values + EPSILON)

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred_tr = rf.predict(X_tr)
        y_pred_te = rf.predict(X_te)

        fe_te   = 10 ** np.abs(y_te - y_pred_te)
        r2_te   = r2_score(y_te, y_pred_te)
        rmse_te = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))
        rho_te, rho_p_te = spearmanr(y_te, y_pred_te)
        gmfe_te = float(np.exp(np.mean(np.log(fe_te))))
        pct3_te = float(np.mean(fe_te <= 3.0) * 100)

        def fe_color(fe_arr):
            return ["#2196F3" if f <= 2 else "#4CAF50" if f <= 3
                    else "#FF9800" if f <= 10 else "#F44336" for f in fe_arr]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_te, y_pred_te, c=fe_color(fe_te), s=70,
                   edgecolors="k", linewidths=0.5, zorder=4)
        ax.scatter(y_tr, y_pred_tr, c="gold", s=20, edgecolors="k",
                   linewidths=0.3, alpha=0.4, zorder=2,
                   label=f"Training (n={len(train_df)})")
        all_vals = np.concatenate([y_tr, y_te, y_pred_tr, y_pred_te])
        lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.fill_between(lims,
                        [v - np.log10(3) for v in lims],
                        [v + np.log10(3) for v in lims],
                        alpha=0.08, color="green")
        legend_els = [
            Patch(facecolor="#2196F3", label="<=2-fold"),
            Patch(facecolor="#4CAF50", label="<=3-fold"),
            Patch(facecolor="#FF9800", label="<=10-fold"),
            Patch(facecolor="#F44336", label=">10-fold"),
        ]
        ax.legend(handles=legend_els + [
            plt.scatter([], [], c="gold", s=40, edgecolors="k",
                        label=f"Training (n={len(train_df)})")
        ], fontsize=7)
        ax.set_xlabel("Gemessen: log10(Clint)", fontsize=10)
        ax.set_ylabel("RF vorhergesagt: log10(Clint)", fontsize=10)
        ax.set_title(
            f"RF Clint-Vorhersage (GCN nicht verfuegbar)\n"
            f"Test-Set (n={len(test_df)})  R^2={r2_te:.3f}  "
            f"GMFE={gmfe_te:.1f}x  <=3-fold: {pct3_te:.0f}%",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS / "gcn_loo_cv_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: results/gcn_loo_cv_scatter.png")
        print("\nDone.")
        return

    df_clean = train_df.copy()

    smiles_cache_path = DATA / "smiles_cache_777.csv"
    smiles_map: dict[str, str] = {}
    if smiles_cache_path.exists():
        sc = pd.read_csv(smiles_cache_path, dtype=str)
        smiles_map = {r["CAS"]: r["SMILES"] for _, r in sc.iterrows()
                      if pd.notna(r.get("SMILES")) and r["SMILES"] != "nan"}
        print(f"\nSMILES aus Cache: {len(smiles_map)}")

    print("\nSMILES-Abruf (Cache + PubChem-Fallback) ...")
    df_clean["SMILES"] = df_clean["CAS"].map(smiles_map)
    missing_smiles = df_clean[df_clean["SMILES"].isna()]
    for i, row in missing_smiles.iterrows():
        smi = get_smiles(str(row["CAS"]).strip(), str(row["Compound"]))
        df_clean.at[i, "SMILES"] = smi
        if smi:
            time.sleep(0.15)

    df_clean = df_clean.dropna(subset=["SMILES"]).copy()
    print(f"Trainingschemikalien mit SMILES: {len(df_clean)}")

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

    n          = len(df_valid)
    y_pred_log = np.full(n, np.nan)

    print(f"\nTrainiere GCN auf {n} Graphen ...")
    t0 = time.time()
    model_final = train_gcn(graphs, y_log)
    print(f"Training abgeschlossen in {time.time()-t0:.1f}s")

    test_df_clean = test_df.copy()
    test_df_clean["SMILES"] = test_df_clean["CAS"].map(smiles_map)
    test_graphs, test_valid_idx = [], []
    for i, row in test_df_clean.iterrows():
        smi = row.get("SMILES")
        if not smi or str(smi) == "nan":
            continue
        g = mol_to_graph(str(smi))
        if g is not None:
            test_graphs.append((g, i))
            test_valid_idx.append(i)

    test_rows = []
    for g, i in test_graphs:
        row    = test_df_clean.loc[i]
        X_te, A_te = g
        pred   = predict_gcn(model_final, X_te, A_te)
        y_true = np.log10(float(row["Clint"]) + EPSILON)
        fe     = 10 ** abs(y_true - pred)
        test_rows.append({
            "CAS": row["CAS"], "Compound": row["Compound"],
            "Clint": row["Clint"], "SMILES": row.get("SMILES"),
            "log10_true": round(y_true, 4),
            "log10_pred_GCN": round(pred, 4),
            "fold_error": round(fe, 3),
        })
        print(f"  {str(row['Compound'])[:28]:<28}  "
              f"wahr={y_true:.2f}  pred={pred:.2f}  FE={fe:.2f}x")

    train_rows = []
    for k in range(n):
        X_tr, A_tr = graphs[k]
        pred_tr    = predict_gcn(model_final, X_tr, A_tr)
        y_true_tr  = y_log[k]
        fe_tr      = 10 ** abs(y_true_tr - pred_tr)
        train_rows.append({
            "CAS": df_valid.iloc[k]["CAS"],
            "Compound": df_valid.iloc[k]["Compound"],
            "Clint": df_valid.iloc[k]["Clint"],
            "SMILES": df_valid.iloc[k].get("SMILES"),
            "log10_true": round(y_true_tr, 4),
            "log10_pred_GCN": round(pred_tr, 4),
            "fold_error": round(fe_tr, 3),
        })

    results_train_df = pd.DataFrame(train_rows)
    results_test_df  = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(columns=results_train_df.columns)

    results_train_df["split"] = "Training"
    results_test_df["split"]  = "Test"
    results_df_all = pd.concat([results_train_df, results_test_df], ignore_index=True)

    if len(results_test_df):
        y_log_test  = results_test_df["log10_true"].values
        y_pred_test = results_test_df["log10_pred_GCN"].values
        fe_all      = results_test_df["fold_error"].values
    else:
        y_log_test  = results_train_df["log10_true"].values
        y_pred_test = results_train_df["log10_pred_GCN"].values
        fe_all      = results_train_df["fold_error"].values

    n_eval = len(y_log_test)

    r2_log   = r2_score(y_log_test, y_pred_test)
    rmse_log = float(np.sqrt(mean_squared_error(y_log_test, y_pred_test)))
    rho, rho_p = spearmanr(y_log_test, y_pred_test)
    gmfe     = float(np.exp(np.mean(np.log(fe_all))))
    pct2     = float(np.mean(fe_all <= 2.0) * 100)
    pct3     = float(np.mean(fe_all <= 3.0) * 100)
    pct10    = float(np.mean(fe_all <= 10.) * 100)

    label_eval = f"Test-Set (n={n_eval})" if len(results_test_df) else f"Trainingsset (n={n_eval})"
    print(f"\n{'='*65}")
    print(f"GCN Ergebnis ({label_eval}):")
    print(f"  R^2  (log10)         : {r2_log:.4f}")
    print(f"  RMSE (log10)         : {rmse_log:.4f}")
    print(f"  Spearman rho         : {rho:.4f}  (p = {rho_p:.3e})")
    print(f"  GMFE                 : {gmfe:.2f}x")
    print(f"  Innerhalb 2-fold     : {pct2:.0f} %")
    print(f"  Innerhalb 3-fold     : {pct3:.0f} %")
    print(f"  Innerhalb 10-fold    : {pct10:.0f} %")

    metrics_text = (
        f"GCN Train/Test  (Training={n}, Test={n_eval})\n"
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

    results_df_all["Clint_pred_GCN"] = np.round(
        10 ** results_df_all["log10_pred_GCN"] - EPSILON, 4).clip(lower=0)
    results_df_all.to_csv(DATA / "pilot_chemicals_gcn.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def fe_color(fe_arr):
        return ["#2196F3" if f <= 2 else "#4CAF50" if f <= 3
                else "#FF9800" if f <= 10 else "#F44336" for f in fe_arr]

    ax = axes[0]
    if len(results_test_df):
        cols_te = fe_color(results_test_df["fold_error"].values)
        ax.scatter(results_test_df["log10_true"], results_test_df["log10_pred_GCN"],
                   c=cols_te, s=80, edgecolors="k", linewidths=0.5, zorder=4)
        for _, row in results_test_df.iterrows():
            ax.annotate(str(row["Compound"])[:10],
                        (row["log10_true"], row["log10_pred_GCN"]),
                        fontsize=6, alpha=0.7)
    ax.scatter(results_train_df["log10_true"], results_train_df["log10_pred_GCN"],
               c="gold", s=30, edgecolors="k", linewidths=0.3, alpha=0.5,
               zorder=2, label=f"Training (n={n})")

    all_y  = results_df_all["log10_true"].tolist() + results_df_all["log10_pred_GCN"].tolist()
    lims   = [min(all_y) - 0.5, max(all_y) + 0.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="ideal")
    ax.fill_between(lims, [v - np.log10(3) for v in lims],
                    [v + np.log10(3) for v in lims],
                    alpha=0.08, color="green", label="3-fold-Band")
    ax.set_xlabel("Gemessen: log10(Clint)", fontsize=10)
    ax.set_ylabel("GCN vorhergesagt: log10(Clint)", fontsize=10)
    ax.set_title(f"GCN Test-Set (n={n_eval})\n"
                 f"R^2={r2_log:.3f}  GMFE={gmfe:.1f}x  "
                 f"<=3-fold: {pct3:.0f}%", fontsize=10)
    legend_els = [Patch(facecolor="#2196F3", label="<=2-fold"),
                  Patch(facecolor="#4CAF50", label="<=3-fold"),
                  Patch(facecolor="#FF9800", label="<=10-fold"),
                  Patch(facecolor="#F44336", label=">10-fold")]
    ax.legend(handles=legend_els + [
        plt.scatter([], [], c="gold", s=40, edgecolors="k", label=f"Training (n={n})")
    ], fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    val_path = RESULTS / "clint_validation_external.csv"
    if val_path.exists() and len(results_test_df):
        val_df = pd.read_csv(val_path)
        merged = results_test_df.merge(
            val_df[["CAS", "log10_pred"]].rename(columns={"log10_pred": "log10_pred_RF"}),
            on="CAS", how="inner",
        )
        if len(merged):
            fe_rf_m  = 10 ** np.abs(merged["log10_true"] - merged["log10_pred_RF"])
            fe_gcn_m = 10 ** np.abs(merged["log10_true"] - merged["log10_pred_GCN"])
            ax.scatter(merged["log10_true"], merged["log10_pred_RF"],
                       label=f"RF/GB  GMFE={np.exp(np.mean(np.log(fe_rf_m))):.1f}x",
                       edgecolors="steelblue", facecolors="lightblue", s=60)
            ax.scatter(merged["log10_true"], merged["log10_pred_GCN"],
                       label=f"GCN    GMFE={np.exp(np.mean(np.log(fe_gcn_m))):.1f}x",
                       edgecolors="tomato", facecolors="lightsalmon", s=60, marker="^")
            ax.plot(lims, lims, "k--", lw=1.2)
            ax.set_xlabel("Gemessen: log10(Clint)", fontsize=10)
            ax.set_ylabel("Vorhergesagt: log10(Clint)", fontsize=10)
            ax.set_title(
                f"GCN vs. RF/GB - Test-Set (n={len(merged)})\n"
                f"Trainiert auf {n} httk-Chemikalien",
                fontsize=10,
            )
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "RF-Ergebnisse nicht verfuegbar",
                ha="center", va="center", transform=ax.transAxes)

    plt.suptitle(
        f"GCN Clint-Vorhersage  |  Training: {n} httk-Chemikalien  |  Test: {n_eval} Hold-out",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "gcn_loo_cv_scatter.png", dpi=150, bbox_inches="tight")
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
