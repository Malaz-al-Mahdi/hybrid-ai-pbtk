"""
utils.py
--------
Gemeinsame Hilfsfunktionen und Klassen fuer das Forschungsprojekt.

Inhalt
~~~~~~
  A. Pfadkonstanten
  B. Toxikokinetische (PK) Konstanten und Funktionen
  C. Feature Engineering fuer ML-Modelle (RF/GB)
  D. Metrik-Berechnung (RMSE, R², GMFE, Fold-Error, Spearman)
  E. SMILES-Abruf (PubChem + CIR)
  F. Molekuelgraph-Konvertierung (RDKit)
  G. GCN-Modell (GCNLayer, MolGCN, train_gcn, predict_gcn)
"""

import time
import json
import warnings
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# A. Pfadkonstanten
# ══════════════════════════════════════════════════════════════════════════════

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

# Standard-Datendateien
PILOT_CSV       = DATA / "pilot_chemicals_full.csv"
PILOT_IMPUTED   = DATA / "pilot_chemicals_imputed.csv"
PILOT_GCN_CSV   = DATA / "pilot_chemicals_gcn.csv"
ALL_777_CSV     = DATA / "all_777_chemicals.csv"
AED_BER_CSV     = RESULTS / "aed_ber_full.csv"
GCN_PRED_CSV    = RESULTS / "gcn_777_predictions.csv"
SMILES_CACHE    = DATA / "smiles_cache_777.csv"


# ══════════════════════════════════════════════════════════════════════════════
# B. Toxikokinetische Konstanten und Funktionen
# ══════════════════════════════════════════════════════════════════════════════

Q_H     = 1.5       # Hepatischer Blutfluss [L/h/kg]
F_LIVER = 26e-3     # Leberanteil am Koerpergewicht [kg/kg]
HEPATO  = 110e6     # Hepatozyten pro g Leber
EPSILON = 1e-3      # Numerische Stabilisierung (log-Schutz)


def clint_uL_to_cl_h(clint_uL: float, fup: float) -> float:
    """
    Well-Stirred-Modell: hepatische Clearance [L/h/kg].

    Parameters
    ----------
    clint_uL : Clint in µL/min/Mio Zellen
    fup      : Freie Fraktion im Plasma (0–1)

    Returns
    -------
    CL_hepatisch [L/h/kg]
    """
    fup      = max(float(fup), 1e-4)
    clint_uL = max(float(clint_uL), 0.0)
    clint_L  = clint_uL * 1e-6 * 60.0 * HEPATO * F_LIVER
    return Q_H * fup * clint_L / (Q_H + fup * clint_L)


def css_per_unit_dose(cl_h: float) -> float:
    """
    Steady-State-Konzentration pro Einheitsdosis [mg/L pro mg/kg/day].

    Css = Dose / (CL_h * 24)   [vollstaendige orale Absorption]
    """
    return 1.0 / (max(cl_h, 1e-9) * 24.0)


def calc_aed(ac50_uM: float, mw: float, clint_uL: float, fup: float) -> float:
    """
    Activity Equivalent Dose [mg/kg/day] via IVIVE (Well-Stirred-Modell).

    AED = AC50 [mg/L] / Css_per_unit_dose [mg/L / (mg/kg/day)]
        = AC50_uM * MW/1000 * CL_h * 24
    """
    if any(np.isnan([ac50_uM, mw, clint_uL, fup])):
        return np.nan
    ac50_mg = float(ac50_uM) * float(mw) / 1000.0
    cl_h    = clint_uL_to_cl_h(clint_uL, fup)
    cps     = css_per_unit_dose(cl_h)
    if cps <= 0:
        return np.nan
    return ac50_mg / cps


def calc_ber(aed: float, exposure: float) -> float:
    """Bioactivity Exposure Ratio = AED / Exposure."""
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in [aed, exposure]):
        return np.nan
    if exposure <= 0 or aed <= 0:
        return np.nan
    return float(aed) / float(exposure)


def concern_label(ber: float) -> str:
    """Klassifizierung nach BER-Wert."""
    if np.isnan(ber):   return "no_data"
    if ber < 1:         return "HIGH  (BER<1)"
    if ber < 10:        return "MEDIUM (BER 1-10)"
    if ber < 100:       return "LOW   (BER 10-100)"
    return "NEGLIGIBLE (BER>100)"


# ══════════════════════════════════════════════════════════════════════════════
# C. Feature Engineering fuer RF / GB
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "log10_MW", "logP", "logP^2", "log10_Fup", "sqrt_Fup",
    "MW*logP",  "MW*Fup", "logP*Fup", "MW",
]


def engineer_features(df_in: pd.DataFrame) -> np.ndarray:
    """
    9 Features aus MW, logP, Fup:
      log10(MW), logP, logP², log10(Fup), sqrt(Fup),
      MW*logP, MW*Fup, logP*Fup, MW

    Parameters
    ----------
    df_in : DataFrame mit Spalten MW, logP, Fup (Zeilen = Chemikalien)

    Returns
    -------
    np.ndarray mit Form (n, 9)
    """
    mw  = np.clip(
        pd.to_numeric(df_in["MW"],   errors="coerce").fillna(300).values,
        1.0, None,
    )
    lgp = pd.to_numeric(df_in["logP"], errors="coerce").fillna(2.0).values
    fup = np.clip(
        pd.to_numeric(df_in["Fup"],  errors="coerce").fillna(0.1).values,
        1e-6, 1.0,
    )
    return np.column_stack([
        np.log10(mw),
        lgp,
        lgp ** 2,
        np.log10(fup + 1e-6),
        np.sqrt(fup),
        mw * lgp,
        mw * fup,
        lgp * fup,
        mw,
    ])


# ══════════════════════════════════════════════════════════════════════════════
# D. Metriken
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict:
    """
    Berechnet Standardmetriken fuer log10-skalierte Vorhersagen.

    Returns
    -------
    dict mit Schluesseln: r2, rmse, spearman_rho, spearman_p,
                          gmfe, pct_2fold, pct_3fold, pct_10fold
    """
    fe   = 10 ** np.abs(y_true_log - y_pred_log)
    rho, p = spearmanr(y_true_log, y_pred_log)
    return {
        "r2":           float(r2_score(y_true_log, y_pred_log)),
        "rmse":         float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "spearman_rho": float(rho),
        "spearman_p":   float(p),
        "gmfe":         float(np.exp(np.mean(np.log(fe)))),
        "pct_2fold":    float(np.mean(fe <= 2.0)  * 100),
        "pct_3fold":    float(np.mean(fe <= 3.0)  * 100),
        "pct_10fold":   float(np.mean(fe <= 10.0) * 100),
    }


def print_metrics(m: dict, label: str = "", n: int | None = None) -> None:
    """Gibt Metriken formatiert aus."""
    n_str = f"  (n={n})" if n is not None else ""
    print(f"\n  [{label}]{n_str}")
    print(f"    R^2       : {m['r2']:.4f}")
    print(f"    RMSE log10: {m['rmse']:.4f}")
    print(f"    Spearman  : {m['spearman_rho']:.4f}  (p={m['spearman_p']:.3e})")
    print(f"    GMFE      : {m['gmfe']:.2f}x")
    print(f"    <=2-fold  : {m['pct_2fold']:.0f}%  "
          f"|  <=3-fold: {m['pct_3fold']:.0f}%  "
          f"|  <=10-fold: {m['pct_10fold']:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# E. SMILES-Abruf
# ══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, timeout: int = 5) -> bytes | None:
    """Einfacher HTTP-GET mit Timeout; gibt None bei Fehler zurueck."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def pubchem_smiles(cas: str) -> str | None:
    """SMILES fuer eine CAS-Nummer via PubChem REST API."""
    raw = _http_get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
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
    """SMILES via NCI Chemical Identifier Resolver (cactus.nci.nih.gov)."""
    try:
        from rdkit import Chem
    except ImportError:
        return None
    raw = _http_get(
        f"https://cactus.nci.nih.gov/chemical/structure/"
        f"{urllib.parse.quote(cas)}/smiles",
        timeout=6,
    )
    if raw:
        smi = raw.decode("utf-8", errors="ignore").strip().split()[0]
        if smi and Chem.MolFromSmiles(smi):
            return smi
    return None


def fetch_smiles(cas: str) -> str | None:
    """SMILES abrufen: PubChem zuerst, dann CIR als Fallback."""
    return pubchem_smiles(cas) or cir_smiles(cas)


def load_smiles_cache(all_cas: list[str], verbose: bool = True) -> dict[str, str]:
    """
    Laedt SMILES aus lokalem Cache; fragt fuer fehlende CAS bei APIs an.

    Liest/schreibt data/smiles_cache_777.csv.

    Returns
    -------
    {cas: smiles}
    """
    # Cache einlesen
    cache: dict[str, str] = {}
    if SMILES_CACHE.exists():
        df_c = pd.read_csv(SMILES_CACHE, dtype=str)
        cache = {
            k: v for k, v in zip(df_c["CAS"], df_c["SMILES"])
            if pd.notna(v) and v != "nan"
        }

    # Pilot-SMILES (aus pilot_chemicals_gcn.csv) einspielen
    if PILOT_GCN_CSV.exists():
        pg = pd.read_csv(PILOT_GCN_CSV, dtype=str)
        for _, row in pg.iterrows():
            if pd.notna(row.get("SMILES")) and str(row["SMILES"]) != "nan":
                cache[str(row["CAS"])] = str(row["SMILES"])

    missing = [c for c in all_cas if c not in cache]
    if verbose:
        print(f"Im Cache: {len(cache)}  |  Fehlend: {len(missing)}")

    if missing:
        # Netzwerk-Test
        test = pubchem_smiles("80-05-7")
        if not test:
            test = cir_smiles("80-05-7")
        if not test:
            if verbose:
                print("Kein Netzwerkzugriff – verwende vorhandenen Cache")
            return cache

        if verbose:
            print(f"Starte Abruf fuer {len(missing)} CAS-Nummern ...")

        for i, cas in enumerate(missing):
            smi = fetch_smiles(cas)
            if smi:
                cache[cas] = smi
            time.sleep(0.15)
            if verbose and (i + 1) % 50 == 0:
                found = sum(1 for c in missing[: i + 1] if c in cache)
                print(f"  {i+1:>4}/{len(missing)}  gefunden={found}  "
                      f"({(i+1)/len(missing)*100:.0f}%)", flush=True)
            if (i + 1) % 50 == 0:
                _save_smiles_cache(cache)

        _save_smiles_cache(cache)
        if verbose:
            found_total = sum(1 for c in missing if c in cache)
            print(f"Gesamt abgerufen: {found_total}/{len(missing)}")

    if verbose:
        print(f"SMILES verfuegbar: {len(cache)} / {len(all_cas)}")
    return cache


def _save_smiles_cache(cache: dict[str, str]) -> None:
    pd.DataFrame(list(cache.items()), columns=["CAS", "SMILES"]).to_csv(
        SMILES_CACHE, index=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# F. Molekuelgraph-Konvertierung (RDKit)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    from rdkit import Chem
    from rdkit.Chem import rdchem

    _ATOM_SYMBOLS   = ["C", "N", "O", "S", "P", "Cl", "F", "Br", "I", "Si", "OTHER"]
    _HYBRID_TYPES   = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]

    def _one_hot(v, choices: list) -> list:
        enc = [0.0] * (len(choices) + 1)
        enc[choices.index(v) if v in choices else len(choices)] = 1.0
        return enc

    def _atom_features(atom) -> list:
        f  = _one_hot(atom.GetSymbol(),        _ATOM_SYMBOLS)
        f += _one_hot(atom.GetDegree(),         [0, 1, 2, 3, 4, 5, 6])
        f += _one_hot(atom.GetTotalNumHs(),     [0, 1, 2, 3, 4])
        f += _one_hot(atom.GetFormalCharge(),   [-2, -1, 0, 1, 2])
        f += _one_hot(atom.GetHybridization(),  _HYBRID_TYPES)
        f.append(float(atom.GetIsAromatic()))
        f.append(float(atom.IsInRing()))
        return f

    N_ATOM_FEAT = len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))

    def mol_to_graph(smiles: str):
        """
        Konvertiert SMILES in Molekuelgraph (X, A) fuer GCN.

        Returns
        -------
        (X, A) – Feature-Matrix [n_atoms, N_ATOM_FEAT] und
                  normalisierte Adjazenzmatrix [n_atoms, n_atoms]
        oder None, wenn SMILES ungueltig.
        """
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        n   = mol.GetNumAtoms()
        X   = torch.tensor(
            [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
        )
        adj = np.eye(n, dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1.0
        deg     = adj.sum(axis=1)
        d_inv   = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        A       = torch.tensor(d_inv @ adj @ d_inv, dtype=torch.float32)
        return X, A

    _GCN_AVAILABLE = True

except ImportError:
    _GCN_AVAILABLE = False
    N_ATOM_FEAT    = 40   # Standardwert

    def mol_to_graph(smiles: str):  # noqa: F811
        raise ImportError("rdkit oder torch fehlt.")


# ══════════════════════════════════════════════════════════════════════════════
# G. GCN-Modell
# ══════════════════════════════════════════════════════════════════════════════

if _GCN_AVAILABLE:
    # Hyperparameter
    GCN_H1, GCN_H2, GCN_H3 = 128, 64, 32
    GCN_DROPOUT  = 0.30
    GCN_EPOCHS   = 500
    GCN_LR       = 5e-4
    GCN_WD       = 1e-4
    GCN_PATIENCE = 80

    class GCNLayer(nn.Module):
        """Einzelne Graph-Convolutional-Schicht: H' = A * H * W."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, A, H):
            return self.linear(A @ H)

    class MolGCN(nn.Module):
        """
        3-schichtiger Graph Convolutional Network fuer Molekuel-Clint-Vorhersage.

        Architektur: N_ATOM_FEAT -> H1 -> H2 -> H3 -> [mean pooling] -> MLP -> 1
        """

        def __init__(
            self,
            n_feat:   int   = N_ATOM_FEAT,
            h1:       int   = GCN_H1,
            h2:       int   = GCN_H2,
            h3:       int   = GCN_H3,
            dropout:  float = GCN_DROPOUT,
        ):
            super().__init__()
            self.gcn1  = GCNLayer(n_feat, h1)
            self.gcn2  = GCNLayer(h1,     h2)
            self.gcn3  = GCNLayer(h2,     h3)
            self.drop  = nn.Dropout(dropout)
            self.mlp   = nn.Sequential(
                nn.Linear(h3, 16), nn.ReLU(), nn.Linear(16, 1)
            )
            self.act   = nn.ReLU()

        def forward(self, A, X):
            h = self.drop(self.act(self.gcn1(A, X)))
            h = self.drop(self.act(self.gcn2(A, h)))
            h = self.act(self.gcn3(A, h))
            return self.mlp(h.mean(dim=0)).squeeze()

    def train_gcn(
        graphs:  list,
        y_log:   np.ndarray,
        epochs:  int   = GCN_EPOCHS,
        patience: int  = GCN_PATIENCE,
        lr:      float = GCN_LR,
        wd:      float = GCN_WD,
        seed:    int   = 42,
    ) -> "MolGCN":
        """
        Trainiert MolGCN auf einem Satz von (X, A)-Graphen und log10(Clint)-Werten.

        Returns
        -------
        Trainiertes MolGCN-Modell mit angeheftetem _scaler_y.
        """
        import torch.optim as optim

        torch.manual_seed(seed)
        np.random.seed(seed)

        scaler = StandardScaler()
        y_sc   = scaler.fit_transform(y_log.reshape(-1, 1)).ravel()

        model  = MolGCN()
        opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        crit   = nn.MSELoss()
        best_loss, best_state, wait = float("inf"), {}, 0

        model.train()
        for ep in range(epochs):
            ep_loss = 0.0
            for i in np.random.permutation(len(graphs)):
                X, A = graphs[i]
                opt.zero_grad()
                loss = crit(
                    model(A, X),
                    torch.tensor(y_sc[i], dtype=torch.float32),
                )
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item()
            avg = ep_loss / max(len(graphs), 1)
            if avg < best_loss - 1e-7:
                best_loss  = avg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait       = 0
            else:
                wait += 1
            if wait >= patience:
                break

        model.load_state_dict(best_state)
        model._scaler_y = scaler
        return model

    @torch.no_grad()
    def predict_gcn(model: "MolGCN", X, A) -> float:
        """Einzelvorhersage: gibt log10(Clint) zurueck."""
        model.eval()
        raw = float(model(A, X).item())
        return float(model._scaler_y.inverse_transform([[raw]])[0, 0])
