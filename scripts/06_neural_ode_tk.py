"""
06_neural_ode_tk.py
-------------------
Neural Ordinary Differential Equations for continuous-time toxicokinetic
(TK) modeling.

Scientific rationale
~~~~~~~~~~~~~~~~~~~~
Classical PK models solve analytically or numerically for observed time
points.  When data are sparse (e.g. only 3–5 plasma measurements per
subject) the classical model must assume a compartment structure.  Neural
ODEs instead *learn* the vector field dC/dt = f_theta(C, t; z_chem) from
data, where z_chem encodes the chemical's physicochemical properties.  This
allows:
  - Interpolation at unobserved times  (handling sparse / irregular data)
  - Learning non-standard compartment dynamics without hand-crafting structure
  - Full gradient flow through the integrator for end-to-end training

Implementation details
~~~~~~~~~~~~~~~~~~~~~~
  - Training data: plasma C(t) generated from the analytical 1-compartment
    oral model using PBTK parameters from Step 1/2 (Clint, Fup, logP).
  - Architecture: ChemEncoder (4-feature MLP → latent embedding z) +
    ODEFunc (MLP vector field using [C, z] as input) + differentiable RK4
    integrator implemented in PyTorch.
  - Evaluation: Leave-One-Out CV on 20 pilot chemicals.
  - Sparse-data demo: train on 5 irregularly spaced observations, predict
    the full 0–48 h profile.

Outputs
~~~~~~~
  results/neural_ode_curves.png        Full C(t): true vs. predicted
  results/neural_ode_sparse_demo.png   Sparse-data interpolation demo
  results/neural_ode_metrics.csv       Per-chemical MAE, RMSE, R²
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    sys.exit(
        "ERROR: PyTorch is required.  Install with:\n"
        "  pip install torch\n"
        "or:  py -m pip install torch"
    )

IMPUTED_CSV = DATA / "pilot_chemicals_imputed.csv"
if not IMPUTED_CSV.exists():
    sys.exit(
        f"ERROR: {IMPUTED_CSV} not found.  Run steps 01 and 02 first."
    )

torch.manual_seed(42)
np.random.seed(42)

# ── PK constants ─────────────────────────────────────────────────────────────
Q_H       = 1.5       # hepatic blood flow  [L / h / kg BW]
F_LIVER   = 26e-3     # liver weight fraction  [kg liver / kg BW]
HEPATO    = 110e6     # hepatocellularity  [cells / g liver]
KA        = 1.2       # oral absorption rate constant  [h⁻¹]
DOSE      = 1.0       # oral dose  [mg / kg BW]
DOSE_UG   = DOSE * 1e3  # µg / kg BW
T_MAX     = 48.0      # simulation horizon  [h]
N_EVAL    = 100       # evaluation time points

# ── Neural ODE hyperparameters ────────────────────────────────────────────────
EMBED_DIM  = 8
HIDDEN_DIM = 32
EPOCHS     = 600
LR         = 2e-3
PATIENCE   = 80        # early-stopping patience

# ── Helper: PK parameter conversions ─────────────────────────────────────────

def clint_to_cl(clint_uL_min_Mcells: float, fup: float) -> float:
    """
    Convert in-vitro Clint to whole-body hepatic blood clearance [L/h/kg].

    Well-stirred liver model:
        CL_h = Q_H * fu * Clint_liver / (Q_H + fu * Clint_liver)
    """
    clint_L_h_kg = clint_uL_min_Mcells * 1e-6 * 60.0 * HEPATO * F_LIVER
    fup_safe = max(float(fup), 1e-4)
    cl_h = Q_H * fup_safe * clint_L_h_kg / (Q_H + fup_safe * clint_L_h_kg)
    return max(cl_h, 1e-6)


def estimate_vd(logP: float) -> float:
    """
    Crude QSPR estimate of volume of distribution [L/kg].
    Rule of thumb: log(Vd) ≈ 0.4 * logP (Lombardo et al. regression).
    """
    vd = 0.5 * 10 ** (0.4 * np.clip(float(logP), -2.0, 6.0))
    return float(np.clip(vd, 0.4, 200.0))


def generate_trajectory(
    clint: float,
    fup: float,
    logP: float,
    t_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 2-state 1-compartment oral PK ODE and return C_plasma(t).

    States: y = [A_gut (µg/kg), C_plasma (µg/L = ng/mL)]

        dA_gut/dt = -ka * A_gut
        dC/dt     = ka * A_gut / Vd  -  ke * C

    Returns (t_eval, C_plasma).
    """
    if t_eval is None:
        t_eval = np.linspace(0.0, T_MAX, N_EVAL)

    cl_h = clint_to_cl(clint, fup)
    vd   = estimate_vd(logP)
    ke   = cl_h / vd                               # h⁻¹
    f_oral = float(np.clip(1.0 - cl_h / Q_H, 0.05, 1.0))

    y0 = [f_oral * DOSE_UG, 0.0]

    def ode_rhs(t, y):
        a, c = y
        return [-KA * a, KA * a / vd - ke * c]

    sol = solve_ivp(
        ode_rhs, [0.0, T_MAX], y0,
        t_eval=t_eval, method="RK45",
        rtol=1e-7, atol=1e-9,
    )
    return sol.t, sol.y[1]


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame):
    """
    Generate (features, t_eval, C_true) tuples for all pilot chemicals.
    Returns:
        features_arr  ndarray  (n_chems, 4)   [MW, logP, Fup, Clint]
        trajs         list of ndarray (N_EVAL,)
        t_eval        ndarray (N_EVAL,)
    """
    features_list = []
    trajs         = []
    t_eval        = np.linspace(0.0, T_MAX, N_EVAL)

    for _, row in df.iterrows():
        clint  = float(row["Clint_final"]) if "Clint_final" in row else float(row.get("Clint", 1.0))
        fup    = float(row["Fup"])
        logP   = float(row["logP"])
        mw     = float(row["MW"])
        clint  = max(clint, 0.01)

        _, c_t = generate_trajectory(clint, fup, logP, t_eval)
        features_list.append([mw, logP, fup, clint])
        trajs.append(c_t)

    return np.array(features_list, dtype=np.float32), trajs, t_eval


# ── Model ─────────────────────────────────────────────────────────────────────

class ChemEncoder(nn.Module):
    """Maps 4 physicochemical features to a latent chemical embedding."""
    def __init__(self, n_feat=4, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 16),
            nn.Tanh(),
            nn.Linear(16, embed_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    """
    Neural ODE vector field:
        dC/dt = f_theta(C, z_chem)

    Inputs:  [C (scalar), z_chem (embed_dim)] → dC/dt (scalar)
    """
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, C: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([C.unsqueeze(-1), z], dim=-1)
        return self.net(inp).squeeze(-1)


class NeuralODETK(nn.Module):
    """
    Full Neural ODE model:
      1. Encode chemical features → z
      2. Integrate dC/dt = ODEFunc(C, z) using RK4
    """
    def __init__(self, n_feat=4, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder  = ChemEncoder(n_feat, embed_dim)
        self.ode_func = ODEFunc(embed_dim, hidden_dim)

    def _rk4_step(
        self,
        C: torch.Tensor,
        z: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        k1 = self.ode_func(C, z)
        k2 = self.ode_func(C + 0.5 * dt * k1, z)
        k3 = self.ode_func(C + 0.5 * dt * k2, z)
        k4 = self.ode_func(C + dt * k3, z)
        return C + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(
        self,
        features: torch.Tensor,     # (n_feat,)
        C0: torch.Tensor,            # scalar
        t_eval: torch.Tensor,        # (T,)
    ) -> torch.Tensor:
        """Return predicted C(t) at times t_eval, shape (T,)."""
        z  = self.encoder(features.unsqueeze(0))   # (1, embed_dim)
        C  = C0.unsqueeze(0)                        # (1,)
        Cs = [C]
        for i in range(len(t_eval) - 1):
            dt = float(t_eval[i + 1] - t_eval[i])
            C  = self._rk4_step(C, z, dt)
            Cs.append(C)
        return torch.cat(Cs, dim=0).squeeze(1)      # (T,)


# ── Training utilities ────────────────────────────────────────────────────────

def train_one_chemical(
    model: NeuralODETK,
    feat_t: torch.Tensor,
    C_true: torch.Tensor,
    t_tensor: torch.Tensor,
    C0: torch.Tensor,
    optimizer: optim.Optimizer,
    n_epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> list[float]:
    """Fine-tune / train the model on a single chemical's C(t) curve."""
    criterion = nn.MSELoss()
    losses    = []
    best_loss = float("inf")
    wait      = 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        C_pred = model(feat_t, C0, t_tensor)
        loss   = criterion(C_pred, C_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(loss.item())

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    return losses


# ── Leave-One-Out evaluation ──────────────────────────────────────────────────

def loo_evaluation(
    features_arr: np.ndarray,
    trajs: list,
    t_eval: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Leave-one-out CV: for each chemical, train on the other 19, then predict.
    Returns a DataFrame of metrics.
    """
    scaler  = StandardScaler()
    feat_sc = scaler.fit_transform(features_arr).astype(np.float32)

    metrics_rows = []
    n = len(df)

    print(f"\nLOO-CV over {n} chemicals …")
    for test_idx in range(n):
        name = df.iloc[test_idx]["Compound"]
        train_idx = [j for j in range(n) if j != test_idx]

        model = NeuralODETK()
        opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        # ─ Train on all others ───────────────────────────────────────────────
        for j in train_idx:
            feat_t  = torch.tensor(feat_sc[j])
            C_true  = torch.tensor(trajs[j], dtype=torch.float32)
            t_tens  = torch.tensor(t_eval, dtype=torch.float32)
            C0      = C_true[0].clone()
            train_one_chemical(
                model, feat_t, C_true, t_tens, C0, opt,
                n_epochs=300, patience=50
            )

        # ─ Predict held-out chemical (no further training) ───────────────────
        model.eval()
        with torch.no_grad():
            feat_t  = torch.tensor(feat_sc[test_idx])
            t_tens  = torch.tensor(t_eval, dtype=torch.float32)
            C_true  = trajs[test_idx]
            C0      = torch.tensor(C_true[0], dtype=torch.float32)
            C_pred  = model(feat_t, C0, t_tens).numpy()

        mae  = float(np.mean(np.abs(C_pred - C_true)))
        rmse = float(np.sqrt(np.mean((C_pred - C_true) ** 2)))
        r2   = float(r2_score(C_true, C_pred))

        metrics_rows.append({
            "Chemical": name,
            "MAE_ngmL":  round(mae,  4),
            "RMSE_ngmL": round(rmse, 4),
            "R2":        round(r2,   4),
        })
        print(f"  [{test_idx+1:2d}/{n}] {name:<24}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    return pd.DataFrame(metrics_rows)


# ── Plotting utilities ────────────────────────────────────────────────────────

def plot_curves(
    features_arr: np.ndarray,
    trajs: list,
    t_eval: np.ndarray,
    df: pd.DataFrame,
    scaler: StandardScaler,
    out_path: Path,
    n_show: int = 8,
) -> None:
    """Plot true vs. predicted C(t) for a selection of chemicals."""
    model = NeuralODETK()
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    feat_sc = scaler.transform(features_arr).astype(np.float32)

    # Train on all chemicals for the plot
    for j in range(len(df)):
        feat_t = torch.tensor(feat_sc[j])
        C_true = torch.tensor(trajs[j], dtype=torch.float32)
        t_tens = torch.tensor(t_eval,   dtype=torch.float32)
        train_one_chemical(model, feat_t, C_true, t_tens, C_true[0], opt,
                           n_epochs=400, patience=60)

    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    model.eval()
    for k, ax in enumerate(axes[:n_show]):
        j       = k % len(df)
        name    = df.iloc[j]["Compound"]
        C_true  = trajs[j]
        feat_t  = torch.tensor(feat_sc[j])
        t_tens  = torch.tensor(t_eval, dtype=torch.float32)
        C0      = torch.tensor(C_true[0], dtype=torch.float32)
        with torch.no_grad():
            C_pred = model(feat_t, C0, t_tens).numpy()

        ax.plot(t_eval, C_true, "k-",  label="Analytical ODE",   lw=2)
        ax.plot(t_eval, C_pred, "r--", label="Neural ODE",       lw=1.5)
        ax.set_title(name[:22], fontsize=8)
        ax.set_xlabel("Time (h)", fontsize=7)
        ax.set_ylabel("C (ng/mL)", fontsize=7)
        if k == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_show:]:
        ax.set_visible(False)

    fig.suptitle("Neural ODE vs. Analytical 1-Compartment Model\nPilot Chemicals (trained, not LOO)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return model, feat_sc


def plot_sparse_demo(
    model: NeuralODETK,
    feat_sc: np.ndarray,
    trajs: list,
    t_eval: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    n_sparse: int = 5,
    demo_idx: int = 0,
) -> None:
    """
    Demonstrate sparse-data handling:
    Fine-tune on n_sparse random observations, predict full curve.
    """
    rng       = np.random.default_rng(42)
    sparse_t_idx = sorted(rng.choice(len(t_eval), size=n_sparse, replace=False))
    sparse_t     = t_eval[sparse_t_idx]
    sparse_C     = trajs[demo_idx][sparse_t_idx]
    name         = df.iloc[demo_idx]["Compound"]

    # Fine-tune on sparse observations only
    sparse_model = NeuralODETK()
    # Copy weights from globally trained model as warm start
    sparse_model.load_state_dict(model.state_dict())
    opt = optim.Adam(sparse_model.parameters(), lr=LR * 0.3, weight_decay=1e-4)

    feat_t      = torch.tensor(feat_sc[demo_idx])
    t_sparse_t  = torch.tensor(sparse_t, dtype=torch.float32)
    C_sparse_t  = torch.tensor(sparse_C, dtype=torch.float32)
    C0_t        = C_sparse_t[0]

    crit = nn.MSELoss()
    for _ in range(400):
        opt.zero_grad()
        C_pred_sparse = sparse_model(feat_t, C0_t, t_sparse_t)
        loss = crit(C_pred_sparse, C_sparse_t)
        loss.backward()
        opt.step()

    sparse_model.eval()
    t_full = torch.tensor(t_eval, dtype=torch.float32)
    with torch.no_grad():
        C_full_pred = sparse_model(feat_t, C0_t, t_full).numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_eval, trajs[demo_idx], "k-",  lw=2, label="True C(t)")
    ax.plot(t_eval, C_full_pred,     "b-",  lw=2, label="Neural ODE prediction (full)")
    ax.scatter(sparse_t, sparse_C, s=120, zorder=5, color="red",
               label=f"Observed ({n_sparse} sparse points)", edgecolors="k", linewidths=0.8)
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Plasma concentration (ng/mL)", fontsize=12)
    ax.set_title(
        f"Neural ODE – Sparse-data interpolation\n"
        f"{name}: trained on {n_sparse} observations, predicts full C(t)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Step 6 – Neural ODE for Toxicokinetic Modeling")
    print("=" * 65, "\n")

    df = pd.read_csv(IMPUTED_CSV)
    print(f"Loaded {len(df)} pilot chemicals.\n")

    # ── 1. Generate PK training trajectories ─────────────────────────────────
    print("Generating 1-compartment PK trajectories …")
    features_arr, trajs, t_eval = build_dataset(df)
    print(f"  {len(trajs)} chemicals  |  {len(t_eval)} time points each")
    print(f"  Cmax range: {min(c.max() for c in trajs):.1f} – "
          f"{max(c.max() for c in trajs):.1f} ng/mL\n")

    # ── 2. Leave-One-Out cross-validation ────────────────────────────────────
    metrics = loo_evaluation(features_arr, trajs, t_eval, df)
    metrics_path = RESULTS / "neural_ode_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    mean_r2   = metrics["R2"].mean()
    mean_rmse = metrics["RMSE_ngmL"].mean()
    print(f"\nLOO-CV summary:  mean R² = {mean_r2:.3f}  |  mean RMSE = {mean_rmse:.3f} ng/mL")

    # ── 3. Train on all chemicals and plot curves ─────────────────────────────
    print("\nTraining full model for visualisation …")
    scaler = StandardScaler()
    scaler.fit(features_arr)

    trained_model, feat_sc = plot_curves(
        features_arr, trajs, t_eval, df, scaler,
        out_path=RESULTS / "neural_ode_curves.png",
        n_show=min(8, len(df)),
    )

    # ── 4. Sparse-data demonstration ─────────────────────────────────────────
    print("\nGenerating sparse-data demonstration …")
    plot_sparse_demo(
        trained_model, feat_sc, trajs, t_eval, df,
        out_path=RESULTS / "neural_ode_sparse_demo.png",
        n_sparse=5,
        demo_idx=0,    # Bisphenol A is first chemical
    )

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Neural ODE summary")
    print("=" * 65)
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to results/")
    print("  neural_ode_curves.png      – fitted C(t) profiles")
    print("  neural_ode_sparse_demo.png – sparse interpolation demo")
    print("  neural_ode_metrics.csv     – per-chemical LOO-CV metrics")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
