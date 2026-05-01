"""
08_bayesian_ber.py
------------------
Probabilistic Uncertainty Analysis for BER via Bayesian Neural Network
(Monte-Carlo Dropout approximation).

Scientific rationale
~~~~~~~~~~~~~~~~~~~~
Classical (point-estimate) BER = AED / Exposure gives a single number.
This conceals uncertainty from three sources:

  1. Clint prediction uncertainty   (not all chemicals are well-characterised)
  2. AED calculation uncertainty    (httk Monte Carlo already covers IVIVE variability)
  3. Exposure estimate uncertainty  (NHANES / SEEM3 provide 90% CI)

A Bayesian Neural Network (BNN) with Monte Carlo Dropout approximates the
posterior predictive distribution p(Clint | features, D_train), providing
genuine credible intervals rather than mere confidence intervals.

At inference time we enable dropout and perform N stochastic forward passes.
Each pass samples a different sub-network → different Clint → different AED
→ different BER.  Aggregating 2000 such samples gives full posterior BER
distributions per chemical.

Architecture
~~~~~~~~~~~~
  - Input: 3 physicochemical features (MW, logP, Fup)
  - 3 hidden layers with Dropout(p=0.3) and ReLU activations
  - Output: log10(Clint)
  - Training: MSE loss + L2 regularisation
  - Inference: 2000 stochastic passes with dropout enabled

Outputs
~~~~~~~
  results/bayesian_ber.csv                Full posterior BER summary per chemical
  results/ber_credible_intervals.png      Waterfall with 90 % credible bands
  results/ber_posterior_top5.png          Posterior distributions for top-5 chemicals
  results/clint_posterior_uncertainty.png Clint prediction uncertainty
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

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

PILOT_CSV   = DATA    / "pilot_chemicals_imputed.csv"
AED_BER_CSV = RESULTS / "aed_ber_full.csv"

for p in (PILOT_CSV,):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found.  Run steps 01 and 02 first.")

torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DROPOUT_P   = 0.30    # dropout rate at every hidden layer
HIDDEN_DIMS = [64, 64, 32]
EPOCHS      = 800
LR          = 3e-3
WEIGHT_DECAY = 1e-4
N_MC        = 2000    # Monte Carlo forward passes for posterior sampling

# ── PK constants for AED scaling ─────────────────────────────────────────────
Q_H     = 1.5        # hepatic blood flow [L/h/kg]
F_LIVER = 26e-3
HEPATO  = 110e6

def clint_to_cl(clint_uL, fup):
    fup = max(float(fup), 1e-4)
    clint_L = float(clint_uL) * 1e-6 * 60.0 * HEPATO * F_LIVER
    return Q_H * fup * clint_L / (Q_H + fup * clint_L)


# ── Model ─────────────────────────────────────────────────────────────────────

class BayesianMLP(nn.Module):
    """
    Feed-forward neural network with dropout at every hidden layer.

    At training time:  standard dropout (regularisation).
    At inference time: keep dropout enabled (model.train()) to approximate
                       Bayesian posterior.
    """
    def __init__(
        self,
        n_in:       int   = 3,
        hidden:     list  = None,
        p_dropout:  float = DROPOUT_P,
    ):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN_DIMS
        layers = []
        prev = n_in
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training ─────────────────────────────────────────────────────────────────

def train_bayesian_mlp(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """
    Train BayesianMLP on (X, y) and return (model, scaler_X, scaler_y).
    """
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    X_t = torch.tensor(scaler_X.transform(X), dtype=torch.float32)
    y_t = torch.tensor(scaler_y.transform(y.reshape(-1, 1)).ravel(),
                       dtype=torch.float32)

    model = BayesianMLP(n_in=X.shape[1])
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.MSELoss()

    model.train()
    best_loss = float("inf")
    best_state = None
    patience, wait = 100, 0

    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if loss.item() < best_loss - 1e-7:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, scaler_X, scaler_y


# ── MC Dropout posterior sampling ─────────────────────────────────────────────

def mc_predict(
    model:    BayesianMLP,
    X:        np.ndarray,
    scaler_X,
    scaler_y,
    n_mc:     int = N_MC,
) -> np.ndarray:
    """
    Perform n_mc stochastic forward passes with dropout enabled.

    Returns an array of shape (n_mc, n_chemicals) with log10(Clint) samples.
    """
    X_t   = torch.tensor(scaler_X.transform(X), dtype=torch.float32)
    model.train()   # enable dropout for MC inference
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            y_std = model(X_t).numpy()
            y_orig = scaler_y.inverse_transform(y_std.reshape(-1, 1)).ravel()
            preds.append(y_orig)
    return np.stack(preds, axis=0)   # (n_mc, n_chem)


# ── BER propagation ──────────────────────────────────────────────────────────

def propagate_to_ber(
    log10_clint_samples: np.ndarray,     # (n_mc, n_chem)
    pilot:               pd.DataFrame,
    aed_ber:             pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Propagate Clint uncertainty through AED → BER.

    AED ∝ AC50 / (CL/Vd)  — simplified scaling:
        AED_scaled = AED_reference * (Clint_ref / Clint_sampled)

    When a reference AED from Step 5/6 is available we use it directly.
    Otherwise we estimate from first principles.

    Returns a DataFrame with per-chemical BER posterior summaries.
    """
    n_mc, n_chem = log10_clint_samples.shape
    results = []

    for i, (_, row) in enumerate(pilot.iterrows()):
        cas  = str(row.get("CAS",  ""))
        name = str(row.get("Compound", f"chem_{i}"))
        fup  = float(row.get("Fup",  0.1))

        # Reference Clint (best available from Step 2)
        clint_ref = float(row.get("Clint_final",
                     row.get("Clint_RF",
                     row.get("Clint", 1.0))))
        clint_ref = max(clint_ref, 0.01)

        # Reference AED from Step 5/6
        aed_ref, ac50_ref, exp_ref = None, None, None
        if aed_ber is not None:
            match = aed_ber[aed_ber["CAS"] == cas] if "CAS" in aed_ber.columns else pd.DataFrame()
            if len(match):
                row_ber = match.iloc[0]
                aed_ref = float(row_ber.get("AED_median", np.nan))
                ac50_ref = float(row_ber.get("AC50_5pct_uM", 1.0))
                exp_ref  = float(row_ber.get("Exposure_median_mg_kg_day", np.nan))

        # Posterior Clint samples (n_mc,)
        clint_samples = 10.0 ** log10_clint_samples[:, i]
        clint_samples = np.clip(clint_samples, 0.001, 1e5)

        if aed_ref is not None and np.isfinite(aed_ref) and aed_ref > 0:
            # Scale reference AED by clearance ratio (linear IVIVE approximation)
            cl_ref     = clint_to_cl(clint_ref,      fup)
            cl_samples = np.array([clint_to_cl(c, fup) for c in clint_samples])
            cl_samples = np.clip(cl_samples, 1e-6, Q_H * 0.999)
            aed_samples = aed_ref * (cl_ref / cl_samples)
        else:
            # First-principles AED: AED ≈ AC50 * BW_fraction (rough)
            ac50 = ac50_ref if (ac50_ref and np.isfinite(ac50_ref)) else 1.0
            cl_samples = np.array([clint_to_cl(c, fup) for c in clint_samples])
            # AED (mg/kg/day) ≈ AC50 (µM) * MW / 1e3 * ke / absorption_factor
            mw = float(row.get("MW", 300))
            aed_samples = ac50 * mw / 1e3 * cl_samples / Q_H  # rough scaling

        if exp_ref is not None and np.isfinite(exp_ref) and exp_ref > 0:
            ber_samples = aed_samples / exp_ref
        else:
            # No exposure data – skip BER but keep Clint distribution
            ber_samples = np.full(n_mc, np.nan)

        row_out = {
            "CAS":           cas,
            "Chemical":      name,
            "Clint_ref_uL":  round(clint_ref, 3),
            "Clint_p2_5":    round(float(np.percentile(clint_samples, 2.5)), 3),
            "Clint_median":  round(float(np.median(clint_samples)), 3),
            "Clint_p97_5":   round(float(np.percentile(clint_samples, 97.5)), 3),
            "AED_p2_5":      round(float(np.nanpercentile(aed_samples, 2.5)), 6),
            "AED_median":    round(float(np.nanmedian(aed_samples)), 6),
            "AED_p97_5":     round(float(np.nanpercentile(aed_samples, 97.5)), 6),
            "BER_p2_5":      round(float(np.nanpercentile(ber_samples, 2.5)), 3)  if np.any(np.isfinite(ber_samples)) else np.nan,
            "BER_median":    round(float(np.nanmedian(ber_samples)), 3)            if np.any(np.isfinite(ber_samples)) else np.nan,
            "BER_p97_5":     round(float(np.nanpercentile(ber_samples, 97.5)), 3) if np.any(np.isfinite(ber_samples)) else np.nan,
            "Exposure_mg_kg_day": round(exp_ref, 6) if (exp_ref and np.isfinite(exp_ref)) else np.nan,
            "_ber_samples":  ber_samples,   # kept for plotting; dropped before export
        }
        results.append(row_out)

    return pd.DataFrame(results)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_credible_intervals(result_df: pd.DataFrame, out_path: Path) -> None:
    """
    Waterfall plot of BER with 90 % posterior credible bands.
    """
    df = result_df.dropna(subset=["BER_median"]).copy()
    df = df.sort_values("BER_median").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x  = np.arange(len(df))
    ax.scatter(x, np.log10(df["BER_median"].clip(1e-6)),
               color="steelblue", s=60, zorder=4, label="BER median")
    ax.errorbar(
        x,
        np.log10(df["BER_median"].clip(1e-6)),
        yerr=[
            np.log10(df["BER_median"].clip(1e-6)) - np.log10(df["BER_p2_5"].clip(1e-6)),
            np.log10(df["BER_p97_5"].clip(1e-6))  - np.log10(df["BER_median"].clip(1e-6)),
        ],
        fmt="none", ecolor="steelblue", elinewidth=1.5, capsize=3,
        alpha=0.6, label="90 % credible interval",
    )
    ax.axhline(0,  color="orange", lw=1.5, ls="--", label="BER = 1  (AED = Exposure)")
    ax.axhline(1,  color="red",    lw=1.0, ls=":",  label="BER = 10")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Chemical"].str[:18], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("log₁₀(BER)  [lower = higher concern]", fontsize=11)
    ax.set_title("Bayesian BER – Posterior Credible Intervals (90 %)\n"
                 "MC Dropout, 2000 samples per chemical", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_posterior_top5(result_df: pd.DataFrame, out_path: Path) -> None:
    """
    Kernel density plots of BER posterior for the 5 highest-concern chemicals.
    """
    df = result_df.dropna(subset=["BER_median"]).nsmallest(5, "BER_median")

    if len(df) == 0:
        print(f"  WARNING: no chemicals with valid BER_median - skipping {out_path.name}")
        return

    from scipy.stats import gaussian_kde

    n_cols = max(len(df), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, df.iterrows()):
        samples = row["_ber_samples"]
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if len(samples) < 10:
            ax.text(0.5, 0.5, "insufficient data",
                    ha="center", transform=ax.transAxes)
            continue
        log_s = np.log10(np.clip(samples, 1e-8, None))
        xs    = np.linspace(log_s.min() - 0.5, log_s.max() + 0.5, 300)
        kde   = gaussian_kde(log_s)
        ax.plot(xs, kde(xs), color="steelblue", lw=2)
        ax.fill_between(xs, kde(xs), alpha=0.25, color="steelblue")
        ax.axvline(np.log10(row["BER_median"]), color="red",   lw=1.5, ls="--", label="median")
        ax.axvline(np.log10(row["BER_p2_5"]),   color="orange",lw=1.0, ls=":",  label="2.5 %")
        ax.axvline(np.log10(row["BER_p97_5"]),  color="orange",lw=1.0, ls=":",  label="97.5 %")
        ax.axvline(0,                            color="black", lw=1.0, ls="-",  label="BER=1")
        ax.set_xlabel("log₁₀(BER)", fontsize=9)
        ax.set_title(str(row["Chemical"])[:20], fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Posterior BER Distributions – Top-5 Highest-Concern Chemicals",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_clint_uncertainty(
    result_df: pd.DataFrame,
    out_path:  Path,
) -> None:
    """
    Scatter: reference Clint (Step 2) vs. posterior Clint median with 90 % CI.
    """
    df = result_df.copy()
    fig, ax = plt.subplots(figsize=(8, 6))

    x  = np.log10(df["Clint_ref_uL"].clip(0.01))
    y  = np.log10(df["Clint_median"].clip(0.01))
    yl = np.log10(df["Clint_p2_5"].clip(0.01))
    yu = np.log10(df["Clint_p97_5"].clip(0.01))

    ax.errorbar(x, y, yerr=[y - yl, yu - y],
                fmt="o", color="steelblue", ecolor="steelblue",
                elinewidth=1.5, capsize=3, ms=6, label="BNN median ± 90 % CI")

    lims = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("log₁₀(Clint_ref)  [Step 2 best estimate]", fontsize=11)
    ax.set_ylabel("log₁₀(Clint_BNN) [MC Dropout posterior median]", fontsize=11)
    ax.set_title("BNN Clint Prediction Uncertainty\n(90 % credible intervals via MC Dropout)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Step 8 - Bayesian BER (MC Dropout Uncertainty Analysis)")
    print("=" * 65, "\n")

    # ─ Load data ─────────────────────────────────────────────────────────────
    pilot = pd.read_csv(PILOT_CSV)
    print(f"Pilot chemicals:  {len(pilot)}")

    aed_ber = None
    if AED_BER_CSV.exists():
        aed_ber = pd.read_csv(AED_BER_CSV)
        print(f"AED/BER results:  {len(aed_ber)} rows")

    # ─ Feature matrix ────────────────────────────────────────────────────────
    feature_cols = ["MW", "logP", "Fup"]
    target_col   = next((c for c in ("Clint_final", "Clint", "Clint_RF")
                         if c in pilot.columns), None)
    if target_col is None:
        sys.exit("ERROR: no Clint column found in pilot data.")

    available = [c for c in feature_cols if c in pilot.columns]
    sub = pilot[available + [target_col]].dropna()
    X   = sub[available].values.astype(np.float32)
    y   = np.log10(sub[target_col].clip(lower=0.01).values.astype(np.float32))

    print(f"Feature columns:  {available}")
    print(f"Training samples: {len(X)}")

    # ─ Train Bayesian MLP ────────────────────────────────────────────────────
    print(f"\nTraining BNN ({EPOCHS} epochs max, dropout p={DROPOUT_P}) ...")
    model, scaler_X, scaler_y = train_bayesian_mlp(X, y)
    print("Training complete.")

    # ─ MC Dropout posterior sampling ─────────────────────────────────────────
    print(f"Sampling {N_MC} posterior draws per chemical ...")
    pilot_sub  = pilot.loc[sub.index].reset_index(drop=True)
    X_pilot    = sub[available].values.astype(np.float32)
    mc_samples = mc_predict(model, X_pilot, scaler_X, scaler_y, n_mc=N_MC)
    print(f"Posterior samples shape: {mc_samples.shape}")

    # ─ BER propagation ───────────────────────────────────────────────────────
    print("Propagating Clint uncertainty -> AED -> BER ...")
    result_df = propagate_to_ber(mc_samples, pilot_sub, aed_ber)

    # ─ Save CSV (drop raw samples) ───────────────────────────────────────────
    export_df = result_df.drop(columns=["_ber_samples"])
    csv_path  = RESULTS / "bayesian_ber.csv"
    export_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ─ Plots ─────────────────────────────────────────────────────────────────
    plot_credible_intervals(result_df, RESULTS / "ber_credible_intervals.png")
    plot_posterior_top5(result_df,     RESULTS / "ber_posterior_top5.png")
    plot_clint_uncertainty(result_df,  RESULTS / "clint_posterior_uncertainty.png")

    # ─ Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Bayesian BER summary (chemicals with exposure data):")
    print("=" * 65)
    show = export_df.dropna(subset=["BER_median"]).sort_values("BER_median").head(15)
    cols = ["Chemical", "BER_p2_5", "BER_median", "BER_p97_5",
            "Clint_p2_5", "Clint_median", "Clint_p97_5"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].to_string(index=False))
    print("\nOutputs saved to results/")
    print("  bayesian_ber.csv                  - full posterior BER summary")
    print("  ber_credible_intervals.png         - waterfall + 90 % CI")
    print("  ber_posterior_top5.png             - density plots, top-5 chemicals")
    print("  clint_posterior_uncertainty.png    - BNN Clint uncertainty scatter")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
