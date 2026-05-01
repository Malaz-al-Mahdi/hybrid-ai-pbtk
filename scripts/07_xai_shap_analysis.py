"""
07_xai_shap_analysis.py
-----------------------
Explainable AI (XAI) for the toxicokinetic / risk-prioritisation pipeline.

Scientific rationale
~~~~~~~~~~~~~~~~~~~~
ML models used in regulatory toxicology must be interpretable – regulators
and risk assessors need to understand *why* a model predicts a high or low
Clint (and thus a high or low BER) for a given chemical.  SHAP (SHapley
Additive exPlanations) provides a unified, theoretically grounded framework
for local and global feature attribution.

This script applies SHAP to:
  A) Random Forest model for Clint prediction (Step 2)
     → Which physicochemical features most drive log10(Clint)?
     → Are there non-linear or interaction effects?
  B) BER prediction from AED/exposure data (Step 5/6)
     → Which input variables explain the BER ranking?

Plots
~~~~~
  results/shap_rf_summary_bar.png        Global RF SHAP bar chart
  results/shap_rf_beeswarm.png           RF SHAP beeswarm plot
  results/shap_rf_dependence_logP.png    logP dependence + interaction
  results/shap_rf_dependence_Fup.png     Fup dependence + interaction
  results/shap_ber_beeswarm.png          BER explainability beeswarm

Data exports
~~~~~~~~~~~~
  results/shap_rf_values.csv             Per-chemical SHAP values (RF)
  results/shap_ber_values.csv            Per-chemical SHAP values (BER model)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import shap
except ImportError:
    sys.exit(
        "ERROR: shap is required.  Install with:\n"
        "  pip install shap\n"
        "or:  py -m pip install shap"
    )

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("ERROR: scikit-learn is required.  pip install scikit-learn")

# ── Required data ─────────────────────────────────────────────────────────────
PILOT_CSV  = DATA  / "pilot_chemicals_imputed.csv"
AED_BER_CSV = RESULTS / "aed_ber_full.csv"

for p in (PILOT_CSV,):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found.  Run steps 01 and 02 first.")


# ── A: Random Forest Clint explainability ─────────────────────────────────────

def section_a_rf_clint(pilot: pd.DataFrame) -> None:
    """
    Retrain the Random Forest from Step 2 and compute SHAP values.
    """
    print("\n── A) RF Clint prediction – SHAP analysis ──")

    # Features used in Step 2
    feature_cols = ["MW", "logP", "Fup"]
    available    = [c for c in feature_cols if c in pilot.columns]

    target_col = None
    for col in ("Clint_final", "Clint", "Clint_RF"):
        if col in pilot.columns:
            target_col = col
            break
    if target_col is None:
        print("  WARNING: no Clint column found – skipping RF section.")
        return

    sub = pilot[available + [target_col]].dropna()
    X   = sub[available].values.astype(np.float32)
    y   = np.log10(sub[target_col].clip(lower=0.01).values).astype(np.float32)

    print(f"  Training data: {len(sub)} chemicals  |  features: {available}")

    # ─ Train RF (same hyper-params as Step 2) ────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    cv_r2 = cross_val_score(rf, X, y, cv=min(5, len(sub)),
                             scoring="r2").mean()
    print(f"  RF cross-validated R² = {cv_r2:.3f}")

    # ─ SHAP TreeExplainer ────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer(X)           # shap.Explanation object

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=[f"SHAP_{c}" for c in available],
    )
    shap_df.insert(0, "Chemical", sub["Compound"].values
                   if "Compound" in sub.columns else sub.index.values)
    shap_df.insert(1, "log10_Clint_pred",
                   rf.predict(X).round(3))
    shap_csv = RESULTS / "shap_rf_values.csv"
    shap_df.to_csv(shap_csv, index=False)
    print(f"  Saved {shap_csv}")

    # ─ Plot 1: Global bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    colors   = cm.RdBu_r(np.linspace(0.2, 0.8, len(available)))
    bars = ax.barh(
        [available[i] for i in order],
        [mean_abs[i] for i in order],
        color=[colors[k] for k in range(len(order))],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value|  (impact on log₁₀(Clint))", fontsize=10)
    ax.set_title("Global Feature Importance – RF Clint Model (SHAP)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p1 = RESULTS / "shap_rf_summary_bar.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Saved {p1}")

    # ─ Plot 2: Beeswarm ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5),
                             sharey=False)
    if len(available) == 1:
        axes = [axes]

    for k, feat in enumerate(available):
        feat_vals  = X[:, k]
        shap_feat  = shap_values.values[:, k]
        sc = axes[k].scatter(
            shap_feat, feat_vals,
            c=feat_vals, cmap="RdBu_r", s=70,
            edgecolors="k", linewidths=0.4, alpha=0.85,
        )
        axes[k].axvline(0, color="gray", lw=0.8, ls="--")
        axes[k].set_xlabel("SHAP value", fontsize=9)
        axes[k].set_ylabel(feat, fontsize=9)
        axes[k].set_title(feat, fontsize=10, fontweight="bold")
        axes[k].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[k], label=feat, fraction=0.04, pad=0.04)

    fig.suptitle("SHAP Beeswarm – RF Clint Prediction\n"
                 "Each dot = one chemical; colour = feature value",
                 fontsize=11)
    plt.tight_layout()
    p2 = RESULTS / "shap_rf_beeswarm.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Saved {p2}")

    # ─ Plot 3 & 4: Dependence plots ───────────────────────────────────────────
    for feat_main, feat_interact in [
        ("logP", "Fup"),
        ("Fup",  "logP"),
        ("MW",   "logP"),
    ]:
        if feat_main not in available:
            continue
        interact_feat = feat_interact if feat_interact in available else None
        idx_main = available.index(feat_main)

        fig, ax = plt.subplots(figsize=(7, 5))
        x_vals   = X[:, idx_main]
        sh_vals  = shap_values.values[:, idx_main]

        if interact_feat:
            idx_int = available.index(interact_feat)
            color_vals = X[:, idx_int]
            cmap = "RdBu_r"
        else:
            color_vals = sh_vals
            cmap = "viridis"

        sc = ax.scatter(x_vals, sh_vals, c=color_vals, cmap=cmap,
                        s=80, edgecolors="k", linewidths=0.5, alpha=0.85)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(feat_main, fontsize=11)
        ax.set_ylabel(f"SHAP value for {feat_main}", fontsize=11)
        cb_label = interact_feat if interact_feat else feat_main
        plt.colorbar(sc, ax=ax, label=cb_label)
        ax.set_title(
            f"SHAP Dependence Plot: {feat_main}\n"
            f"(interaction colour: {cb_label})",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = feat_main.replace("/", "_").replace(" ", "_")
        dep_path  = RESULTS / f"shap_rf_dependence_{safe_name}.png"
        plt.savefig(dep_path, dpi=150)
        plt.close()
        print(f"  Saved {dep_path}")


# ── B: BER explainability ─────────────────────────────────────────────────────

def section_b_ber(pilot: pd.DataFrame) -> None:
    """
    Fit a gradient-boosted regressor to predict log10(BER) from chemical
    descriptors and compute SHAP values.  This makes the BER ranking
    interpretable: we can explain *why* a chemical is high-priority.
    """
    print("\n── B) BER ranking – SHAP analysis ──")

    if not AED_BER_CSV.exists():
        print(f"  WARNING: {AED_BER_CSV} not found – skipping BER section.")
        return

    ber_df  = pd.read_csv(AED_BER_CSV)
    ber_sub = ber_df[ber_df["BER"].notna() & (ber_df["BER"] > 0)].copy()

    # Merge with pilot descriptors (CAS-based)
    merge_cols = ["MW", "logP", "Fup"]
    cas_col    = "CAS" if "CAS" in pilot.columns else None
    if cas_col and "CAS" in ber_sub.columns:
        merged = ber_sub.merge(
            pilot[[cas_col] + [c for c in merge_cols if c in pilot.columns]],
            on="CAS", how="left",
        )
    else:
        merged = ber_sub.copy()

    feature_cols = [c for c in merge_cols if c in merged.columns]

    # Supplement with derived columns from AED/BER table
    for col in ("AED_median", "AC50_5pct_uM", "Exposure_median_mg_kg_day"):
        if col in merged.columns:
            feature_cols.append(col)

    sub = merged[feature_cols + ["BER"]].dropna()
    if len(sub) < 5:
        print(f"  Only {len(sub)} complete rows – not enough for SHAP.  Skipping.")
        return

    X_ber = sub[feature_cols].values.astype(np.float32)
    y_ber = np.log10(sub["BER"].clip(lower=1e-6).values).astype(np.float32)

    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    gb.fit(X_ber, y_ber)
    cv_r2 = cross_val_score(gb, X_ber, y_ber,
                             cv=min(5, len(sub)), scoring="r2").mean()
    print(f"  GB BER model cross-validated R² = {cv_r2:.3f}")

    explainer_ber = shap.TreeExplainer(gb)
    shap_ber      = explainer_ber(X_ber)

    # Save
    shap_ber_df = pd.DataFrame(
        shap_ber.values,
        columns=[f"SHAP_{c}" for c in feature_cols],
    )
    shap_ber_df.insert(0, "log10_BER", y_ber.round(3))
    shap_ber_df.insert(0, "Chemical", merged["Compound"].values[:len(sub)]
                       if "Compound" in merged.columns else np.arange(len(sub)))
    shap_csv_ber = RESULTS / "shap_ber_values.csv"
    shap_ber_df.to_csv(shap_csv_ber, index=False)
    print(f"  Saved {shap_csv_ber}")

    # Beeswarm plot
    fig, axes = plt.subplots(1, len(feature_cols),
                              figsize=(4 * len(feature_cols), 5), sharey=False)
    if len(feature_cols) == 1:
        axes = [axes]

    for k, feat in enumerate(feature_cols):
        feat_vals  = X_ber[:, k]
        shap_feat  = shap_ber.values[:, k]
        sc = axes[k].scatter(
            shap_feat, feat_vals, c=feat_vals,
            cmap="RdBu_r", s=70, edgecolors="k", linewidths=0.4, alpha=0.85,
        )
        axes[k].axvline(0, color="gray", lw=0.8, ls="--")
        axes[k].set_xlabel("SHAP value", fontsize=9)
        axes[k].set_ylabel(feat, fontsize=9)
        axes[k].set_title(feat, fontsize=9, fontweight="bold")
        axes[k].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[k], label=feat, fraction=0.04, pad=0.04)

    fig.suptitle("SHAP Beeswarm – BER Risk Prioritisation\n"
                 "Each dot = one chemical; SHAP → impact on log₁₀(BER)",
                 fontsize=11)
    plt.tight_layout()
    p_ber = RESULTS / "shap_ber_beeswarm.png"
    plt.savefig(p_ber, dpi=150)
    plt.close()
    print(f"  Saved {p_ber}")

    # ─ Local explanation: top-3 highest-concern chemicals ────────────────────
    top3_idx = np.argsort(y_ber)[:3]   # lowest BER = highest concern
    fig, axes = plt.subplots(1, len(top3_idx), figsize=(5 * len(top3_idx), 5))
    if len(top3_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, top3_idx):
        sh  = shap_ber.values[idx]
        base = float(shap_ber.base_values[0])
        names = feature_cols
        colors = ["#d73027" if v > 0 else "#4575b4" for v in sh]
        y_pos  = range(len(names))
        ax.barh(list(y_pos), sh, color=colors, edgecolor="k", lw=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        chem_name = str(shap_ber_df.iloc[idx]["Chemical"])[:20]
        ax.set_title(
            f"{chem_name}\nlog₁₀(BER)={y_ber[idx]:.2f}  (base={base:.2f})",
            fontsize=9,
        )
        ax.set_xlabel("SHAP", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Local SHAP Explanation – Top-3 Highest-Concern Chemicals",
                 fontsize=11)
    plt.tight_layout()
    local_path = RESULTS / "shap_ber_local_top3.png"
    plt.savefig(local_path, dpi=150)
    plt.close()
    print(f"  Saved {local_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Step 7 – Explainable AI (SHAP) for Clint and BER")
    print("=" * 65)

    pilot = pd.read_csv(PILOT_CSV)
    print(f"Loaded {len(pilot)} pilot chemicals from {PILOT_CSV.name}")

    section_a_rf_clint(pilot)
    section_b_ber(pilot)

    print("\n" + "=" * 65)
    print("XAI outputs:")
    for fname in sorted(RESULTS.glob("shap_*.png")):
        print(f"  {fname.name}")
    for fname in sorted(RESULTS.glob("shap_*.csv")):
        print(f"  {fname.name}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
