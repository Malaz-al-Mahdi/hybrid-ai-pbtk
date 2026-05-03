"""
10_clint_literature_validation.py
----------------------------------
Validates ML-predicted Clint against literature (Wetmore 2012 / httk).

Scientific rationale
~~~~~~~~~~~~~~~~~~~~
The ML model (Step 2) is trained on 19 pilot chemicals.  The httk database
(all_777_chemicals.csv -> Human.Clint) contains curated in-vitro Clint
measurements from Wetmore et al. (2012) "Advancing High-Throughput Prioritization
Approaches" and subsequent studies.

Validation design
~~~~~~~~~~~~~~~~~
  A) INTERNAL (LOO-CV, n=19)
     Already reported in Step 2. Re-shown here for completeness.

  B) EXTERNAL (n=777, independent)
     Train on all 19 pilot chemicals.
     Predict for ALL 777 httk chemicals that have measured Human.Clint.
     Metrics computed on this held-out set = genuine external validation.

Metrics
~~~~~~~
  R2 (log10 scale)     -- primary goodness-of-fit
  RMSE (log10)         -- spread of predictions
  Spearman rho         -- rank correlation (robust to outliers)
  Geometric Mean Fold-Error (GMFE) -- regulatory standard for TK models
  % within 2-fold / 3-fold / 10-fold  -- acceptance criteria

Outputs
~~~~~~~
  results/clint_validation_external.csv    Full predicted vs. literature table
  results/clint_validation_metrics.csv     Metric summary (internal + external)
  results/clint_validation_scatter.png     Log-log scatter plots
  results/clint_validation_residuals.png   Residual distribution + Q-Q
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, probplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

PILOT_CSV  = DATA / "pilot_chemicals_full.csv"
FULL_CSV   = DATA / "all_777_chemicals.csv"
LOO_CSV    = DATA / "rf_clint_predictions.csv"

for p in (PILOT_CSV, FULL_CSV):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found.")

EPSILON = 1e-3

# ── Feature engineering (identical to Step 2) ─────────────────────────────────

def engineer_features(df_in: pd.DataFrame) -> np.ndarray:
    mw   = np.clip(pd.to_numeric(df_in["MW"],   errors="coerce").fillna(300).values, 1.0, None)
    logp = pd.to_numeric(df_in["logP"], errors="coerce").fillna(2.0).values
    fup  = np.clip(pd.to_numeric(df_in["Fup"],  errors="coerce").fillna(0.1).values, 1e-6, 1.0)
    return np.column_stack([
        np.log10(mw),
        logp,
        logp ** 2,
        np.log10(fup + 1e-6),
        np.sqrt(fup),
        mw * logp,
        mw * fup,
        logp * fup,
        mw,
    ])

FEATURE_NAMES = [
    "log10_MW", "logP", "logP^2", "log10_Fup", "sqrt_Fup",
    "MW_x_logP", "MW_x_Fup", "logP_x_Fup", "MW",
]


def make_gb():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=2, subsample=0.8, min_samples_leaf=2, random_state=42,
        )),
    ])

def make_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(
            n_estimators=1000, max_features=None,
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        )),
    ])


def compute_metrics(y_true_log, y_pred_log, label: str) -> dict:
    r2    = r2_score(y_true_log, y_pred_log)
    rmse  = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    rho, rho_p = spearmanr(y_true_log, y_pred_log)
    fe    = 10 ** np.abs(y_true_log - y_pred_log)
    gmfe  = float(np.exp(np.mean(np.log(fe))))
    p2    = float(np.mean(fe <= 2.0)  * 100)
    p3    = float(np.mean(fe <= 3.0)  * 100)
    p10   = float(np.mean(fe <= 10.0) * 100)
    n     = len(y_true_log)
    print(f"\n  {label}  (n={n})")
    print(f"    R^2 (log10)      : {r2:.4f}")
    print(f"    RMSE (log10)     : {rmse:.4f}")
    print(f"    Spearman rho     : {rho:.4f}  (p={rho_p:.3e})")
    print(f"    GMFE             : {gmfe:.2f}x")
    print(f"    Within 2-fold    : {p2:.0f} %")
    print(f"    Within 3-fold    : {p3:.0f} %")
    print(f"    Within 10-fold   : {p10:.0f} %")
    return dict(Set=label, N=n, R2_log=round(r2,4), RMSE_log=round(rmse,4),
                Spearman=round(rho,4), Spearman_p=round(rho_p,4),
                GMFE=round(gmfe,2), Pct_2fold=round(p2,1),
                Pct_3fold=round(p3,1), Pct_10fold=round(p10,1))


# ── 1. Load pilot training data ───────────────────────────────────────────────
print("=" * 65)
print("Step 10 - Clint Validation vs. Wetmore 2012 / httk Literature")
print("=" * 65)

pilot     = pd.read_csv(PILOT_CSV)
df_clean  = pilot.dropna(subset=["Clint"]).copy()
df_clean["Fup"] = df_clean["Fup"].clip(lower=1e-6)

X_train = engineer_features(df_clean)
y_train = np.log10(df_clean["Clint"].values + EPSILON)

print(f"\nTraining set (pilot): {len(df_clean)} chemicals with measured Clint")
print(f"Clint range          : {df_clean['Clint'].min():.2f} - {df_clean['Clint'].max():.2f}")

# Determine best model (same logic as Step 2: GB usually wins)
# We re-use the result from rf_clint_predictions.csv
best_model_name = "GradientBoosting"
if LOO_CSV.exists():
    loo_df = pd.read_csv(LOO_CSV)
    if "model" in loo_df.columns:
        best_model_name = loo_df["model"].iloc[0]

print(f"Best model (from Step 2): {best_model_name}")

# Train on ALL pilot chemicals
final_pipe = make_gb() if best_model_name == "GradientBoosting" else make_rf()
final_pipe.fit(X_train, y_train)
print("Model trained on full pilot set.")


# ── 2. Load 777-chemical reference set ───────────────────────────────────────
print(f"\nLoading all_777_chemicals.csv ...")
full = pd.read_csv(FULL_CSV)

# Standardise column names (httk uses Human.Clint / Human.Funbound.plasma)
col_map = {
    "Human.Clint":           "Clint",
    "Human.Funbound.plasma": "Fup",
    "Human.Rblood2plasma":   "Rblood2plasma",
}
full = full.rename(columns=col_map)
full["Clint"] = pd.to_numeric(full["Clint"], errors="coerce")
full["Fup"]   = pd.to_numeric(full["Fup"],   errors="coerce").clip(lower=1e-6)
full["MW"]    = pd.to_numeric(full["MW"],     errors="coerce")
full["logP"]  = pd.to_numeric(full["logP"],   errors="coerce")

# Keep only chemicals with measured Clint (literature reference)
val = full.dropna(subset=["Clint", "MW", "logP", "Fup"]).copy()
val = val[val["Clint"] > 0].copy()   # exclude zero-clearance entries
print(f"Chemicals with measured Clint in httk: {len(val)}")

# Flag pilot chemicals (internal vs. external)
pilot_cas = set(df_clean["CAS"].astype(str).str.strip())
val["in_pilot"] = val["CAS"].astype(str).str.strip().isin(pilot_cas)
n_ext = (~val["in_pilot"]).sum()
print(f"  Pilot (internal)  : {val['in_pilot'].sum()}")
print(f"  External          : {n_ext}")


# ── 3. Predict Clint for all 777 chemicals ────────────────────────────────────
X_val  = engineer_features(val)
pred_log = final_pipe.predict(X_val)
val = val.copy()
val["Clint_pred"]    = np.round(10 ** pred_log - EPSILON, 4)
val["log10_lit"]     = np.round(np.log10(val["Clint"] + EPSILON), 4)
val["log10_pred"]    = np.round(pred_log, 4)
val["fold_error"]    = np.round(10 ** np.abs(val["log10_lit"] - val["log10_pred"]), 3)


# ── 4. Metrics ────────────────────────────────────────────────────────────────
print("\n--- Validation Metrics ---")
all_metrics = []

# A) Internal (LOO from Step 2 results)
if LOO_CSV.exists():
    loo_df = pd.read_csv(LOO_CSV)
    loo_df = loo_df[loo_df["Clint_true"] > 0].copy()
    loo_df["log10_true"] = np.log10(loo_df["Clint_true"] + EPSILON)
    m_int = compute_metrics(loo_df["log10_true"].values,
                            loo_df["log10_pred"].values,
                            "A) Internal LOO-CV (n=19, excl. Clint=0)")
    all_metrics.append(m_int)

# B) All 777 chemicals (includes pilot = optimistic upper bound)
m_all = compute_metrics(val["log10_lit"].values,
                        val["log10_pred"].values,
                        "B) All 777 httk chemicals")
all_metrics.append(m_all)

# C) External only (pilot chemicals excluded)
ext = val[~val["in_pilot"]]
if len(ext) > 0:
    m_ext = compute_metrics(ext["log10_lit"].values,
                            ext["log10_pred"].values,
                            "C) External validation (pilot excluded)")
    all_metrics.append(m_ext)

# D) Chemicals with Clint > 1 (exclude near-zero clearance, more meaningful)
val_gt1 = val[val["Clint"] > 1.0]
if len(val_gt1) > 0:
    m_gt1 = compute_metrics(val_gt1["log10_lit"].values,
                            val_gt1["log10_pred"].values,
                            "D) Clint > 1 uL/min/10^6 (active clearance)")
    all_metrics.append(m_gt1)

metrics_df = pd.DataFrame(all_metrics)
metrics_path = RESULTS / "clint_validation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved: {metrics_path}")


# ── 5. Export full validation table ──────────────────────────────────────────
export_cols = ["CAS", "Compound", "Clint", "Clint_pred",
               "log10_lit", "log10_pred", "fold_error", "in_pilot"]
export = val[[c for c in export_cols if c in val.columns]].copy()
export = export.rename(columns={"Clint": "Clint_literature_uL_min_Mcells"})
export = export.sort_values("fold_error", ascending=False)

ext_path = RESULTS / "clint_validation_external.csv"
export.to_csv(ext_path, index=False)
print(f"Full table saved: {ext_path}  ({len(export)} chemicals)")


# ── 6. Scatter plots ──────────────────────────────────────────────────────────
print("\nGenerating plots ...")

fig = plt.figure(figsize=(20, 5))
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

def fold_color(fe):
    """Color by fold-error: green=2x, orange=3x, red=10x, darkred=beyond."""
    colors = []
    for f in fe:
        if f <= 2.0:   colors.append("#2196F3")   # blue = within 2-fold
        elif f <= 3.0: colors.append("#4CAF50")   # green = within 3-fold
        elif f <= 10.: colors.append("#FF9800")   # orange = within 10-fold
        else:           colors.append("#F44336")   # red = outside 10-fold
    return colors

# -- Panel A: Internal LOO-CV (pilot, excl. Clint=0) -------------------------
ax = fig.add_subplot(gs[0])
if LOO_CSV.exists():
    loo_pos = loo_df[loo_df["Clint_true"] > 0]
    fe_col  = fold_color(10 ** np.abs(loo_pos["log10_true"] - loo_pos["log10_pred"]))
    ax.scatter(loo_pos["log10_true"], loo_pos["log10_pred"],
               c=fe_col, s=80, edgecolors="k", linewidths=0.5, zorder=3)
    for _, row in loo_pos.iterrows():
        ax.annotate(str(row["Compound"])[:10],
                    (row["log10_true"], row["log10_pred"]),
                    fontsize=5, alpha=0.7)
    lims = [min(loo_pos["log10_true"].min(), loo_pos["log10_pred"].min()) - 0.5,
            max(loo_pos["log10_true"].max(), loo_pos["log10_pred"].max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="ideal")
    ax.fill_between(lims, [v-np.log10(3) for v in lims],
                    [v+np.log10(3) for v in lims], alpha=0.08, color="green")
    m = all_metrics[0]
    ax.set_title(f"A) Internal LOO-CV (n={m['N']})\n"
                 f"R²={m['R2_log']:.3f}  GMFE={m['GMFE']:.1f}x", fontsize=9)
    ax.set_xlabel("log10(Clint measured)", fontsize=8)
    ax.set_ylabel("log10(Clint predicted)", fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

# -- Panel B: All 777 chemicals -----------------------------------------------
ax = fig.add_subplot(gs[1])
fe_col = fold_color(val["fold_error"].values)
# Pilot points highlighted differently
pilot_mask = val["in_pilot"].values
ax.scatter(val.loc[~pilot_mask, "log10_lit"],
           val.loc[~pilot_mask, "log10_pred"],
           c=[fe_col[i] for i in range(len(fe_col)) if not pilot_mask[i]],
           s=20, edgecolors="none", alpha=0.6, label="External")
ax.scatter(val.loc[pilot_mask, "log10_lit"],
           val.loc[pilot_mask, "log10_pred"],
           c="gold", s=80, edgecolors="k", linewidths=0.8,
           zorder=5, label="Pilot (train)")
lims2 = [val["log10_lit"].min() - 0.5, val["log10_lit"].max() + 0.5]
ax.plot(lims2, lims2, "k--", lw=1.2)
ax.fill_between(lims2, [v-np.log10(3) for v in lims2],
                [v+np.log10(3) for v in lims2], alpha=0.08, color="green")
m = m_all
ax.set_title(f"B) All 777 httk chemicals (n={m['N']})\n"
             f"R²={m['R2_log']:.3f}  GMFE={m['GMFE']:.1f}x", fontsize=9)
ax.set_xlabel("log10(Clint literature)", fontsize=8)
ax.set_ylabel("log10(Clint predicted)", fontsize=8)
ax.set_xlim(lims2); ax.set_ylim(lims2)
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)

# -- Panel C: External only ---------------------------------------------------
ax = fig.add_subplot(gs[2])
if len(ext) > 0:
    fe_col_ext = fold_color(ext["fold_error"].values)
    ax.scatter(ext["log10_lit"], ext["log10_pred"],
               c=fe_col_ext, s=20, edgecolors="none", alpha=0.7)
    ax.plot(lims2, lims2, "k--", lw=1.2)
    ax.fill_between(lims2, [v-np.log10(3) for v in lims2],
                    [v+np.log10(3) for v in lims2], alpha=0.08, color="green")
    m = m_ext
    ax.set_title(f"C) External validation (n={m['N']})\n"
                 f"R²={m['R2_log']:.3f}  GMFE={m['GMFE']:.1f}x", fontsize=9)
    ax.set_xlabel("log10(Clint literature)", fontsize=8)
    ax.set_ylabel("log10(Clint predicted)", fontsize=8)
    ax.set_xlim(lims2); ax.set_ylim(lims2)
    ax.grid(True, alpha=0.3)
# Legend for fold-error colours
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#2196F3", label="within 2-fold"),
    Patch(facecolor="#4CAF50", label="within 3-fold"),
    Patch(facecolor="#FF9800", label="within 10-fold"),
    Patch(facecolor="#F44336", label="outside 10-fold"),
]
ax.legend(handles=legend_els, fontsize=6, loc="upper left")

# -- Panel D: Residual histogram (external) -----------------------------------
ax = fig.add_subplot(gs[3])
log_res = (ext["log10_pred"] - ext["log10_lit"]).values if len(ext) > 0 else \
          (val["log10_pred"] - val["log10_lit"]).values
ax.hist(log_res, bins=40, color="#2196F380", edgecolor="white")
ax.axvline(0, color="red", lw=2, label="no bias")
ax.axvline(np.log10(2),  color="steelblue", lw=1.5, ls="--", label="2-fold")
ax.axvline(-np.log10(2), color="steelblue", lw=1.5, ls="--")
ax.axvline(np.log10(3),  color="darkgreen", lw=1.5, ls=":",  label="3-fold")
ax.axvline(-np.log10(3), color="darkgreen", lw=1.5, ls=":")
bias = 10 ** np.mean(log_res)
ax.set_xlabel("log10(Predicted / Literature Clint)", fontsize=8)
ax.set_ylabel("Count", fontsize=8)
ax.set_title(f"D) Residual distribution\nBias (GMR) = {bias:.2f}x", fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Clint Validation: {best_model_name} (trained on 19 pilot chemicals)\n"
    "vs. Wetmore 2012 / httk literature (Human.Clint)",
    fontsize=11, y=1.02,
)
plt.savefig(RESULTS / "clint_validation_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: results/clint_validation_scatter.png")


# ── 7. Q-Q plot for residuals ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
(osm, osr), (slope, intercept, r) = probplot(log_res, dist="norm")
ax.scatter(osm, osr, s=10, color="#2196F3", alpha=0.6, edgecolors="none")
ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
ax.set_xlabel("Theoretical quantiles", fontsize=9)
ax.set_ylabel("Sample quantiles", fontsize=9)
ax.set_title(f"Q-Q plot: log10(Pred/Lit)\nR={r:.3f}", fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(val["log10_lit"], val["log10_pred"] - val["log10_lit"],
           c=fold_color(val["fold_error"].values), s=15, alpha=0.6, edgecolors="none")
ax.axhline(0, color="red", lw=1.5)
ax.axhline(np.log10(3),  color="darkgreen", lw=1, ls="--", label="+3-fold")
ax.axhline(-np.log10(3), color="darkgreen", lw=1, ls="--", label="-3-fold")
ax.set_xlabel("log10(Clint literature)", fontsize=9)
ax.set_ylabel("Residual: log10(Pred - Lit)", fontsize=9)
ax.set_title("Residuals vs. Observed Clint", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / "clint_validation_residuals.png", dpi=150)
plt.close()
print(f"Saved: results/clint_validation_residuals.png")


# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY -- Clint Prediction vs. Wetmore 2012 / httk Literature")
print("=" * 65)
print(metrics_df[["Set","N","R2_log","RMSE_log","GMFE",
                   "Pct_2fold","Pct_3fold","Pct_10fold"]].to_string(index=False))

print("\nTop-10 worst predictions (external set):")
worst = export[~export.get("in_pilot", False)].head(10) if "in_pilot" in export.columns \
        else export.head(10)
print(worst[["Compound","Clint_literature_uL_min_Mcells",
             "Clint_pred","fold_error"]].to_string(index=False))

print("\nOutputs:")
print("  results/clint_validation_external.csv   -- full predicted vs. literature")
print("  results/clint_validation_metrics.csv    -- R^2, RMSE, GMFE per set")
print("  results/clint_validation_scatter.png    -- 4-panel scatter plots")
print("  results/clint_validation_residuals.png  -- Q-Q + residual plots")
print("\nDone.")
