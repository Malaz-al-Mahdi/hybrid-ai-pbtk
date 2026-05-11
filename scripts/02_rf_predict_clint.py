"""
02_rf_predict_clint.py
----------------------
Trains a Random Forest regressor to predict hepatic intrinsic clearance (Clint)
from physicochemical descriptors (MW, logP, Fup).

Key improvements over naive RF:
  1. Feature engineering  – log-transforms + interaction terms (9 features total)
  2. Fixed max_features=None – with only 3 raw features, sqrt(3)=1 is too restrictive
  3. Gradient Boosting as alternative model – often better on small datasets
  4. Primary metric on log10 scale – not dominated by outlier Thiram (Clint=816)
  5. Geometric Mean Fold-Error (GMFE) – standard benchmark in TK literature

Evaluation strategy:
  - Leave-One-Out Cross-Validation (LOO-CV) on the pilot chemicals
  - Both RF and GB evaluated; best model by log10-R^2 used for imputation
  - External validation vs. Wetmore 2012 / httk literature (all 777 chemicals)
    (merged from former 10_clint_literature_validation.py)

Outputs:
  data/rf_clint_predictions.csv         - LOO-CV predictions vs. true values
  results/rf_loo_cv_metrics.txt         - summary statistics (internal)
  results/rf_loo_cv_scatter.png         - observed vs. predicted scatter plot
  data/pilot_chemicals_imputed.csv      - full table with RF-imputed Clint
  results/clint_validation_metrics.csv  - internal + external metrics
  results/clint_validation_external.csv - full external prediction table
  results/clint_validation_scatter.png  - log-log scatter (4 panels)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, probplot
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from utils import (
    ROOT, DATA, RESULTS, PILOT_CSV as FULL_CSV, ALL_777_CSV,
    EPSILON, engineer_features, FEATURE_NAMES, compute_metrics,
)

if not FULL_CSV.exists():
    sys.exit(f"ERROR: {FULL_CSV} not found. Run 01_extract_httk_data.R first.")

# ---- 1. Load data -----------------------------------------------------------
df = pd.read_csv(FULL_CSV)
print(f"Loaded {len(df)} chemicals from {FULL_CSV.name}")
print(df[["CAS", "Compound", "Clint"]].to_string(index=False))
print()

RAW_FEATURES = ["MW", "logP", "Fup"]
TARGET       = "Clint"

df_clean = df.dropna(subset=[TARGET]).copy()
print(f"Rows with observed Clint : {len(df_clean)}")
print("Missing raw feature values:")
print(df_clean[RAW_FEATURES].isna().sum().to_string())

if len(df_clean) < 5:
    sys.exit("Too few rows with observed Clint for meaningful training.")

X = engineer_features(df_clean)
y     = df_clean[TARGET].values
y_log = np.log10(y + EPSILON)

print(f"\nEngineered feature matrix : {X.shape[0]} x {X.shape[1]}")
print(f"Clint range : {y.min():.2f} - {y.max():.2f}  "
      f"(log10: {y_log.min():.2f} - {y_log.max():.2f})")


# ---- 3. Model definitions with tuned hyperparameters -----------------------
# Hyperparameter choices for small datasets (n~18 per fold):
#
# RF:
#   max_features=None  -> use ALL 9 engineered features at every split
#                         (sqrt(9)=3 was the old default -- too restrictive)
#   min_samples_leaf=1 -> allow individual leaves (n is tiny)
#   n_estimators=1000  -> many trees for stable variance reduction
#   max_depth=None     -> let trees grow fully; bagging handles overfitting
#
# GB:
#   learning_rate=0.05 -> conservative; prevents overfitting on small sets
#   n_estimators=200   -> few but precise boosting rounds
#   max_depth=2        -> shallow stumps for n~18 (avoids high variance)
#   subsample=0.8      -> stochastic gradient boosting for regularisation
#   min_samples_leaf=2 -> slight leaf regularisation

def make_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(
            n_estimators=1000,
            max_features=None,       # all 9 features at every split
            min_samples_leaf=1,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])

def make_gb():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=42,
        )),
    ])


# ---- 4. LOO-CV --------------------------------------------------------------
loo            = LeaveOneOut()
y_pred_rf      = np.full_like(y_log, np.nan)
y_pred_gb      = np.full_like(y_log, np.nan)
feat_imp_rf    = np.zeros(X.shape[1])

print("\nLOO-CV fold results:")
print(f"{'#':>3}  {'Chemical':<28}  {'True':>6}  {'RF':>6}  {'GB':>6}  {'FE_RF':>6}  {'FE_GB':>6}")
print("-" * 65)

for fold, (tr, te) in enumerate(loo.split(X)):
    X_tr, X_te = X[tr], X[te]
    y_tr        = y_log[tr]
    name        = df_clean.iloc[te[0]]["Compound"]

    # Random Forest
    rf = make_rf()
    rf.fit(X_tr, y_tr)
    y_pred_rf[te] = rf.predict(X_te)
    feat_imp_rf  += rf.named_steps["model"].feature_importances_

    # Gradient Boosting
    gb = make_gb()
    gb.fit(X_tr, y_tr)
    y_pred_gb[te] = gb.predict(X_te)

    fe_rf = 10 ** abs(y_log[te[0]] - y_pred_rf[te[0]])
    fe_gb = 10 ** abs(y_log[te[0]] - y_pred_gb[te[0]])
    print(f"{fold+1:>3}  {name:<28}  "
          f"{y_log[te[0]]:>6.2f}  "
          f"{y_pred_rf[te[0]]:>6.2f}  "
          f"{y_pred_gb[te[0]]:>6.2f}  "
          f"{fe_rf:>6.2f}x  "
          f"{fe_gb:>6.2f}x")

feat_imp_rf /= len(df_clean)


# ---- 5. Select best model ---------------------------------------------------
r2_rf_log = r2_score(y_log, y_pred_rf)
r2_gb_log = r2_score(y_log, y_pred_gb)

if r2_gb_log > r2_rf_log:
    best_name    = "GradientBoosting"
    y_pred_log   = y_pred_gb
else:
    best_name    = "RandomForest"
    y_pred_log   = y_pred_rf

print(f"\nLog10-R^2 : RF={r2_rf_log:.4f}   GB={r2_gb_log:.4f}")
print(f"=> Best model: {best_name}")


# ---- 6. Metrics -------------------------------------------------------------
y_pred_orig = np.clip(10 ** y_pred_log - EPSILON, 0, None)

rmse_log  = float(np.sqrt(mean_squared_error(y_log, y_pred_log)))
r2_log    = float(r2_score(y_log, y_pred_log))
rho_log, rho_p_log = spearmanr(y_log, y_pred_log)

rmse_orig = float(np.sqrt(mean_squared_error(y, y_pred_orig)))
r2_orig   = float(r2_score(y, y_pred_orig))
rho_orig, rho_p_orig = spearmanr(y, y_pred_orig)

fold_errors = 10 ** np.abs(y_log - y_pred_log)
gmfe        = float(np.exp(np.mean(np.log(fold_errors))))
pct_2fold   = float(np.mean(fold_errors <= 2.0) * 100)
pct_3fold   = float(np.mean(fold_errors <= 3.0) * 100)


# ---- 7. Per-chemical results ------------------------------------------------
results_df = df_clean[["CAS", "Compound"]].copy()
results_df["Clint_true"]    = y
results_df["Clint_pred"]    = np.round(y_pred_orig, 4)
results_df["abs_error"]     = np.round(np.abs(y - y_pred_orig), 4)
results_df["log10_true"]    = np.round(y_log, 4)
results_df["log10_pred"]    = np.round(y_pred_log, 4)
results_df["fold_error"]    = np.round(fold_errors, 3)
results_df["model"]         = best_name

results_df.to_csv(DATA / "rf_clint_predictions.csv", index=False)
print("\nLOO-CV per-chemical results:")
print(results_df[["Compound","Clint_true","Clint_pred",
                   "log10_true","log10_pred","fold_error"]].to_string(index=False))


# ---- 8. Metrics report ------------------------------------------------------
fi_pairs = sorted(zip(FEATURE_NAMES, feat_imp_rf), key=lambda x: -x[1])

metrics_text = (
    f"Clint Prediction  --  LOO-CV  (n = {len(df_clean)} chemicals)\n"
    f"Best model: {best_name}\n"
    f"{'='*52}\n\n"
    f"PRIMARY METRIC (log10 scale  --  recommended for TK data)\n"
    f"  R^2  log10           : {r2_log:.4f}\n"
    f"  RMSE log10           : {rmse_log:.4f}  log10 units\n"
    f"  Spearman rho         : {rho_log:.4f}  (p = {rho_p_log:.4e})\n"
    f"  Geom. mean fold-error: {gmfe:.2f}x\n"
    f"  Within 2-fold        : {pct_2fold:.0f} %\n"
    f"  Within 3-fold        : {pct_3fold:.0f} %\n\n"
    f"ORIGINAL SCALE (informative; note: Thiram outlier Clint=816)\n"
    f"  RMSE (uL/min/10^6)   : {rmse_orig:.4f}\n"
    f"  R^2                  : {r2_orig:.4f}\n"
    f"  Spearman rho         : {rho_orig:.4f}  (p = {rho_p_orig:.4e})\n\n"
    f"Model comparison (log10-R^2):\n"
    f"  RandomForest         : {r2_rf_log:.4f}\n"
    f"  GradientBoosting     : {r2_gb_log:.4f}\n\n"
    f"RF Feature importances (mean across LOO folds):\n"
)
for feat, imp in fi_pairs:
    metrics_text += f"  {feat:20s}: {imp:.4f}\n"

with open(RESULTS / "rf_loo_cv_metrics.txt", "w") as f:
    f.write(metrics_text)
print(f"\n{metrics_text}")


# ---- 9. Scatter plots -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 9a. Original scale
ax = axes[0]
ax.scatter(y, y_pred_orig, edgecolors="steelblue", facecolors="lightblue", s=60)
lim = max(y.max(), y_pred_orig.max()) * 1.1
ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal")
ax.set_xlabel("Observed Clint (uL/min/10^6 cells)")
ax.set_ylabel(f"Predicted Clint ({best_name})")
ax.set_title(f"Original scale  |  R2={r2_orig:.3f}")
ax.legend()

# 9b. Log10 scale – primary evaluation
ax = axes[1]
sc = ax.scatter(y_log, y_pred_log,
                c=np.log10(fold_errors), cmap="RdYlGn_r", s=80,
                edgecolors="k", linewidths=0.4, vmin=0, vmax=np.log10(10))
lim_log = [min(y_log.min(), y_pred_log.min()) - 0.3,
           max(y_log.max(), y_pred_log.max()) + 0.3]
ax.plot(lim_log, lim_log, "k--", alpha=0.5, label="ideal")
ax.fill_between(lim_log,
                [v - np.log10(3) for v in lim_log],
                [v + np.log10(3) for v in lim_log],
                alpha=0.1, color="green", label="3-fold band")
plt.colorbar(sc, ax=ax, label="log10(fold-error)")
ax.set_xlabel("Observed log10(Clint)")
ax.set_ylabel("Predicted log10(Clint)")
ax.set_title(f"Log10 scale  |  R2={r2_log:.3f}  GMFE={gmfe:.2f}x")
ax.legend(fontsize=8)
for i, row in results_df.iterrows():
    ax.annotate(row["Compound"][:10],
                (row["log10_true"], row["log10_pred"]),
                fontsize=6, alpha=0.7)

# 9c. RF vs GB comparison
ax = axes[2]
ax.scatter(y_log, y_pred_rf, label=f"RF  R2={r2_rf_log:.3f}",
           edgecolors="steelblue", facecolors="lightblue", s=60)
ax.scatter(y_log, y_pred_gb, label=f"GB  R2={r2_gb_log:.3f}",
           edgecolors="tomato", facecolors="lightsalmon", s=60, marker="^")
ax.plot(lim_log, lim_log, "k--", alpha=0.5)
ax.set_xlabel("Observed log10(Clint)")
ax.set_ylabel("Predicted log10(Clint)")
ax.set_title("RF vs. GradientBoosting (log10 scale)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS / "rf_loo_cv_scatter.png", dpi=150)
print(f"Plot saved to results/rf_loo_cv_scatter.png")


# ---- 10. Export imputed dataset ---------------------------------------------
# Retrain best model on ALL labelled data (no LOO needed for imputation)

if best_name == "GradientBoosting":
    final_pipe = make_gb()
else:
    final_pipe = make_rf()

final_pipe.fit(X, y_log)

df_all  = pd.read_csv(FULL_CSV)
X_all   = engineer_features(df_all)
pred_log = final_pipe.predict(X_all)

df_all["Clint_RF"]    = np.round(10 ** pred_log - EPSILON, 4)
df_all["Clint_final"] = df_all[TARGET]
na_mask               = df_all["Clint_final"].isna()
df_all.loc[na_mask, "Clint_final"]  = df_all.loc[na_mask, "Clint_RF"]
df_all["Clint_source"] = "httk"
df_all.loc[na_mask, "Clint_source"] = "RF_predicted"

df_all.to_csv(DATA / "pilot_chemicals_imputed.csv", index=False)
print(f"Saved data/pilot_chemicals_imputed.csv  ({len(df_all)} chemicals)")


# ── 11. Externe Validierung vs. Wetmore 2012 / httk (ehemals Step 10) ────────
print("\n" + "=" * 65)
print("Step 2b – Externe Validierung vs. Wetmore 2012 / httk-Literatur")
print("=" * 65)

if not ALL_777_CSV.exists():
    print(f"  WARNUNG: {ALL_777_CSV} nicht gefunden – externe Validierung uebersprungen.")
    print("Done.")
else:
    # Lade und standardisiere all_777_chemicals.csv
    full777 = pd.read_csv(ALL_777_CSV)
    full777 = full777.rename(columns={
        "Human.Clint":           "Clint",
        "Human.Funbound.plasma": "Fup",
        "Human.Rblood2plasma":   "Rblood2plasma",
    })
    for col in ("Clint", "Fup", "MW", "logP"):
        full777[col] = pd.to_numeric(full777[col], errors="coerce")
    full777["Fup"] = full777["Fup"].clip(lower=1e-6)

    # Nur Chemikalien mit gemessenem Clint (Literatur-Referenz)
    val777 = full777.dropna(subset=["Clint", "MW", "logP", "Fup"]).copy()
    val777 = val777[val777["Clint"] > 0].copy()
    print(f"\nChemikalien mit gemessenem Clint in httk: {len(val777)}")

    pilot_cas_set = set(df_clean["CAS"].astype(str).str.strip())
    val777["in_pilot"] = val777["CAS"].astype(str).str.strip().isin(pilot_cas_set)
    print(f"  Pilot (intern) : {val777['in_pilot'].sum()}")
    print(f"  Extern         : {(~val777['in_pilot']).sum()}")

    X_val777  = engineer_features(val777)
    pred_log777 = final_pipe.predict(X_val777)
    val777 = val777.copy()
    val777["Clint_pred"]  = np.round(10 ** pred_log777 - EPSILON, 4)
    val777["log10_lit"]   = np.round(np.log10(val777["Clint"] + EPSILON), 4)
    val777["log10_pred"]  = np.round(pred_log777, 4)
    val777["fold_error"]  = np.round(10 ** np.abs(val777["log10_lit"] - val777["log10_pred"]), 3)

    # ─ Metriken ────────────────────────────────────────────────────────────────
    print("\n--- Validierungsmetriken ---")
    all_val_metrics = []

    def _metrics_dict(y_true_log, y_pred_log, label):
        r2   = r2_score(y_true_log, y_pred_log)
        rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
        rho, rho_p = spearmanr(y_true_log, y_pred_log)
        fe   = 10 ** np.abs(y_true_log - y_pred_log)
        gmfe = float(np.exp(np.mean(np.log(fe))))
        p2   = float(np.mean(fe <= 2.0)  * 100)
        p3   = float(np.mean(fe <= 3.0)  * 100)
        p10  = float(np.mean(fe <= 10.0) * 100)
        n    = len(y_true_log)
        print(f"\n  {label}  (n={n})")
        print(f"    R^2 (log10)   : {r2:.4f}")
        print(f"    RMSE (log10)  : {rmse:.4f}")
        print(f"    Spearman rho  : {rho:.4f}  (p={rho_p:.3e})")
        print(f"    GMFE          : {gmfe:.2f}x")
        print(f"    Within 2-fold : {p2:.0f} %")
        print(f"    Within 3-fold : {p3:.0f} %")
        print(f"    Within 10-fold: {p10:.0f} %")
        return dict(Set=label, N=n, R2_log=round(r2,4), RMSE_log=round(rmse,4),
                    Spearman=round(rho,4), Spearman_p=round(rho_p,4),
                    GMFE=round(gmfe,2), Pct_2fold=round(p2,1),
                    Pct_3fold=round(p3,1), Pct_10fold=round(p10,1))

    # A) Intern (LOO-CV Ergebnisse)
    loo_csv_path = DATA / "rf_clint_predictions.csv"
    if loo_csv_path.exists():
        loo_data = pd.read_csv(loo_csv_path)
        loo_pos  = loo_data[loo_data["Clint_true"] > 0].copy()
        loo_pos["log10_true"] = np.log10(loo_pos["Clint_true"] + EPSILON)
        all_val_metrics.append(
            _metrics_dict(loo_pos["log10_true"].values,
                          loo_pos["log10_pred"].values,
                          "A) Intern LOO-CV (Piloten)")
        )

    # B) Alle 777 Chemikalien
    all_val_metrics.append(
        _metrics_dict(val777["log10_lit"].values, val777["log10_pred"].values,
                      "B) Alle 777 httk-Chemikalien")
    )

    # C) Extern (Piloten ausgeschlossen)
    ext777 = val777[~val777["in_pilot"]]
    if len(ext777) > 0:
        all_val_metrics.append(
            _metrics_dict(ext777["log10_lit"].values, ext777["log10_pred"].values,
                          "C) Extern (ohne Piloten)")
        )

    # D) Clint > 1 (aktive Clearance)
    val777_gt1 = val777[val777["Clint"] > 1.0]
    if len(val777_gt1) > 0:
        all_val_metrics.append(
            _metrics_dict(val777_gt1["log10_lit"].values, val777_gt1["log10_pred"].values,
                          "D) Clint > 1 uL/min/10^6 (aktive Clearance)")
        )

    metrics_val_df = pd.DataFrame(all_val_metrics)
    metrics_val_df.to_csv(RESULTS / "clint_validation_metrics.csv", index=False)
    print(f"\nMetriken gespeichert: results/clint_validation_metrics.csv")

    # ─ Export vollstaendige Validierungstabelle ────────────────────────────────
    export_cols = ["CAS", "Compound", "Clint", "Clint_pred",
                   "log10_lit", "log10_pred", "fold_error", "in_pilot"]
    export777 = val777[[c for c in export_cols if c in val777.columns]].copy()
    export777 = export777.rename(columns={"Clint": "Clint_literature_uL_min_Mcells"})
    export777 = export777.sort_values("fold_error", ascending=False)
    export777.to_csv(RESULTS / "clint_validation_external.csv", index=False)
    print(f"Vollstaendige Tabelle: results/clint_validation_external.csv  ({len(export777)} Chemikalien)")

    # ─ Scatter-Plots (4 Panels) ────────────────────────────────────────────────
    print("\nErstelle Validierungs-Plots ...")

    def fold_color(fe_arr):
        return ["#2196F3" if f <= 2.0 else "#4CAF50" if f <= 3.0
                else "#FF9800" if f <= 10.0 else "#F44336" for f in fe_arr]

    fig_val = plt.figure(figsize=(20, 5))
    gs_val  = gridspec.GridSpec(1, 4, figure=fig_val, wspace=0.35)

    # Panel A: Intern LOO-CV
    ax = fig_val.add_subplot(gs_val[0])
    if loo_csv_path.exists():
        loo_pos2 = loo_data[loo_data["Clint_true"] > 0].copy()
        loo_pos2["log10_true2"] = np.log10(loo_pos2["Clint_true"] + EPSILON)
        fe_col_int = fold_color(
            10 ** np.abs(loo_pos2["log10_true2"] - loo_pos2["log10_pred"])
        )
        ax.scatter(loo_pos2["log10_true2"], loo_pos2["log10_pred"],
                   c=fe_col_int, s=80, edgecolors="k", linewidths=0.5, zorder=3)
        for _, row_l in loo_pos2.iterrows():
            ax.annotate(str(row_l.get("Compound",""))[:10],
                        (row_l["log10_true2"], row_l["log10_pred"]),
                        fontsize=5, alpha=0.7)
        lims_a = [loo_pos2["log10_true2"].min()-0.5, loo_pos2["log10_true2"].max()+0.5]
        ax.plot(lims_a, lims_a, "k--", lw=1.2)
        ax.fill_between(lims_a, [v-np.log10(3) for v in lims_a],
                        [v+np.log10(3) for v in lims_a], alpha=0.08, color="green")
        m_a = all_val_metrics[0]
        ax.set_title(f"A) Intern LOO-CV (n={m_a['N']})\n"
                     f"R²={m_a['R2_log']:.3f}  GMFE={m_a['GMFE']:.1f}x", fontsize=9)
        ax.set_xlabel("log10(Clint gemessen)", fontsize=8)
        ax.set_ylabel("log10(Clint vorhergesagt)", fontsize=8)
        ax.set_xlim(lims_a); ax.set_ylim(lims_a)
        ax.grid(True, alpha=0.3)

    # Panel B: Alle 777
    ax = fig_val.add_subplot(gs_val[1])
    fe_col_all = fold_color(val777["fold_error"].values)
    pilot_mask = val777["in_pilot"].values
    ax.scatter(val777.loc[~pilot_mask, "log10_lit"], val777.loc[~pilot_mask, "log10_pred"],
               c=[fe_col_all[i] for i in range(len(fe_col_all)) if not pilot_mask[i]],
               s=20, edgecolors="none", alpha=0.6, label="Extern")
    ax.scatter(val777.loc[pilot_mask, "log10_lit"], val777.loc[pilot_mask, "log10_pred"],
               c="gold", s=80, edgecolors="k", linewidths=0.8, zorder=5, label="Pilot (Train)")
    lims_b = [val777["log10_lit"].min()-0.5, val777["log10_lit"].max()+0.5]
    ax.plot(lims_b, lims_b, "k--", lw=1.2)
    ax.fill_between(lims_b, [v-np.log10(3) for v in lims_b],
                    [v+np.log10(3) for v in lims_b], alpha=0.08, color="green")
    m_b = next(m for m in all_val_metrics if "777" in m["Set"])
    ax.set_title(f"B) Alle 777 Chemikalien (n={m_b['N']})\n"
                 f"R²={m_b['R2_log']:.3f}  GMFE={m_b['GMFE']:.1f}x", fontsize=9)
    ax.set_xlabel("log10(Clint Literatur)", fontsize=8)
    ax.set_ylabel("log10(Clint vorhergesagt)", fontsize=8)
    ax.set_xlim(lims_b); ax.set_ylim(lims_b)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)

    # Panel C: Extern
    ax = fig_val.add_subplot(gs_val[2])
    if len(ext777) > 0:
        fe_col_ext = fold_color(ext777["fold_error"].values)
        ax.scatter(ext777["log10_lit"], ext777["log10_pred"],
                   c=fe_col_ext, s=20, edgecolors="none", alpha=0.7)
        ax.plot(lims_b, lims_b, "k--", lw=1.2)
        ax.fill_between(lims_b, [v-np.log10(3) for v in lims_b],
                        [v+np.log10(3) for v in lims_b], alpha=0.08, color="green")
        m_c = next(m for m in all_val_metrics if "Extern" in m["Set"])
        ax.set_title(f"C) Extern (n={m_c['N']})\nR²={m_c['R2_log']:.3f}  GMFE={m_c['GMFE']:.1f}x",
                     fontsize=9)
        ax.set_xlabel("log10(Clint Literatur)", fontsize=8)
        ax.set_ylabel("log10(Clint vorhergesagt)", fontsize=8)
        ax.set_xlim(lims_b); ax.set_ylim(lims_b)
        ax.grid(True, alpha=0.3)
    legend_els = [Patch(facecolor="#2196F3", label="<=2-fold"),
                  Patch(facecolor="#4CAF50", label="<=3-fold"),
                  Patch(facecolor="#FF9800", label="<=10-fold"),
                  Patch(facecolor="#F44336", label=">10-fold")]
    ax.legend(handles=legend_els, fontsize=6, loc="upper left")

    # Panel D: Residual-Histogramm
    ax = fig_val.add_subplot(gs_val[3])
    log_res = (ext777["log10_pred"] - ext777["log10_lit"]).values if len(ext777) > 0 \
              else (val777["log10_pred"] - val777["log10_lit"]).values
    ax.hist(log_res, bins=40, color="#2196F380", edgecolor="white")
    ax.axvline(0, color="red", lw=2, label="kein Bias")
    ax.axvline(np.log10(2),  color="steelblue", lw=1.5, ls="--", label="2-fold")
    ax.axvline(-np.log10(2), color="steelblue", lw=1.5, ls="--")
    ax.axvline(np.log10(3),  color="darkgreen", lw=1.5, ls=":", label="3-fold")
    ax.axvline(-np.log10(3), color="darkgreen", lw=1.5, ls=":")
    bias = 10 ** np.mean(log_res)
    ax.set_xlabel("log10(Vorhergesagt / Literatur)", fontsize=8)
    ax.set_ylabel("Anzahl", fontsize=8)
    ax.set_title(f"D) Residual-Verteilung\nBias (GMR) = {bias:.2f}x", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Clint Validierung: {best_name} (trainiert auf 19 Pilotchemikalien)\n"
        "vs. Wetmore 2012 / httk-Literatur (Human.Clint)",
        fontsize=11, y=1.02,
    )
    plt.savefig(RESULTS / "clint_validation_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: results/clint_validation_scatter.png")

    print("\nDone. Proceed to 03_httk_pbtk_simulation.R")
