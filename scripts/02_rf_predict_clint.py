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

Outputs:
  data/rf_clint_predictions.csv    - LOO-CV predictions vs. true values
  results/rf_loo_cv_metrics.txt    - summary statistics
  results/rf_loo_cv_scatter.png    - observed vs. predicted scatter plot
  data/pilot_chemicals_imputed.csv - full table with RF-imputed Clint
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths ------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"
FULL_CSV = DATA / "pilot_chemicals_full.csv"

if not FULL_CSV.exists():
    sys.exit(f"ERROR: {FULL_CSV} not found. Run 01_extract_httk_data.R first.")

# ---- 1. Load data -----------------------------------------------------------
df = pd.read_csv(FULL_CSV)
print(f"Loaded {len(df)} chemicals from {FULL_CSV.name}")
print(df[["CAS", "Compound", "Clint"]].to_string(index=False))
print()

RAW_FEATURES = ["MW", "logP", "Fup"]
TARGET       = "Clint"
EPSILON      = 1e-3


# ---- 2. Feature engineering -------------------------------------------------
# Raw MW/logP/Fup are not linearly related to Clint.
# Biological rationale for each transform:
#   log10(MW)     -- size effect scales logarithmically
#   log10(Fup)    -- plasma binding spans orders of magnitude
#   logP^2        -- optimal lipophilicity (bell-shaped clearance curves)
#   sqrt(Fup)     -- alternative binding scale
#   MW x logP     -- large lipophilic molecules tend to have low Clint
#   logP x Fup    -- lipophilicity-binding interaction drives hepatic uptake

def engineer_features(df_in: pd.DataFrame) -> np.ndarray:
    mw   = np.clip(df_in["MW"].values.astype(float),   1.0,  None)
    logp = df_in["logP"].values.astype(float)
    fup  = np.clip(df_in["Fup"].values.astype(float),  1e-6, 1.0)
    return np.column_stack([
        np.log10(mw),            # log10(MW)
        logp,                    # logP
        logp ** 2,               # logP^2
        np.log10(fup + 1e-6),   # log10(Fup)
        np.sqrt(fup),            # sqrt(Fup)
        mw * logp,               # MW x logP
        mw * fup,                # MW x Fup
        logp * fup,              # logP x Fup
        mw,                      # MW (raw)
    ])

FEATURE_NAMES = [
    "log10_MW", "logP", "logP^2", "log10_Fup", "sqrt_Fup",
    "MW_x_logP", "MW_x_Fup", "logP_x_Fup", "MW",
]

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
print("Done. Proceed to 03_httk_pbtk_simulation.R")
