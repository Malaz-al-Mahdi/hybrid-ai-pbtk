"""
02_rf_predict_clint.py
----------------------
Trains a Random Forest regressor to predict hepatic intrinsic clearance (Clint)
from physicochemical descriptors (MW, logP, Fup, Rblood2plasma).

Evaluation strategy:
  - Leave-One-Out Cross-Validation (LOO-CV) on the 20 pilot chemicals
  - Reports per-chemical absolute error, overall RMSE, R^2, and
    Spearman rank correlation (appropriate for skewed TK data)

Outputs:
  data/rf_clint_predictions.csv   – LOO-CV predictions vs. true values
  results/rf_loo_cv_metrics.txt   – summary statistics
  results/rf_loo_cv_scatter.png   – observed vs. predicted scatter plot
  data/pilot_chemicals_imputed.csv – full table with RF-imputed Clint
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

FULL_CSV = DATA / "pilot_chemicals_full.csv"

if not FULL_CSV.exists():
    sys.exit(
        f"ERROR: {FULL_CSV} not found. Run 01_extract_httk_data.R first."
    )

# ---- 1. Load data ----------------------------------------------------------
df = pd.read_csv(FULL_CSV)
print(f"Loaded {len(df)} chemicals from {FULL_CSV.name}")
print(df[["CAS", "Compound", "Clint"]].to_string(index=False))
print()

FEATURES = ["MW", "logP", "Fup", "Rblood2plasma"]
TARGET = "Clint"

# Keep all rows with a known target. Missing feature values are imputed.
df_clean = df.dropna(subset=[TARGET]).copy()
print(f"Rows with observed Clint: {len(df_clean)}")
print("Missing values per feature:")
print(df_clean[FEATURES].isna().sum().to_string())

if len(df_clean) < 5:
    sys.exit("Too few rows with observed Clint for meaningful RF training.")

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

# Log-transform Clint (often log-normally distributed)
# Add small epsilon to handle zero clearance values
EPSILON = 1e-3
y_log = np.log10(y + EPSILON)

# ---- 2. LOO-CV with Random Forest -----------------------------------------
loo = LeaveOneOut()
y_pred_log = np.full_like(y_log, np.nan)
feature_importance_acc = np.zeros(len(FEATURES))

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y_log[train_idx]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred_log[test_idx] = model.predict(X_test)
    feature_importance_acc += model.named_steps["rf"].feature_importances_

# Back-transform to original scale
y_pred = 10 ** y_pred_log - EPSILON
y_pred = np.clip(y_pred, 0, None)

feature_importance_avg = feature_importance_acc / len(df_clean)

# ---- 3. Metrics ------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
rho, rho_p = spearmanr(y, y_pred)

# Per-chemical results
results_df = df_clean[["CAS", "Compound"]].copy()
results_df["Clint_true"] = y
results_df["Clint_pred_RF"] = np.round(y_pred, 4)
results_df["abs_error"] = np.round(np.abs(y - y_pred), 4)
results_df["log10_true"] = np.round(y_log, 4)
results_df["log10_pred"] = np.round(y_pred_log, 4)

results_df.to_csv(DATA / "rf_clint_predictions.csv", index=False)
print("\nLOO-CV Results:")
print(results_df.to_string(index=False))

metrics_text = (
    f"Random Forest LOO-CV  (n = {len(df_clean)} chemicals)\n"
    f"{'='*50}\n"
    f"RMSE (original scale) : {rmse:.4f}  uL/min/10^6 cells\n"
    f"R^2  (original scale) : {r2:.4f}\n"
    f"Spearman rho          : {rho:.4f}  (p = {rho_p:.4e})\n\n"
    f"Feature importances (mean across LOO folds):\n"
)
for feat, imp in zip(FEATURES, feature_importance_avg):
    metrics_text += f"  {feat:20s}: {imp:.4f}\n"

with open(RESULTS / "rf_loo_cv_metrics.txt", "w") as f:
    f.write(metrics_text)

print(f"\n{metrics_text}")

# ---- 4. Scatter plot -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 4a. Original scale
ax = axes[0]
ax.scatter(y, y_pred, edgecolors="steelblue", facecolors="lightblue", s=60)
lim = max(y.max(), y_pred.max()) * 1.1
ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal")
ax.set_xlabel("Observed Clint (uL/min/10^6 cells)")
ax.set_ylabel("Predicted Clint (RF LOO-CV)")
ax.set_title(f"Original scale  |  RMSE={rmse:.2f}  R²={r2:.3f}")
ax.legend()

# 4b. Log10 scale
ax = axes[1]
ax.scatter(y_log, y_pred_log, edgecolors="darkgreen", facecolors="lightgreen", s=60)
lim_log = [min(y_log.min(), y_pred_log.min()) - 0.3,
           max(y_log.max(), y_pred_log.max()) + 0.3]
ax.plot(lim_log, lim_log, "k--", alpha=0.5, label="ideal")
ax.set_xlabel("Observed log10(Clint)")
ax.set_ylabel("Predicted log10(Clint)")
ax.set_title(f"Log10 scale  |  Spearman ρ={rho:.3f}")
ax.legend()

for i, row in results_df.iterrows():
    axes[1].annotate(
        row["Compound"][:10],
        (row["log10_true"], row["log10_pred"]),
        fontsize=6, alpha=0.7
    )

plt.tight_layout()
plt.savefig(RESULTS / "rf_loo_cv_scatter.png", dpi=150)
print(f"\nPlot saved to results/rf_loo_cv_scatter.png")

# ---- 5. Export imputed dataset ----------------------------------------------
#
# For the pilot we retrain on ALL available data, then "impute" Clint for
# every chemical (including those that already had it -- for consistency).
# In a real pipeline, this step would fill gaps for chemicals without Clint.

rf_final = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
rf_final.fit(X, y_log)

# Predict for ALL chemicals in the original df (including incomplete ones)
df_all = pd.read_csv(FULL_CSV)
X_all = df_all[FEATURES].values

# The imputer enables predictions even when some descriptors are missing.
pred_log = rf_final.predict(X_all)
df_all["Clint_RF"] = np.round(10 ** pred_log - EPSILON, 4)

# Clint_final: use original if available, otherwise RF prediction
df_all["Clint_final"] = df_all[TARGET]
na_mask = df_all["Clint_final"].isna()
df_all.loc[na_mask, "Clint_final"] = df_all.loc[na_mask, "Clint_RF"]
df_all["Clint_source"] = "httk"
df_all.loc[na_mask, "Clint_source"] = "RF_predicted"

df_all.to_csv(DATA / "pilot_chemicals_imputed.csv", index=False)
print(f"Saved  data/pilot_chemicals_imputed.csv  ({len(df_all)} chemicals)")
print("Done. Proceed to 03_httk_pbtk_simulation.R")
