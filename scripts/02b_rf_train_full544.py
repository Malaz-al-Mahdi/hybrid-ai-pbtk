import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils import (
    ROOT, DATA, RESULTS, ALL_777_CSV,
    EPSILON, engineer_features, FEATURE_NAMES, compute_metrics,
)

if not ALL_777_CSV.exists():
    sys.exit(f"ERROR: {ALL_777_CSV} nicht gefunden. Erst 01_extract_httk_data.R ausfuehren.")

print("=" * 65)
print("Step 2b - RF/GB auf 544 httk-Chemikalien (500 Train / 44 Test)")
print("=" * 65)

full = pd.read_csv(ALL_777_CSV)
full = full.rename(columns={
    "Human.Clint":           "Clint",
    "Human.Funbound.plasma": "Fup",
    "Human.Rblood2plasma":   "Rblood2plasma",
})
for col in ("Clint", "Fup", "MW", "logP"):
    full[col] = pd.to_numeric(full[col], errors="coerce")
full["Fup"] = full["Fup"].clip(lower=1e-6)
full["CAS"] = full["CAS"].astype(str).str.strip()

measured = full[full["Clint"] > 0].copy().reset_index(drop=True)
no_clint = full[full["Clint"] <= 0].copy().reset_index(drop=True)

print(f"\nGesamt Chemikalien      : {len(full)}")
print(f"Mit gemessenem Clint    : {len(measured)}")
print(f"Ohne gemessenen Clint   : {len(no_clint)}")

if len(measured) < 50:
    sys.exit("Zu wenige Chemikalien mit gemessenem Clint.")

measured["log10_Clint"] = np.log10(measured["Clint"] + EPSILON)
measured["strat_bin"]   = pd.cut(measured["log10_Clint"], bins=6, labels=False)

TEST_SIZE = 44
train_df, test_df = train_test_split(
    measured,
    test_size=TEST_SIZE,
    stratify=measured["strat_bin"],
    random_state=42,
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"\nSplit:")
print(f"  Training : {len(train_df)} Chemikalien")
print(f"  Test     : {len(test_df)} Chemikalien")
print(f"  Clint-Bereich Training : {train_df['Clint'].min():.2f} - {train_df['Clint'].max():.2f}")
print(f"  Clint-Bereich Test     : {test_df['Clint'].min():.2f} - {test_df['Clint'].max():.2f}")

X_train   = engineer_features(train_df)
X_test    = engineer_features(test_df)
X_noClint = engineer_features(no_clint)

y_train     = train_df["Clint"].values
y_test      = test_df["Clint"].values
y_train_log = np.log10(y_train + EPSILON)
y_test_log  = np.log10(y_test  + EPSILON)

def make_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=2,
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
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        )),
    ])

print("\nTrainiere RF ...")
rf = make_rf()
rf.fit(X_train, y_train_log)

print("Trainiere GB ...")
gb = make_gb()
gb.fit(X_train, y_train_log)

rf_train_pred = rf.predict(X_train)
gb_train_pred = gb.predict(X_train)
r2_rf_train = r2_score(y_train_log, rf_train_pred)
r2_gb_train = r2_score(y_train_log, gb_train_pred)
print(f"\nTrain-Set R^2:  RF={r2_rf_train:.4f}  GB={r2_gb_train:.4f}")

rf_test_pred = rf.predict(X_test)
gb_test_pred = gb.predict(X_test)

r2_rf_test  = r2_score(y_test_log, rf_test_pred)
r2_gb_test  = r2_score(y_test_log, gb_test_pred)

if r2_gb_test >= r2_rf_test:
    best_name     = "GradientBoosting"
    best_model    = gb
    y_test_pred   = gb_test_pred
else:
    best_name     = "RandomForest"
    best_model    = rf
    y_test_pred   = rf_test_pred

print(f"\nTest-Set R^2:   RF={r2_rf_test:.4f}  GB={r2_gb_test:.4f}")
print(f"=> Bestes Modell: {best_name}")

def _metrics(y_true_log, y_pred_log, label):
    r2   = r2_score(y_true_log, y_pred_log)
    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    rho, rho_p = spearmanr(y_true_log, y_pred_log)
    fe   = 10 ** np.abs(y_true_log - y_pred_log)
    gmfe = float(np.exp(np.mean(np.log(fe))))
    p2   = float(np.mean(fe <= 2.0) * 100)
    p3   = float(np.mean(fe <= 3.0) * 100)
    p10  = float(np.mean(fe <= 10.0) * 100)
    n    = len(y_true_log)
    print(f"\n--- {label} (n={n}) ---")
    print(f"  R^2 (log10)    : {r2:.4f}")
    print(f"  RMSE (log10)   : {rmse:.4f}")
    print(f"  Spearman rho   : {rho:.4f}  (p={rho_p:.3e})")
    print(f"  GMFE           : {gmfe:.2f}x")
    print(f"  Within 2-fold  : {p2:.0f} %")
    print(f"  Within 3-fold  : {p3:.0f} %")
    print(f"  Within 10-fold : {p10:.0f} %")
    return dict(Set=label, N=n, R2_log=round(r2,4), RMSE_log=round(rmse,4),
                Spearman=round(rho,4), GMFE=round(gmfe,2),
                Pct_2fold=round(p2,1), Pct_3fold=round(p3,1), Pct_10fold=round(p10,1))

metrics_list = []
metrics_list.append(_metrics(y_train_log, rf_train_pred, f"Train-Set RF   (n={len(train_df)})"))
metrics_list.append(_metrics(y_train_log, gb_train_pred, f"Train-Set GB   (n={len(train_df)})"))
metrics_list.append(_metrics(y_test_log,  rf_test_pred,  "Test-Set RF    (44 Hold-out)"))
metrics_list.append(_metrics(y_test_log,  gb_test_pred,  "Test-Set GB    (44 Hold-out)"))

test_df = test_df.copy()
test_df["log10_true"]    = np.round(y_test_log, 4)
test_df["log10_pred_RF"] = np.round(rf_test_pred, 4)
test_df["log10_pred_GB"] = np.round(gb_test_pred, 4)
test_df["fold_error_RF"] = np.round(10 ** np.abs(y_test_log - rf_test_pred), 3)
test_df["fold_error_GB"] = np.round(10 ** np.abs(y_test_log - gb_test_pred), 3)

print("\nTest-Set Chemikalien (sortiert nach fold_error_RF):")
show_cols = ["CAS", "Compound", "Clint", "log10_true",
             "log10_pred_RF", "log10_pred_GB", "fold_error_RF"]
print(test_df[show_cols].sort_values("fold_error_RF", ascending=False).to_string(index=False))

old_metrics_path = RESULTS / "rf_loo_cv_metrics.txt"
old_r2_str = ""
if old_metrics_path.exists():
    with open(old_metrics_path) as f:
        for line in f:
            if "R^2  log10" in line:
                old_r2_str = line.strip()
                break

metrics_text = (
    f"RF/GB Clint-Vorhersage -- Trainiert auf {len(train_df)} httk-Chemikalien\n"
    f"Bestes Modell: {best_name}\n"
    f"{'='*60}\n\n"
    f"SPLIT\n"
    f"  Training : {len(train_df)} Chemikalien (gemessener Clint > 0)\n"
    f"  Test     : {len(test_df)} Chemikalien (Hold-out)\n"
    f"  Ohne Clint (Vorhersage): {len(no_clint)} Chemikalien\n\n"
)
for m in metrics_list:
    metrics_text += (
        f"[{m['Set']}]\n"
        f"  R^2 log10  : {m['R2_log']:.4f}\n"
        f"  RMSE log10 : {m['RMSE_log']:.4f}\n"
        f"  Spearman   : {m['Spearman']:.4f}\n"
        f"  GMFE       : {m['GMFE']:.2f}x\n"
        f"  <=2-fold   : {m['Pct_2fold']:.0f}%\n"
        f"  <=3-fold   : {m['Pct_3fold']:.0f}%\n\n"
    )
if old_r2_str:
    metrics_text += f"VERGLEICH 19-Piloten-Modell:\n  {old_r2_str}\n"

with open(RESULTS / "full544_metrics.txt", "w") as f:
    f.write(metrics_text)
print(f"\nMetriken -> results/full544_metrics.txt")
print(metrics_text)

def fold_color(fe_arr):
    return ["#2196F3" if f <= 2.0 else "#4CAF50" if f <= 3.0
            else "#FF9800" if f <= 10.0 else "#F44336" for f in fe_arr]

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

ax = axes[0]
fe_rf  = test_df["fold_error_RF"].values
colors = fold_color(fe_rf)
ax.scatter(test_df["log10_true"], test_df["log10_pred_RF"],
           c=colors, s=70, edgecolors="k", linewidths=0.5, zorder=3)
lims = [test_df["log10_true"].min()-0.5, test_df["log10_true"].max()+0.5]
ax.plot(lims, lims, "k--", lw=1.2, label="ideal")
ax.fill_between(lims, [v-np.log10(3) for v in lims],
                [v+np.log10(3) for v in lims],
                alpha=0.08, color="green", label="3-fold Band")
for _, row in test_df.iterrows():
    ax.annotate(str(row.get("Compound",""))[:12],
                (row["log10_true"], row["log10_pred_RF"]),
                fontsize=5, alpha=0.7)
m_test_rf = next(m for m in metrics_list if "Test-Set RF" in m["Set"])
ax.set_title(f"A) Hold-out Test-Set RF (n=44)\n"
             f"R^2={m_test_rf['R2_log']:.3f}  GMFE={m_test_rf['GMFE']:.2f}x  "
             f"<=3-fold={m_test_rf['Pct_3fold']:.0f}%", fontsize=9)
ax.set_xlabel("log10(Clint gemessen)"); ax.set_ylabel("log10(Clint RF)"); ax.grid(alpha=0.3)
legend_els = [Patch(facecolor="#2196F3", label="<=2-fold"),
              Patch(facecolor="#4CAF50", label="<=3-fold"),
              Patch(facecolor="#FF9800", label="<=10-fold"),
              Patch(facecolor="#F44336", label=">10-fold")]
ax.legend(handles=legend_els, fontsize=7, loc="upper left")

ax = axes[1]
ax.scatter(y_test_log, rf_test_pred,
           label=f"RF   R^2={r2_rf_test:.3f}",
           edgecolors="steelblue", facecolors="lightblue", s=60, alpha=0.8)
ax.scatter(y_test_log, gb_test_pred,
           label=f"GB   R^2={r2_gb_test:.3f}",
           edgecolors="tomato", facecolors="lightsalmon", s=60, alpha=0.8, marker="^")
ax.plot(lims, lims, "k--", lw=1.2)
ax.set_title("B) RF vs. GB auf Test-Set (44)", fontsize=9)
ax.set_xlabel("log10(Clint gemessen)"); ax.set_ylabel("log10(Clint vorhergesagt)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
residuals = y_test_pred - y_test_log
ax.hist(residuals, bins=20, color="#2196F380", edgecolor="white")
ax.axvline(0, color="red", lw=2, label="kein Bias")
ax.axvline( np.log10(2), color="steelblue", lw=1.5, ls="--", label="2-fold")
ax.axvline(-np.log10(2), color="steelblue", lw=1.5, ls="--")
ax.axvline( np.log10(3), color="darkgreen", lw=1.5, ls=":", label="3-fold")
ax.axvline(-np.log10(3), color="darkgreen", lw=1.5, ls=":")
bias = 10 ** np.mean(residuals)
ax.set_xlabel("log10(Pred / Gemessen)"); ax.set_ylabel("Anzahl")
ax.set_title(f"C) Residuen ({best_name})\nBias (GMR) = {bias:.2f}x", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle(
    f"Clint-Vorhersage: {best_name}  |  Training auf {len(train_df)} httk-Chemikalien  |  Test: 44 Hold-out",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig(RESULTS / "full544_test_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/full544_test_scatter.png")

old_pred_path = DATA / "rf_clint_predictions.csv"
if old_pred_path.exists():
    old_pred = pd.read_csv(old_pred_path)
    overlap = test_df.merge(
        old_pred[["CAS","log10_pred","log10_true"]].rename(
            columns={"log10_pred": "old_log10_pred", "log10_true": "old_log10_true"}),
        on="CAS", how="inner",
    )
    if len(overlap) > 0:
        print(f"\nUeberschneidung Test-Set / 19-Piloten: {len(overlap)} Chemikalien")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(overlap["old_log10_true"], overlap["old_log10_pred"],
                    label="19-Piloten-Modell (LOO-CV)", s=80,
                    edgecolors="steelblue", facecolors="lightblue", zorder=3)
        ax2.scatter(overlap["log10_true"], overlap["log10_pred_RF"],
                    label=f"544er-Modell RF (Test-Set)", s=80,
                    edgecolors="tomato", facecolors="lightsalmon", marker="^", zorder=4)
        lims2 = [min(overlap["old_log10_true"].min(), overlap["log10_true"].min()) - 0.3,
                 max(overlap["old_log10_true"].max(), overlap["log10_true"].max()) + 0.3]
        ax2.plot(lims2, lims2, "k--", lw=1.2)
        ax2.set_xlabel("log10(Clint gemessen)"); ax2.set_ylabel("log10(Clint vorhergesagt)")
        ax2.set_title(f"Modellvergleich auf gemeinsamen Chemikalien (n={len(overlap)})")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS / "full544_comparison_vs_19.png", dpi=150)
        plt.close()
        print("Saved: results/full544_comparison_vs_19.png")

print("\n" + "=" * 65)
print("Finales Modell: Retrain auf allen 544 Chemikalien")
print("=" * 65)

X_all544   = engineer_features(measured)
y_all544   = measured["Clint"].values
y_all544_log = np.log10(y_all544 + EPSILON)

final_model = make_gb() if best_name == "GradientBoosting" else make_rf()
final_model.fit(X_all544, y_all544_log)
print(f"Modell auf {len(measured)} Chemikalien trainiert.")

pred_log_233 = final_model.predict(X_noClint)
no_clint = no_clint.copy()
no_clint["Clint_RF_pred"]     = np.round(10 ** pred_log_233 - EPSILON, 4).clip(0)
no_clint["log10_Clint_pred"]  = np.round(pred_log_233, 4)
no_clint["Clint_source"]      = "RF_predicted_544model"

print(f"\nVorhersage fuer {len(no_clint)} Chemikalien ohne Clint:")
print(f"  Vorhersage-Bereich: {no_clint['Clint_RF_pred'].min():.2f} - "
      f"{no_clint['Clint_RF_pred'].max():.2f} uL/min/10^6")
print(f"  Median: {no_clint['Clint_RF_pred'].median():.2f}")

out_233 = no_clint[["CAS", "Compound", "MW", "logP", "Fup",
                     "Clint_RF_pred", "log10_Clint_pred", "Clint_source"]].copy()
out_233 = out_233.sort_values("Clint_RF_pred", ascending=False)
out_233.to_csv(DATA / "clint_predicted_233.csv", index=False)
print(f"\nGespeichert: data/clint_predicted_233.csv  ({len(out_233)} Chemikalien)")

measured_out = measured[["CAS", "Compound", "MW", "logP", "Fup", "Clint"]].copy()
measured_out["Clint_final"]  = measured_out["Clint"]
measured_out["Clint_source"] = "httk_measured"

no_clint_out = no_clint[["CAS", "Compound", "MW", "logP", "Fup"]].copy()
no_clint_out["Clint"]        = np.nan
no_clint_out["Clint_final"]  = no_clint["Clint_RF_pred"].values
no_clint_out["Clint_source"] = "RF_predicted_544model"

all777_final = pd.concat([measured_out, no_clint_out], ignore_index=True)
all777_final.to_csv(DATA / "clint_all777_final.csv", index=False)
print(f"Gespeichert: data/clint_all777_final.csv   ({len(all777_final)} Chemikalien)")

fig3, ax3 = plt.subplots(figsize=(10, 5))
bins = np.linspace(-3, 4, 40)
ax3.hist(np.log10(measured["Clint"] + EPSILON), bins=bins,
         color="#2196F3", alpha=0.7, label=f"Gemessen (n={len(measured)})", edgecolor="white")
ax3.hist(no_clint["log10_Clint_pred"], bins=bins,
         color="#FF9800", alpha=0.6, label=f"RF-Vorhersage (n={len(no_clint)})", edgecolor="white")
ax3.axvline(np.log10(1),  color="gray",  lw=1.5, ls="--", alpha=0.7, label="Clint=1")
ax3.axvline(np.log10(10), color="gray",  lw=1.0, ls=":", alpha=0.7)
ax3.set_xlabel("log10(Clint [uL/min/10^6])"); ax3.set_ylabel("Anzahl Chemikalien")
ax3.set_title(f"Clint-Verteilung: Gemessen vs. RF-Vorhersage (544er-Modell)\n"
              f"Gesamt: {len(all777_final)} Chemikalien", fontsize=11)
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "full544_clint_distribution.png", dpi=150)
plt.close()
print("Saved: results/full544_clint_distribution.png")

print("\n" + "=" * 65)
print("ZUSAMMENFASSUNG")
print("=" * 65)
print(f"\n  Training (544er-Modell) : {len(train_df)} Chemikalien")
print(f"  Test (Hold-out)         : {len(test_df)} Chemikalien")
print(f"  Vorhersage (kein Clint) : {len(no_clint)} Chemikalien")
print(f"\n  Bestes Modell : {best_name}")
abbrev = "RF" if best_name == "RandomForest" else "GB"
m_test = next(m for m in metrics_list if "Test-Set" in m["Set"] and abbrev in m["Set"])
print(f"  Test-Set R^2  : {m_test['R2_log']:.4f}")
print(f"  Test-Set GMFE : {m_test['GMFE']:.2f}x")
print(f"  <=3-fold      : {m_test['Pct_3fold']:.0f}%")
print(f"\n  Ausgaben:")
print(f"    results/full544_metrics.txt")
print(f"    results/full544_test_scatter.png")
print(f"    results/full544_clint_distribution.png")
print(f"    data/clint_predicted_233.csv")
print(f"    data/clint_all777_final.csv")
print("\nDone.")
