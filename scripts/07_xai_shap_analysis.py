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
  A) Random Forest model for Clint prediction (Step 2, simple 3-feature model)
     → Which physicochemical features most drive log10(Clint)?
     → Are there non-linear or interaction effects?
  B) BER prediction from AED/exposure data (Step 5/6)
     → Which input variables explain the BER ranking?
  C) Outlier SHAP – full 9-feature model on all 777 httk chemicals
     → Why does the model fail for Tacrine, Phenylparaben, Acibenzolar?
     → Waterfall plots per outlier + comparison with well-predicted chemicals
     (merged from former 12_shap_outlier_analysis.py)

Plots
~~~~~
  results/shap_rf_summary_bar.png        Global RF SHAP bar chart (Section A)
  results/shap_rf_beeswarm.png           RF SHAP beeswarm plot (Section A)
  results/shap_rf_dependence_*.png       Feature dependence plots (Section A)
  results/shap_ber_beeswarm.png          BER explainability beeswarm (Section B)
  results/shap_outlier_global_bar.png    Global bar + beeswarm (Section C)
  results/shap_outlier_waterfall_*.png   Per-outlier waterfall plots (Section C)
  results/shap_outlier_comparison.png    Outlier vs. well-predicted (Section C)

Data exports
~~~~~~~~~~~~
  results/shap_rf_values.csv             Per-chemical SHAP values (RF, Section A)
  results/shap_ber_values.csv            Per-chemical SHAP values (BER, Section B)
  results/shap_outlier_values.csv        All 777 SHAP values + fold-errors (Section C)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

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
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
except ImportError:
    sys.exit("ERROR: scikit-learn is required.  pip install scikit-learn")

# ── Shared utilities (feature engineering, engineered FEATURE_NAMES) ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import engineer_features, FEATURE_NAMES, EPSILON

# ── Required data ─────────────────────────────────────────────────────────────
PILOT_CSV   = DATA  / "clint_all777_final.csv"
AED_BER_CSV = RESULTS / "aed_ber_full.csv"
FULL_CSV    = DATA  / "all_777_chemicals.csv"

for p in (PILOT_CSV,):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found.  Run steps 01 and 02b first.")


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


# ── C: Outlier SHAP – full 9-feature model on all 777 chemicals ──────────────

def section_c_outlier_shap(pilot: pd.DataFrame) -> None:
    """
    Train the full 9-feature model (same as Step 2) and run SHAP analysis
    on ALL 777 httk chemicals.  Highlights outliers Tacrine, Phenylparaben.
    Merged from former 12_shap_outlier_analysis.py.
    """
    print("\n── C) Outlier SHAP – alle 777 httk-Chemikalien ──")

    if not FULL_CSV.exists():
        print(f"  WARNING: {FULL_CSV} not found – skipping outlier section.")
        return

    # ─ Trainingsdaten ─────────────────────────────────────────────────────────
    df_train = pilot.dropna(subset=["Clint"]).copy()
    df_train["Fup"] = df_train["Fup"].clip(lower=1e-6)
    X_train  = engineer_features(df_train)
    y_train  = np.log10(df_train["Clint"].values + EPSILON)
    print(f"  Training: {len(df_train)} Pilotchemikalien  |  {X_train.shape[1]} Features")

    # ─ Bestes Modell ──────────────────────────────────────────────────────────
    best_name = "GradientBoosting"
    loo_csv   = DATA / "rf_clint_predictions.csv"
    if loo_csv.exists():
        loo_df = pd.read_csv(loo_csv)
        if "model" in loo_df.columns:
            best_name = loo_df["model"].iloc[0]
    print(f"  Modell: {best_name}")

    imputer  = SimpleImputer(strategy="median")
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(imputer.fit_transform(X_train))

    if best_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2,
            subsample=0.8, min_samples_leaf=2, random_state=42,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=1000, max_features=None,
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        )
    model.fit(X_tr_sc, y_train)

    # ─ Alle 777 Chemikalien laden ─────────────────────────────────────────────
    full = pd.read_csv(FULL_CSV)
    full = full.rename(columns={
        "Human.Clint":           "Clint",
        "Human.Funbound.plasma": "Fup",
    })
    for col in ("Clint", "Fup", "MW", "logP"):
        full[col] = pd.to_numeric(full[col], errors="coerce")
    full["Fup"] = full["Fup"].clip(lower=1e-6)

    val = full.dropna(subset=["Clint", "MW", "logP", "Fup"]).copy()
    val = val[val["Clint"] > 0].reset_index(drop=True)

    X_val_raw = engineer_features(val)
    X_val_sc  = scaler.transform(imputer.transform(X_val_raw))

    pred_log = model.predict(X_val_sc)
    val["Clint_pred"] = 10 ** pred_log - EPSILON
    val["log10_lit"]  = np.log10(val["Clint"] + EPSILON)
    val["log10_pred"] = pred_log
    val["fold_error"] = 10 ** np.abs(val["log10_lit"] - val["log10_pred"])

    pilot_cas = set(df_train["CAS"].astype(str).str.strip())
    val["in_pilot"] = val["CAS"].astype(str).str.strip().isin(pilot_cas)
    print(f"  Validierungsset: {len(val)} Chemikalien mit gemessenem Clint")

    # ─ SHAP TreeExplainer ─────────────────────────────────────────────────────
    print("  Berechne SHAP-Werte ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_val_sc)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    base_value    = float(shap_values.base_values[0])
    order         = np.argsort(mean_abs_shap)

    shap_df = pd.DataFrame(shap_values.values,
                           columns=[f"SHAP_{f}" for f in FEATURE_NAMES])
    shap_df.insert(0, "CAS",        val["CAS"].values)
    shap_df.insert(1, "Compound",   val["Compound"].values)
    shap_df.insert(2, "log10_lit",  val["log10_lit"].values)
    shap_df.insert(3, "log10_pred", val["log10_pred"].values)
    shap_df.insert(4, "fold_error", val["fold_error"].values)
    shap_df.insert(5, "in_pilot",   val["in_pilot"].values)
    for f in ["MW", "logP", "Fup", "Clint"]:
        shap_df[f] = val[f].values
    out_csv = RESULTS / "shap_outlier_values.csv"
    shap_df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv.name}")

    # ─ Plot A: Globale Feature-Importance + Beeswarm ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors_bar = plt.cm.RdBu_r(np.linspace(0.15, 0.85, len(FEATURE_NAMES)))
    ax.barh([FEATURE_NAMES[i] for i in order],
            [mean_abs_shap[i] for i in order],
            color=[colors_bar[k] for k in range(len(order))],
            edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Mean |SHAP-Wert|  (Einfluss auf log10 Clint)", fontsize=10)
    ax.set_title(f"Globale Feature-Importance\n{best_name} auf {len(val)} Chemikalien",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    sc = None
    for k, feat_idx in enumerate(order):
        shap_vals = shap_values.values[:, feat_idx]
        feat_vals = X_val_sc[:, feat_idx]
        norm_fv   = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-9)
        y_jitter  = np.random.default_rng(feat_idx).uniform(-0.35, 0.35, len(shap_vals))
        sc = ax.scatter(shap_vals, np.full(len(shap_vals), k) + y_jitter,
                        c=norm_fv, cmap="RdBu_r", s=8, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=9)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("SHAP-Wert", fontsize=10)
    ax.set_title("Beeswarm: Jeder Punkt = 1 Chemikalie", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="Normierter Feature-Wert", fraction=0.03, pad=0.04)

    plt.tight_layout()
    plt.savefig(RESULTS / "shap_outlier_global_bar.png", dpi=150)
    plt.close()
    print("  Saved: results/shap_outlier_global_bar.png")

    # ─ Plot B: Waterfall fuer Top-5-Ausreisser ───────────────────────────────
    TARGET_CHEMS = ["Tacrine", "Phenylparaben", "Acibenzolar"]
    # When training on all measured chemicals (full-dataset model), all val chemicals
    # are "in_pilot". Fall back to using all chemicals for outlier/good selection.
    external_val = val[~val["in_pilot"]]
    if len(external_val) < 5:
        external_val = val

    good = external_val[external_val["fold_error"] <= 1.5].nsmallest(5, "fold_error")
    if len(good) == 0:
        good = external_val.nsmallest(5, "fold_error")

    print(f"\n  Ziel-Ausreisser:")
    for name in TARGET_CHEMS:
        row = val[val["Compound"].str.contains(name, case=False, na=False)]
        if len(row):
            r = row.iloc[0]
            print(f"    {r['Compound'][:35]:<35} Clint_lit={r['Clint']:.0f}  "
                  f"Clint_pred={r['Clint_pred']:.2f}  FE={r['fold_error']:.0f}x")
        else:
            print(f"    {name}: nicht im Validierungsset gefunden")

    top5_out = external_val.nlargest(5, "fold_error")
    for rank, (_, row_v) in enumerate(top5_out.iterrows()):
        chem_idx  = val.index.get_loc(row_v.name)
        shap_chem = shap_values.values[chem_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_idx = np.argsort(np.abs(shap_chem))[::-1]
        colors_wf  = ["#d73027" if v > 0 else "#4575b4" for v in shap_chem[sorted_idx]]
        ax.barh([FEATURE_NAMES[i] for i in sorted_idx], shap_chem[sorted_idx],
                color=colors_wf, edgecolor="k", linewidth=0.4)
        ax.axvline(0, color="k", lw=1.2)

        raw_feats = X_val_raw[chem_idx]
        for bar_idx, feat_idx in enumerate(sorted_idx):
            v = shap_chem[feat_idx]
            ax.text(v + (0.01 if v >= 0 else -0.01), bar_idx,
                    f"  {FEATURE_NAMES[feat_idx]}={raw_feats[feat_idx]:.3g}",
                    va="center", fontsize=8, ha="left" if v >= 0 else "right")

        compound   = str(row_v["Compound"])
        clint_lit  = float(row_v["Clint"])
        clint_pred = float(row_v["Clint_pred"])
        fe         = float(row_v["fold_error"])
        pred_log_v = float(row_v["log10_pred"])

        ax.set_xlabel("SHAP-Wert (Beitrag zur log10 Clint-Vorhersage)", fontsize=10)
        ax.set_title(
            f"SHAP Waterfall: {compound}\n"
            f"Literatur Clint={clint_lit:.0f}  |  Vorhergesagt={clint_pred:.2f}  |  "
            f"Fold-Error={fe:.0f}x\n"
            f"Basiswert={base_value:.2f}  +  SHAP-Summe={shap_chem.sum():.2f}  "
            f"=  Vorhersage={pred_log_v:.2f}",
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.3)
        mw_v   = float(row_v["MW"])
        logp_v = float(row_v["logP"])
        fup_v  = float(row_v["Fup"])
        textbox = (f"MW={mw_v:.0f}  logP={logp_v:.2f}  Fup={fup_v:.4f}\n"
                   f"Warum falsch: Das Modell kennt keine\n"
                   f"reaktiven Gruppen / Enzymspezifitaet")
        ax.text(0.98, 0.02, textbox, transform=ax.transAxes,
                fontsize=8, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="orange", alpha=0.9))

        safe_name = compound.replace(" ", "_").replace("/", "_")[:25]
        out_path  = RESULTS / f"shap_outlier_waterfall_{rank+1}_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path.name}")

    # ─ Plot C: Vergleich Ausreisser vs. gut vorhergesagte Chemikalien ─────────
    COMPARE_CHEMS = {"Ausreisser": [], "Gut": []}
    for name in TARGET_CHEMS:
        row = val[val["Compound"].str.contains(name, case=False, na=False)]
        if len(row):
            idx = val.index.get_loc(row.index[0])
            COMPARE_CHEMS["Ausreisser"].append((row.iloc[0]["Compound"], idx))
    for _, row_g in good.iterrows():
        COMPARE_CHEMS["Gut"].append((row_g["Compound"], val.index.get_loc(row_g.name)))

    all_compare = COMPARE_CHEMS["Ausreisser"] + COMPARE_CHEMS["Gut"]
    if all_compare:
        fig = plt.figure(figsize=(18, 10))
        gs_obj = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
        for k, (chem_name, chem_idx) in enumerate(all_compare[:6]):
            row_data   = val.iloc[chem_idx]
            shap_c     = shap_values.values[chem_idx]
            fe         = float(row_data["fold_error"])
            is_outlier = k < len(COMPARE_CHEMS["Ausreisser"])
            ax = fig.add_subplot(gs_obj[k // 3, k % 3])
            sorted_idx = np.argsort(np.abs(shap_c))[::-1]
            bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_c[sorted_idx]]
            ax.barh([FEATURE_NAMES[i] for i in sorted_idx], shap_c[sorted_idx],
                    color=bar_colors, edgecolor="k", linewidth=0.3)
            ax.axvline(0, color="k", lw=0.8)
            border_color = "#d73027" if is_outlier else "#2e7d32"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
            label = "AUSREISSER" if is_outlier else "GUT VORHERGESAGT"
            ax.set_title(
                f"[{label}] {str(chem_name)[:25]}\n"
                f"Lit={row_data['Clint']:.0f}  Pred={row_data['Clint_pred']:.1f}  FE={fe:.1f}x",
                fontsize=8, color=border_color, fontweight="bold",
            )
            ax.set_xlabel("SHAP", fontsize=8)
            ax.grid(axis="x", alpha=0.2)
            ax.tick_params(axis="both", labelsize=7)

        fig.suptitle(
            "SHAP Vergleich: Ausreisser vs. gut vorhergesagte Chemikalien\n"
            "Rot=Beitrag erhoehend, Blau=Beitrag erniedrigend | Roter Rahmen=Ausreisser",
            fontsize=11, y=1.01,
        )
        plt.savefig(RESULTS / "shap_outlier_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: results/shap_outlier_comparison.png")

    # ─ Mechanistische Erklaerung ──────────────────────────────────────────────
    print(f"\n  Wichtigstes Feature: {FEATURE_NAMES[np.argmax(mean_abs_shap)]}")
    print(f"  SHAP Basiswert:      {base_value:.3f}")
    for name in TARGET_CHEMS:
        row = val[val["Compound"].str.contains(name, case=False, na=False)]
        if not len(row):
            continue
        r = row.iloc[0]
        idx     = val.index.get_loc(row.index[0])
        shap_c  = shap_values.values[idx]
        order_c = np.argsort(np.abs(shap_c))[::-1]
        print(f"\n  {r['Compound']}:")
        print(f"    Gemessen={r['Clint']:.0f}  Vorhergesagt={r['Clint_pred']:.2f}  "
              f"FE={r['fold_error']:.0f}x")
        for feat_idx in order_c[:3]:
            direction = "erhoehend" if shap_c[feat_idx] > 0 else "erniedrigend"
            print(f"    {FEATURE_NAMES[feat_idx]:20s}: SHAP={shap_c[feat_idx]:+.3f}  ({direction})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Step 7 – Explainable AI (SHAP) for Clint and BER")
    print("=" * 65)

    pilot = pd.read_csv(PILOT_CSV)
    if "Clint_source" in pilot.columns:
        pilot = pilot[pilot["Clint_source"] == "httk_measured"].copy().reset_index(drop=True)
    print(f"Loaded {len(pilot)} measured chemicals from {PILOT_CSV.name}")

    section_a_rf_clint(pilot)
    section_b_ber(pilot)
    section_c_outlier_shap(pilot)

    print("\n" + "=" * 65)
    print("XAI outputs:")
    for fname in sorted(RESULTS.glob("shap_*.png")):
        print(f"  {fname.name}")
    for fname in sorted(RESULTS.glob("shap_*.csv")):
        print(f"  {fname.name}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
