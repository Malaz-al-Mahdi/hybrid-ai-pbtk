"""
04b_aed_analysis.py
-------------------
Reads the Monte Carlo AED results from Step 4 (R) and produces
publication-quality visualizations and a summary report:

  1) Paired comparison: native vs. RF-imputed AED per chemical
  2) Population variability fan chart (5th / median / 95th percentile)
  3) Cumulative AED distribution across the pilot set
  4) Summary statistics table exported to CSV & printed

Outputs:
  results/aed_paired_comparison.png
  results/aed_variability_fan.png
  results/aed_cumulative.png
  results/aed_summary_report.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

AED_FILE = RESULTS / "aed_monte_carlo.csv"
SAMPLES_FILE = RESULTS / "aed_mc_samples.csv"
AC50_FILE = DATA / "toxcast_ac50_pilot.csv"

for f in [AED_FILE, SAMPLES_FILE, AC50_FILE]:
    if not f.exists():
        sys.exit(f"ERROR: {f} not found. Run 04_reverse_dosimetry.R first.")

# ---- 1.  Load data --------------------------------------------------------
aed = pd.read_csv(AED_FILE)
samples = pd.read_csv(SAMPLES_FILE)
ac50 = pd.read_csv(AC50_FILE)

print(f"AED summary:  {len(aed)} rows")
print(f"MC samples:   {len(samples)} rows")
print(f"AC50 summary: {len(ac50)} chemicals\n")

# Short compound names for plot labels
aed["short_name"] = aed["Compound"].str[:16]
samples["short_name"] = samples["Compound"].str[:16]

# ---- 2.  Paired AED comparison (native vs RF) -----------------------------

native = aed[aed["Track"] == "httk_native"].set_index("CAS")
rf = aed[aed["Track"] == "rf_imputed"].set_index("CAS")
paired = native.join(rf, lsuffix="_nat", rsuffix="_rf", how="inner")

if len(paired) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))

    x = paired["AED_median_nat"].values
    y = paired["AED_median_rf"].values
    labels = paired["short_name_nat"].values

    ax.scatter(x, y, s=70, edgecolors="steelblue", facecolors="lightblue",
               zorder=3)

    # Error bars: 5th-95th percentile
    xerr = np.array([
        x - paired["AED_5pct_nat"].values,
        paired["AED_95pct_nat"].values - x
    ])
    yerr = np.array([
        y - paired["AED_5pct_rf"].values,
        paired["AED_95pct_rf"].values - y
    ])
    xerr = np.clip(xerr, 0, None)
    yerr = np.clip(yerr, 0, None)

    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt="none", ecolor="gray", alpha=0.4, zorder=2)

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=6.5, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")

    lim_min = min(np.min(x), np.min(y)) * 0.5
    lim_max = max(np.max(x), np.max(y)) * 2.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4,
            label="1:1 line")
    # 3-fold envelope
    ax.fill_between([lim_min, lim_max],
                    [lim_min / 3, lim_max / 3],
                    [lim_min * 3, lim_max * 3],
                    alpha=0.07, color="green", label="3-fold envelope")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("AED median - httk native (mg/kg/day)")
    ax.set_ylabel("AED median - RF imputed (mg/kg/day)")
    ax.set_title("Reverse Dosimetry: httk native vs RF-imputed AED")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "aed_paired_comparison.png", dpi=150)
    print("Saved results/aed_paired_comparison.png")
    plt.close()

# ---- 3.  Population variability fan chart ----------------------------------

native_aed = aed[aed["Track"] == "httk_native"].sort_values("AED_median")

if len(native_aed) > 0:
    fig, ax = plt.subplots(figsize=(max(10, len(native_aed) * 0.6), 6))

    x_pos = np.arange(len(native_aed))
    medians = native_aed["AED_median"].values
    lo = native_aed["AED_5pct"].values
    hi = native_aed["AED_95pct"].values
    names = native_aed["short_name"].values

    ax.fill_between(x_pos, lo, hi, alpha=0.25, color="steelblue",
                    label="5th-95th percentile")
    ax.plot(x_pos, medians, "o-", color="steelblue", markersize=6,
            label="Median AED")

    # Overlay RF track if available
    rf_aed = aed[aed["Track"] == "rf_imputed"]
    if len(rf_aed) > 0:
        rf_ordered = rf_aed.set_index("CAS").loc[native_aed["CAS"].values]
        rf_ordered = rf_ordered.dropna(subset=["AED_median"])
        if len(rf_ordered) > 0:
            rf_x = [list(native_aed["CAS"]).index(c) for c in rf_ordered.index
                    if c in list(native_aed["CAS"].values)]
            ax.plot(rf_x, rf_ordered["AED_median"].values, "s--",
                    color="tomato", markersize=5, label="RF-imputed median",
                    alpha=0.8)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("AED (mg/kg/day)")
    ax.set_title("Population Variability in AED (Monte Carlo, n=1000)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "aed_variability_fan.png", dpi=150)
    print("Saved results/aed_variability_fan.png")
    plt.close()

# ---- 4.  Cumulative AED distribution across all chemicals -----------------

native_samples = samples[samples["Track"] == "httk_native"]

if len(native_samples) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))

    all_aed = native_samples["AED"].values
    all_aed_sorted = np.sort(all_aed[all_aed > 0])
    cdf = np.arange(1, len(all_aed_sorted) + 1) / len(all_aed_sorted)

    ax.plot(all_aed_sorted, cdf, color="steelblue", lw=1.5,
            label="httk native (all chemicals pooled)")

    rf_samples = samples[samples["Track"] == "rf_imputed"]
    if len(rf_samples) > 0:
        rf_aed = rf_samples["AED"].values
        rf_aed_sorted = np.sort(rf_aed[rf_aed > 0])
        cdf_rf = np.arange(1, len(rf_aed_sorted) + 1) / len(rf_aed_sorted)
        ax.plot(rf_aed_sorted, cdf_rf, color="tomato", lw=1.5, ls="--",
                label="RF-imputed (all chemicals pooled)")

    ax.set_xscale("log")
    ax.set_xlabel("AED (mg/kg/day)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative AED Distribution (Monte Carlo Samples)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "aed_cumulative.png", dpi=150)
    print("Saved results/aed_cumulative.png")
    plt.close()

# ---- 5.  Summary report ---------------------------------------------------

report = aed.pivot_table(
    index=["CAS", "Compound"],
    columns="Track",
    values=["AED_median", "AED_5pct", "AED_95pct"],
    aggfunc="first"
)
report.columns = ["_".join(col).strip() for col in report.columns.values]
report = report.reset_index()

# Add AC50 info
report = report.merge(
    ac50[["CAS", "n_active_assays", "AC50_5pct_uM", "AC50_median_uM"]],
    on="CAS", how="left"
)

# Fold-change between native and RF
nat_col = [c for c in report.columns if "AED_median_httk_native" in c]
rf_col  = [c for c in report.columns if "AED_median_rf_imputed" in c]
if nat_col and rf_col:
    report["FC_AED_rf_vs_native"] = (
        report[rf_col[0]] / report[nat_col[0]]
    ).round(3)

report.to_csv(RESULTS / "aed_summary_report.csv", index=False)
print(f"\nSaved results/aed_summary_report.csv ({len(report)} chemicals)")

print("\n" + "=" * 70)
print("AED SUMMARY REPORT")
print("=" * 70)
print(report.to_string(index=False))
print("\nDone.")
