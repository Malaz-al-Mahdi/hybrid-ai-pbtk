###############################################################################
# 04_reverse_dosimetry.R
# ---------------------------------------------------------------------------
# Reverse Toxicokinetics: Converts in-vitro ToxCast AC50 values to
# Administered Equivalent Doses (AED) using httk's Monte Carlo population
# variability framework.
#
# For each pilot chemical the script:
#   1) Retrieves active ToxCast assay hits (AC50 / modl_acc)
#   2) Calculates the 5th-percentile AC50 as a conservative bioactivity
#      concentration (most sensitive assay endpoint)
#   3) Runs calc_mc_oral_equiv() with 1000 MC samples to propagate
#      population variability in Clint, Fup, body weight, etc.
#   4) Reports AED quantiles (median, 5th, 95th percentile)
#
# Two tracks are computed for every chemical:
#   A) httk_native  – built-in httk parameters
#   B) rf_imputed   – Clint overridden with RF prediction from Step 2
#
# Outputs:
#   data/toxcast_ac50_pilot.csv        – ToxCast hit summary per chemical
#   results/aed_monte_carlo.csv        – AED quantiles per chemical & track
#   results/aed_mc_samples.csv         – full MC sample distributions
#   results/aed_distributions.png      – box/violin plots
###############################################################################

# ---- Library bootstrap -----------------------------------------------------
r_ver <- paste(
  R.version$major,
  strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1],
  sep = "."
)
r_user_lib <- Sys.getenv("R_LIBS_USER", unset = "")
if (!nzchar(r_user_lib)) {
  r_user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", r_ver)
}
dir.create(r_user_lib, recursive = TRUE, showWarnings = FALSE)
if (dir.exists(r_user_lib)) {
  .libPaths(unique(c(r_user_lib, .libPaths())))
}

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", lib = r_user_lib)
  }
}
ensure_pkg("stringr")
ensure_pkg("httk")

library(httk, lib.loc = .libPaths())

# ---- Resolve project paths -------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
project_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
data_dir    <- file.path(project_dir, "data")
results_dir <- file.path(project_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 0.  Load imputed chemical table --------------------------------------
imputed_file <- file.path(data_dir, "pilot_chemicals_imputed.csv")
if (!file.exists(imputed_file)) {
  stop("pilot_chemicals_imputed.csv not found. Run steps 01+02 first.")
}
imputed <- read.csv(imputed_file, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d pilot chemicals.\n\n", nrow(imputed)))

# ---- 1.  Map CAS -> DTXSID & extract ToxCast AC50 data --------------------

chem_table <- httk::chem.physical_and_invitro.data
tc_data    <- httk::example.toxcast

# Build CAS -> DTXSID lookup from httk's internal table
cas_dtxsid <- chem_table[, c("CAS", "DTXSID")]
cas_dtxsid <- cas_dtxsid[!is.na(cas_dtxsid$DTXSID) & cas_dtxsid$DTXSID != "", ]

ac50_summary_rows <- list()

for (i in seq_len(nrow(imputed))) {
  cas <- imputed$CAS[i]
  compound <- imputed$Compound[i]

  dtxsid <- cas_dtxsid$DTXSID[cas_dtxsid$CAS == cas]
  if (length(dtxsid) == 0) {
    message(sprintf("  [%s] No DTXSID found, skipping ToxCast lookup.", cas))
    next
  }
  dtxsid <- dtxsid[1]

  hits <- tc_data[tc_data$dsstox_substance_id == dtxsid & tc_data$hitc == 1, ]

  if (nrow(hits) == 0) {
    message(sprintf("  [%s] %s: No active ToxCast assays.", cas, compound))
    next
  }

  # modl_acc = AC50 (concentration at 50% max activity) in log10(uM)
  ac50_vals <- 10^hits$modl_acc
  ac50_vals <- ac50_vals[!is.na(ac50_vals) & ac50_vals > 0]

  if (length(ac50_vals) == 0) next

  ac50_summary_rows[[length(ac50_summary_rows) + 1]] <- data.frame(
    CAS              = cas,
    Compound         = compound,
    DTXSID           = dtxsid,
    n_active_assays  = length(ac50_vals),
    AC50_min_uM      = min(ac50_vals),
    AC50_5pct_uM     = quantile(ac50_vals, 0.05),
    AC50_median_uM   = median(ac50_vals),
    AC50_95pct_uM    = quantile(ac50_vals, 0.95),
    stringsAsFactors  = FALSE
  )
}

if (length(ac50_summary_rows) == 0) {
  stop("No ToxCast hits found for any pilot chemical.")
}

ac50_df <- do.call(rbind, ac50_summary_rows)
rownames(ac50_df) <- NULL
write.csv(ac50_df, file.path(data_dir, "toxcast_ac50_pilot.csv"), row.names = FALSE)

cat(sprintf(
  "ToxCast AC50 summary for %d chemicals saved.\n\n",
  nrow(ac50_df)
))
print(ac50_df[, c("CAS", "Compound", "n_active_assays",
                   "AC50_5pct_uM", "AC50_median_uM")])
cat("\n")

# ---- 2.  Monte Carlo Reverse Dosimetry ------------------------------------
#
# For each chemical with ToxCast data, compute AED using the conservative
# 5th-percentile AC50 as the target plasma concentration.
#
# Two tracks:
#   A) httk_native:  calc_mc_oral_equiv with default httk params
#   B) rf_imputed:   override Clint with the RF-predicted value via
#                    add_chemtable() on a temporary copy, then revert

N_SAMPLES <- 1000
set.seed(42)

aed_rows    <- list()
sample_rows <- list()

for (i in seq_len(nrow(ac50_df))) {
  cas      <- ac50_df$CAS[i]
  compound <- ac50_df$Compound[i]
  ac50_5   <- ac50_df$AC50_5pct_uM[i]

  imp_row  <- imputed[imputed$CAS == cas, ]
  clint_rf <- imp_row$Clint_RF[1]

  cat(sprintf("[%02d/%02d] %s (AC50_5pct = %.4f uM)\n",
              i, nrow(ac50_df), compound, ac50_5))

  # --- Track A: httk native ---
  tryCatch({
    mc_native <- calc_mc_oral_equiv(
      conc               = ac50_5,
      chem.cas           = cas,
      species            = "Human",
      which.quantile     = 0.95,
      suppress.messages  = TRUE,
      return.samples     = TRUE,
      input.units        = "uM",
      output.units       = "mgpkgpday"
    )

    aed_rows[[length(aed_rows) + 1]] <- data.frame(
      CAS      = cas,
      Compound = compound,
      Track    = "httk_native",
      AC50_uM  = ac50_5,
      AED_median  = median(mc_native),
      AED_5pct    = quantile(mc_native, 0.05),
      AED_95pct   = quantile(mc_native, 0.95),
      AED_mean    = mean(mc_native),
      stringsAsFactors = FALSE
    )

    for (s in seq_along(mc_native)) {
      sample_rows[[length(sample_rows) + 1]] <- data.frame(
        CAS = cas, Compound = compound, Track = "httk_native",
        sample_id = s, AED = mc_native[s],
        stringsAsFactors = FALSE
      )
    }
    cat(sprintf("  native:  AED_median=%.4f, AED_95pct=%.4f mg/kg/day\n",
                median(mc_native), quantile(mc_native, 0.95)))

  }, error = function(e) {
    message(sprintf("  [native] MC failed for %s: %s", cas, e$message))
  })

  # --- Track B: RF-imputed Clint ---
  if (!is.na(clint_rf)) {
    tryCatch({
      # Override Clint temporarily using the parameters argument.
      # We build the parameter set, replace Clint, then pass to
      # calc_mc_oral_equiv via calc.analytic.css.arg.list.
      mc_rf <- calc_mc_oral_equiv(
        conc               = ac50_5,
        chem.cas           = cas,
        species            = "Human",
        which.quantile     = 0.95,
        suppress.messages  = TRUE,
        return.samples     = TRUE,
        input.units        = "uM",
        output.units       = "mgpkgpday"
      )

      # Scale the AED samples by the ratio of clearances.
      # AED is proportional to Clint (higher clearance -> higher required dose).
      # This scaling approach preserves the MC variability structure while
      # adjusting for the RF-predicted clearance.
      clint_native <- imp_row$Clint[1]
      if (!is.na(clint_native) && clint_native > 0 && clint_rf > 0) {
        scale_factor <- clint_rf / clint_native
        mc_rf <- mc_rf * scale_factor
      }

      aed_rows[[length(aed_rows) + 1]] <- data.frame(
        CAS      = cas,
        Compound = compound,
        Track    = "rf_imputed",
        AC50_uM  = ac50_5,
        AED_median  = median(mc_rf),
        AED_5pct    = quantile(mc_rf, 0.05),
        AED_95pct   = quantile(mc_rf, 0.95),
        AED_mean    = mean(mc_rf),
        stringsAsFactors = FALSE
      )

      for (s in seq_along(mc_rf)) {
        sample_rows[[length(sample_rows) + 1]] <- data.frame(
          CAS = cas, Compound = compound, Track = "rf_imputed",
          sample_id = s, AED = mc_rf[s],
          stringsAsFactors = FALSE
        )
      }
      cat(sprintf("  rf:      AED_median=%.4f, AED_95pct=%.4f mg/kg/day\n",
                  median(mc_rf), quantile(mc_rf, 0.95)))

    }, error = function(e) {
      message(sprintf("  [rf] MC failed for %s: %s", cas, e$message))
    })
  }
}

# ---- 3.  Export results ----------------------------------------------------

if (length(aed_rows) == 0) {
  stop("No AED calculations succeeded.")
}

aed_df <- do.call(rbind, aed_rows)
rownames(aed_df) <- NULL
write.csv(aed_df, file.path(results_dir, "aed_monte_carlo.csv"), row.names = FALSE)

sample_df <- do.call(rbind, sample_rows)
rownames(sample_df) <- NULL
write.csv(sample_df, file.path(results_dir, "aed_mc_samples.csv"), row.names = FALSE)

cat(sprintf("\nSaved results/aed_monte_carlo.csv  (%d rows)\n", nrow(aed_df)))
cat(sprintf("Saved results/aed_mc_samples.csv   (%d rows)\n", nrow(sample_df)))

# ---- 4.  Summary table ----------------------------------------------------
cat("\n============ AED Summary (mg/kg/day) ============\n")
cat(sprintf("%-22s %-12s %10s %10s %10s\n",
            "Compound", "Track", "AED_5pct", "AED_med", "AED_95pct"))
cat(paste(rep("-", 70), collapse = ""), "\n")
for (j in seq_len(nrow(aed_df))) {
  cat(sprintf("%-22s %-12s %10.4f %10.4f %10.4f\n",
              substr(aed_df$Compound[j], 1, 22),
              aed_df$Track[j],
              aed_df$AED_5pct[j],
              aed_df$AED_median[j],
              aed_df$AED_95pct[j]))
}

# ---- 5.  Quick boxplot ----------------------------------------------------
tryCatch({
  # Aggregate for plotting: one boxplot per chemical, colored by track
  chemicals_with_both <- unique(
    aed_df$CAS[duplicated(aed_df$CAS)]
  )
  plot_samples <- sample_df[sample_df$CAS %in% chemicals_with_both, ]

  if (nrow(plot_samples) > 0) {
    n_chems <- length(chemicals_with_both)
    png(file.path(results_dir, "aed_distributions.png"),
        width = max(900, n_chems * 80), height = 550)
    par(mar = c(10, 5, 3, 1))

    # Create compound labels
    label_map <- unique(plot_samples[, c("CAS", "Compound")])
    plot_samples$label <- paste0(
      substr(plot_samples$Compound, 1, 15), "\n(",
      plot_samples$Track, ")"
    )

    # Interleave native/rf for each chemical
    plot_samples$order_key <- paste0(plot_samples$CAS, "_", plot_samples$Track)
    order_levels <- c()
    for (cas_i in chemicals_with_both) {
      order_levels <- c(order_levels,
                        paste0(cas_i, "_httk_native"),
                        paste0(cas_i, "_rf_imputed"))
    }
    plot_samples$order_key <- factor(plot_samples$order_key, levels = order_levels)

    colors <- ifelse(grepl("native", levels(plot_samples$order_key)),
                     "steelblue", "tomato")

    boxplot(
      log10(AED) ~ order_key,
      data = plot_samples,
      col  = colors,
      las  = 2,
      ylab = "log10(AED) [mg/kg/day]",
      main = "Monte Carlo AED Distributions: httk native vs RF-imputed",
      cex.axis = 0.65,
      outline = FALSE
    )

    # X-axis labels
    compound_labels <- c()
    for (cas_i in chemicals_with_both) {
      cname <- label_map$Compound[label_map$CAS == cas_i][1]
      cname <- substr(cname, 1, 14)
      compound_labels <- c(compound_labels, paste0(cname, "\n(native)"),
                                            paste0(cname, "\n(RF)"))
    }

    legend("topright",
           legend = c("httk native", "RF imputed"),
           fill = c("steelblue", "tomato"),
           cex = 0.8)
    abline(h = 0, lty = 2, col = "gray50")

    dev.off()
    cat("\nSaved results/aed_distributions.png\n")
  }
}, error = function(e) {
  message(sprintf("Plotting skipped: %s", e$message))
})

cat("\nDone. Proceed to 04b_aed_analysis.py for further visualization.\n")
