###############################################################################
# 05_full_rtk_aed_ber.R
# ---------------------------------------------------------------------------
# Complete Reverse Toxicokinetics Pipeline:
#
#   1. Identifies the 777-chemical intersection (3compartmentss + ToxCast)
#   2. Trains a Random Forest on all available Clint data to predict
#      Clint for chemicals with missing/zero clearance values
#   3. For each chemical:
#      a) Retrieves ToxCast AC50 data (10th percentile as conservative threshold)
#      b) Computes AED via Monte Carlo reverse dosimetry (calc_mc_oral_equiv)
#         - Track A: httk-native Clint
#         - Track B: RF-imputed Clint (overrides database temporarily)
#   4. Calculates Bioactivity-Exposure Ratio (BER = AED / SEEM exposure)
#   5. Prioritises chemicals by BER (low BER = high concern)
#
# Outputs:
#   data/all_777_chemicals.csv            – full eligible chemical list
#   data/rf_clint_model_777.rds           – trained RF model (via Python export)
#   results/aed_ber_full.csv              – AED + BER for all chemicals
#   results/aed_ber_summary.csv           – top-priority chemicals (BER < 1)
#   results/ber_ranking_plot.png          – BER waterfall plot
###############################################################################

# ---- Library bootstrap -----------------------------------------------------
r_ver <- paste(
  R.version$major,
  strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1],
  sep = "."
)
r_user_lib <- Sys.getenv("R_LIBS_USER", unset = "")
if (!nzchar(r_user_lib)) {
  r_user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R",
                          "win-library", r_ver)
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
  dirname(normalizePath(sub("^--file=", "", file_arg[1]),
                        winslash = "/", mustWork = FALSE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
project_dir <- normalizePath(file.path(script_dir, ".."),
                             winslash = "/", mustWork = FALSE)
data_dir    <- file.path(project_dir, "data")
results_dir <- file.path(project_dir, "results")
dir.create(data_dir,    recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

###############################################################################
# STEP 1: Build the eligible chemical universe
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 1: Building eligible chemical universe\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

all_3css <- suppressWarnings(
  get_cheminfo(info = "all", median.only = TRUE, model = "3compartmentss")
)
cat(sprintf("  3compartmentss-parametrisable: %d\n", nrow(all_3css)))

tc_flags <- sapply(all_3css$CAS, function(cas)
  tryCatch(is.toxcast(cas), error = function(e) FALSE))
eligible <- all_3css[tc_flags, ]
cat(sprintf("  Also ToxCast chemicals:       %d\n", nrow(eligible)))

seem_flags <- sapply(eligible$CAS, function(cas)
  tryCatch(is.seem(cas), error = function(e) FALSE))
eligible$has_SEEM <- seem_flags
cat(sprintf("  Also with SEEM exposure data: %d\n\n", sum(seem_flags)))

# Enrich with SEEM exposure estimates
seem_data <- example.seem
eligible$SEEM_mg_kg_day   <- NA_real_
eligible$SEEM_l95         <- NA_real_
eligible$SEEM_u95         <- NA_real_
eligible$SEEM_pathway     <- NA_character_

for (i in seq_len(nrow(eligible))) {
  s <- seem_data[seem_data$CAS == eligible$CAS[i], ]
  if (nrow(s) > 0) {
    eligible$SEEM_mg_kg_day[i] <- s$seem3[1]
    eligible$SEEM_l95[i]       <- s$seem3.l95[1]
    eligible$SEEM_u95[i]       <- s$seem3.u95[1]
    eligible$SEEM_pathway[i]   <- as.character(s$Pathway[1])
  }
}

write.csv(eligible, file.path(data_dir, "all_777_chemicals.csv"),
          row.names = FALSE)
cat(sprintf("  Saved data/all_777_chemicals.csv (%d rows)\n\n", nrow(eligible)))

###############################################################################
# STEP 2: RF-imputed Clint values (read from Python output)
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 2: Loading RF-imputed Clint values\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

# Read the imputed Clint from the pilot study (Step 2 output).
# For chemicals NOT in the pilot set, we'll use native httk Clint.
imputed_file <- file.path(data_dir, "pilot_chemicals_imputed.csv")
has_rf <- FALSE
rf_lookup <- list()

if (file.exists(imputed_file)) {
  imp <- read.csv(imputed_file, stringsAsFactors = FALSE)
  for (j in seq_len(nrow(imp))) {
    if (!is.na(imp$Clint_RF[j])) {
      rf_lookup[[imp$CAS[j]]] <- imp$Clint_RF[j]
    }
  }
  has_rf <- length(rf_lookup) > 0
  cat(sprintf("  RF predictions available for %d chemicals.\n\n", length(rf_lookup)))
} else {
  cat("  No RF predictions found (pilot_chemicals_imputed.csv missing).\n")
  cat("  Will run native-only track.\n\n")
}

###############################################################################
# STEP 3: ToxCast AC50 retrieval (10th percentile)
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 3: Retrieving ToxCast AC50 values\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

tc_data    <- httk::example.toxcast
chem_table <- httk::chem.physical_and_invitro.data
cas_dtxsid <- chem_table[!is.na(chem_table$DTXSID) & chem_table$DTXSID != "",
                         c("CAS", "DTXSID")]

eligible$DTXSID_tc     <- NA_character_
eligible$n_tc_assays   <- 0L
eligible$AC50_10pct_uM <- NA_real_
eligible$AC50_med_uM   <- NA_real_

for (i in seq_len(nrow(eligible))) {
  cas <- eligible$CAS[i]
  dtxsid <- cas_dtxsid$DTXSID[cas_dtxsid$CAS == cas]
  if (length(dtxsid) == 0) next
  dtxsid <- dtxsid[1]
  eligible$DTXSID_tc[i] <- dtxsid

  hits <- tc_data[tc_data$dsstox_substance_id == dtxsid & tc_data$hitc == 1, ]
  if (nrow(hits) == 0) next

  ac50_vals <- 10^hits$modl_acc
  ac50_vals <- ac50_vals[!is.na(ac50_vals) & ac50_vals > 0]
  if (length(ac50_vals) == 0) next

  eligible$n_tc_assays[i]   <- length(ac50_vals)
  eligible$AC50_10pct_uM[i] <- as.numeric(quantile(ac50_vals, 0.10))
  eligible$AC50_med_uM[i]   <- median(ac50_vals)
}

has_tc <- sum(eligible$n_tc_assays > 0)
cat(sprintf("  Chemicals with real ToxCast AC50:       %d\n", has_tc))
cat(sprintf("  Chemicals using default 1 uM fallback:  %d\n\n",
            nrow(eligible) - has_tc))

# Fallback: if no ToxCast assays in example.toxcast, use 1 uM
eligible$AC50_10pct_uM[is.na(eligible$AC50_10pct_uM)] <- 1.0
eligible$AC50_med_uM[is.na(eligible$AC50_med_uM)] <- 1.0

###############################################################################
# STEP 4: Monte Carlo Reverse Dosimetry (AED) – dual track
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 4: Monte Carlo Reverse Dosimetry (AED)\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

N_SAMPLES <- 1000
set.seed(42)

# Build a lookup from CAS -> native Clint for RF scaling.
# At steady state AED is proportional to clearance, so:
#   AED_rf = AED_native * (Clint_RF / Clint_native)
# This avoids modifying httk's locked internal database.
native_clint_lookup <- list()
for (j in seq_len(nrow(eligible))) {
  raw <- as.character(eligible$Human.Clint[j])
  vals <- suppressWarnings(as.numeric(unlist(strsplit(raw, ","))))
  vals <- vals[!is.na(vals)]
  if (length(vals) > 0) native_clint_lookup[[eligible$CAS[j]]] <- vals[1]
}

aed_rows <- list()
n_total  <- nrow(eligible)

for (i in seq_len(n_total)) {
  cas      <- eligible$CAS[i]
  compound <- eligible$Compound[i]
  ac50     <- eligible$AC50_10pct_uM[i]

  if (i %% 50 == 1 || i == n_total) {
    cat(sprintf("[%3d/%d] %s ...\n", i, n_total, compound))
  }

  # --- Track A: httk native ---
  mc_samples <- tryCatch({
    calc_mc_oral_equiv(
      conc = ac50, chem.cas = cas, species = "Human",
      which.quantile = 0.95, suppress.messages = TRUE,
      return.samples = TRUE, input.units = "uM",
      output.units = "mgpkgpday"
    )
  }, error = function(e) NULL)

  if (!is.null(mc_samples)) {
    aed_rows[[length(aed_rows) + 1]] <- data.frame(
      CAS = cas, Compound = compound, Track = "httk_native",
      AC50_10pct_uM = ac50,
      AED_median  = median(mc_samples),
      AED_95pct   = as.numeric(quantile(mc_samples, 0.95)),
      AED_5pct    = as.numeric(quantile(mc_samples, 0.05)),
      stringsAsFactors = FALSE
    )

    # --- Track B: RF-imputed Clint (scaling approach) ---
    if (has_rf && cas %in% names(rf_lookup)) {
      clint_native <- native_clint_lookup[[cas]]
      clint_rf     <- rf_lookup[[cas]]

      if (!is.null(clint_native) && !is.na(clint_native) &&
          clint_native > 0 && !is.na(clint_rf) && clint_rf > 0) {
        scale <- clint_rf / clint_native
        mc_rf <- mc_samples * scale

        aed_rows[[length(aed_rows) + 1]] <- data.frame(
          CAS = cas, Compound = compound, Track = "rf_imputed",
          AC50_10pct_uM = ac50,
          AED_median  = median(mc_rf),
          AED_95pct   = as.numeric(quantile(mc_rf, 0.95)),
          AED_5pct    = as.numeric(quantile(mc_rf, 0.05)),
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

aed_df <- do.call(rbind, aed_rows)
rownames(aed_df) <- NULL

cat(sprintf("\n  AED calculated for %d chemical-track pairs.\n\n",
            nrow(aed_df)))

###############################################################################
# STEP 5: Bioactivity-Exposure Ratio (BER)
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 5: Bioactivity-Exposure Ratio (BER = AED / Exposure)\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

# Merge AED with SEEM exposure data
seem_cols <- eligible[, c("CAS", "SEEM_mg_kg_day", "SEEM_l95", "SEEM_u95",
                          "SEEM_pathway", "has_SEEM", "n_tc_assays",
                          "AC50_med_uM")]
aed_df <- merge(aed_df, seem_cols, by = "CAS", all.x = TRUE)

# BER = AED_95pct / SEEM exposure
# AED_95pct covers the sensitive 5% of the population
# Using SEEM median exposure estimate
aed_df$BER <- ifelse(
  !is.na(aed_df$SEEM_mg_kg_day) & aed_df$SEEM_mg_kg_day > 0,
  aed_df$AED_95pct / aed_df$SEEM_mg_kg_day,
  NA_real_
)

# Classify risk concern
aed_df$concern <- ifelse(
  is.na(aed_df$BER), "no_exposure_data",
  ifelse(aed_df$BER < 1, "HIGH (BER<1)",
  ifelse(aed_df$BER < 10, "MODERATE (1<BER<10)",
  ifelse(aed_df$BER < 100, "LOW (10<BER<100)",
         "MINIMAL (BER>100)")))
)

# Export full results
write.csv(aed_df, file.path(results_dir, "aed_ber_full.csv"),
          row.names = FALSE)
cat(sprintf("  Saved results/aed_ber_full.csv (%d rows)\n", nrow(aed_df)))

# Summary: focus on native track for BER ranking
native_ber <- aed_df[aed_df$Track == "httk_native" & !is.na(aed_df$BER), ]
native_ber <- native_ber[order(native_ber$BER), ]

cat(sprintf("\n  BER statistics (httk_native track, n = %d):\n", nrow(native_ber)))
cat(sprintf("    Chemicals with BER < 1  (HIGH concern):     %d\n",
            sum(native_ber$BER < 1)))
cat(sprintf("    Chemicals with BER < 10 (MODERATE+ concern): %d\n",
            sum(native_ber$BER < 10)))
cat(sprintf("    Chemicals with BER < 100:                    %d\n",
            sum(native_ber$BER < 100)))
cat(sprintf("    BER range: %.2e -- %.2e\n",
            min(native_ber$BER), max(native_ber$BER)))

# Top 30 highest-priority chemicals
top30 <- head(native_ber, 30)
write.csv(top30, file.path(results_dir, "aed_ber_summary.csv"),
          row.names = FALSE)
cat(sprintf("\n  Saved results/aed_ber_summary.csv (top 30 by BER)\n"))

cat("\n  Top 20 chemicals by BER (lowest = highest concern):\n")
cat(sprintf("  %-24s %10s %12s %12s %8s\n",
            "Compound", "AED_95pct", "SEEM_exp", "BER", "Concern"))
cat(paste("  ", paste(rep("-", 75), collapse = ""), "\n"))
for (j in seq_len(min(20, nrow(top30)))) {
  cat(sprintf("  %-24s %10.4f %12.2e %12.2f %8s\n",
              substr(top30$Compound[j], 1, 24),
              top30$AED_95pct[j],
              top30$SEEM_mg_kg_day[j],
              top30$BER[j],
              top30$concern[j]))
}

###############################################################################
# STEP 6: BER waterfall plot
###############################################################################

cat(paste("\n", paste(rep("=", 70), collapse = ""), "\n"))
cat("STEP 6: Generating BER ranking plot\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

tryCatch({
  plot_df <- head(native_ber, 40)
  n_plot  <- nrow(plot_df)

  png(file.path(results_dir, "ber_ranking_plot.png"),
      width = 1000, height = 600)
  par(mar = c(10, 5, 3, 2))

  colors <- ifelse(plot_df$BER < 1, "firebrick",
            ifelse(plot_df$BER < 10, "darkorange",
            ifelse(plot_df$BER < 100, "goldenrod", "steelblue")))

  bp <- barplot(
    log10(plot_df$BER),
    names.arg = substr(plot_df$Compound, 1, 14),
    col = colors,
    las = 2,
    cex.names = 0.6,
    ylab = "log10(BER)",
    main = "Bioactivity-Exposure Ratio (BER) Ranking\n(lower = higher concern)"
  )

  abline(h = 0, lty = 2, col = "firebrick", lwd = 2)   # BER = 1
  abline(h = 1, lty = 2, col = "darkorange", lwd = 1)   # BER = 10
  abline(h = 2, lty = 2, col = "goldenrod", lwd = 1)     # BER = 100

  legend("topright",
         legend = c("BER < 1 (HIGH)", "1-10 (MODERATE)",
                    "10-100 (LOW)", "> 100 (MINIMAL)"),
         fill = c("firebrick", "darkorange", "goldenrod", "steelblue"),
         cex = 0.75)

  dev.off()
  cat("  Saved results/ber_ranking_plot.png\n")
}, error = function(e) {
  message(sprintf("  Plot skipped: %s", e$message))
})

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Pipeline complete.\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
