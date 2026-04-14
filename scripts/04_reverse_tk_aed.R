###############################################################################
# 04_reverse_tk_aed.R
# ---------------------------------------------------------------------------
# Reverse Toxicokinetics (RTK) Pilot
#
# 1. Finds the intersection of chemicals that are both parametrisable with
#    the 3compartmentss model AND flagged as ToxCast chemicals in httk.
# 2. Selects a diverse pilot set of 20 (including Bisphenol A).
# 3. For each chemical:
#      a) Retrieves ToxCast AC50 data (active assays from example.toxcast),
#         computes the 10th-percentile AC50 as a conservative bioactivity
#         threshold.
#      b) Uses calc_mc_oral_equiv() with 1000 Monte Carlo samples to convert
#         the AC50 to an Administered Equivalent Dose (AED, mg/kg/day).
#         The 95th percentile of the MC distribution captures the dose
#         required to reach the target concentration in the most sensitive
#         5 % of the population.
# 4. Exports results/pilot_aed_results.csv
#
# References:
#   - Wetmore et al. (2015) - RTK framework
#   - Ring et al. (2017) - Monte Carlo population variability in httk
#   - Pearce et al. (2017) - IVIVE with httk
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
# STEP 1: Find the intersection of 3compartmentss-parametrisable chemicals
#         that also have ToxCast data
###############################################################################

cat("===== Step 1: Finding 3compartmentss x ToxCast intersection =====\n\n")

all_3css <- suppressWarnings(
  get_cheminfo(info = "all", median.only = TRUE, model = "3compartmentss")
)
cat(sprintf("Chemicals parametrisable with 3compartmentss: %d\n", nrow(all_3css)))

tc_flags <- sapply(all_3css$CAS, function(cas) {
  tryCatch(is.toxcast(cas), error = function(e) FALSE)
})
eligible <- all_3css[tc_flags, ]
cat(sprintf("Of those, also ToxCast chemicals:            %d\n\n", nrow(eligible)))

###############################################################################
# STEP 2: Select a diverse pilot set of 20 chemicals (keep BPA)
###############################################################################

cat("===== Step 2: Selecting 20 diverse pilot chemicals =====\n\n")

set.seed(42)

bpa <- eligible[eligible$CAS == "80-05-7", ]
others <- eligible[eligible$CAS != "80-05-7", ]
others <- others[order(as.numeric(others$MW)), ]

# Evenly-spaced selection across the MW range for chemical diversity
idx <- round(seq(1, nrow(others), length.out = 19))
pilot20 <- rbind(bpa, others[idx, ])

cat(sprintf("Selected %d pilot chemicals.\n", nrow(pilot20)))
cat(sprintf("  MW range:  %.1f -- %.1f\n",
            min(as.numeric(pilot20$MW)), max(as.numeric(pilot20$MW))))
cat(sprintf("  logP range: %.2f -- %.2f\n\n",
            min(as.numeric(pilot20$logP)), max(as.numeric(pilot20$logP))))

print(pilot20[, c("CAS", "Compound", "MW", "logP")])
cat("\n")

###############################################################################
# STEP 3: Retrieve ToxCast AC50 data & compute 10th-percentile AC50
###############################################################################

cat("===== Step 3: ToxCast AC50 retrieval =====\n\n")

tc_data     <- httk::example.toxcast
chem_table  <- httk::chem.physical_and_invitro.data
cas_dtxsid  <- chem_table[!is.na(chem_table$DTXSID) & chem_table$DTXSID != "",
                          c("CAS", "DTXSID")]

ac50_rows <- list()

for (i in seq_len(nrow(pilot20))) {
  cas      <- pilot20$CAS[i]
  compound <- pilot20$Compound[i]

  dtxsid <- cas_dtxsid$DTXSID[cas_dtxsid$CAS == cas]
  if (length(dtxsid) == 0) {
    message(sprintf("  [%s] %s: No DTXSID, skipping.", cas, compound))
    next
  }
  dtxsid <- dtxsid[1]

  hits <- tc_data[tc_data$dsstox_substance_id == dtxsid &
                  tc_data$hitc == 1, ]

  if (nrow(hits) == 0) {
    # No active assays in example.toxcast; use is.toxcast flag to still
    # include the chemical. We'll use a placeholder AC50 from the
    # calc_mc_oral_equiv default (1 uM) so the RTK calculation can proceed.
    ac50_rows[[length(ac50_rows) + 1]] <- data.frame(
      CAS             = cas,
      Compound        = compound,
      DTXSID          = dtxsid,
      n_active_assays = 0,
      AC50_10pct_uM   = 1.0,
      AC50_median_uM  = 1.0,
      AC50_source     = "default_1uM",
      stringsAsFactors = FALSE
    )
    next
  }

  ac50_vals <- 10^hits$modl_acc
  ac50_vals <- ac50_vals[!is.na(ac50_vals) & ac50_vals > 0]

  if (length(ac50_vals) == 0) {
    ac50_rows[[length(ac50_rows) + 1]] <- data.frame(
      CAS = cas, Compound = compound, DTXSID = dtxsid,
      n_active_assays = nrow(hits),
      AC50_10pct_uM = 1.0, AC50_median_uM = 1.0,
      AC50_source = "default_1uM",
      stringsAsFactors = FALSE
    )
    next
  }

  ac50_rows[[length(ac50_rows) + 1]] <- data.frame(
    CAS             = cas,
    Compound        = compound,
    DTXSID          = dtxsid,
    n_active_assays = length(ac50_vals),
    AC50_10pct_uM   = as.numeric(quantile(ac50_vals, 0.10)),
    AC50_median_uM  = median(ac50_vals),
    AC50_source     = "example.toxcast",
    stringsAsFactors = FALSE
  )
}

ac50_df <- do.call(rbind, ac50_rows)
rownames(ac50_df) <- NULL

cat(sprintf("AC50 data for %d chemicals:\n", nrow(ac50_df)))
cat(sprintf("  With real ToxCast assays: %d\n",
            sum(ac50_df$AC50_source == "example.toxcast")))
cat(sprintf("  Using default 1 uM:      %d\n\n",
            sum(ac50_df$AC50_source == "default_1uM")))

print(ac50_df[, c("CAS", "Compound", "n_active_assays",
                   "AC50_10pct_uM", "AC50_source")])
cat("\n")

###############################################################################
# STEP 4: Monte Carlo Reverse Dosimetry -> AED
###############################################################################

cat("===== Step 4: Monte Carlo Reverse Dosimetry =====\n\n")

N_SAMPLES <- 1000
set.seed(42)

aed_rows <- list()

for (i in seq_len(nrow(ac50_df))) {
  cas      <- ac50_df$CAS[i]
  compound <- ac50_df$Compound[i]
  ac50_10  <- ac50_df$AC50_10pct_uM[i]

  cat(sprintf("[%02d/%02d] %s (%s)  AC50_10pct = %.4f uM ... ",
              i, nrow(ac50_df), compound, cas, ac50_10))

  tryCatch({
    mc_samples <- calc_mc_oral_equiv(
      conc              = ac50_10,
      chem.cas          = cas,
      species           = "Human",
      which.quantile    = 0.95,
      suppress.messages = TRUE,
      return.samples    = TRUE,
      input.units       = "uM",
      output.units      = "mgpkgpday"
    )

    aed_rows[[length(aed_rows) + 1]] <- data.frame(
      CAS              = cas,
      Compound         = compound,
      AC50_10pct_uM    = ac50_10,
      AC50_source      = ac50_df$AC50_source[i],
      n_active_assays  = ac50_df$n_active_assays[i],
      AED_median       = median(mc_samples),
      AED_mean         = mean(mc_samples),
      AED_5pct         = as.numeric(quantile(mc_samples, 0.05)),
      AED_95pct        = as.numeric(quantile(mc_samples, 0.95)),
      AED_sd           = sd(mc_samples),
      n_mc_samples     = length(mc_samples),
      stringsAsFactors = FALSE
    )

    cat(sprintf("AED_95pct = %.4f mg/kg/day\n", quantile(mc_samples, 0.95)))

  }, error = function(e) {
    message(sprintf("FAILED: %s", e$message))
  })
}

###############################################################################
# STEP 5: Export results
###############################################################################

if (length(aed_rows) == 0) {
  stop("No AED calculations succeeded. Check httk installation.")
}

aed_df <- do.call(rbind, aed_rows)
rownames(aed_df) <- NULL

# Sort by AED_95pct (most potent first)
aed_df <- aed_df[order(aed_df$AED_95pct), ]

out_file <- file.path(results_dir, "pilot_aed_results.csv")
write.csv(aed_df, out_file, row.names = FALSE)

cat(sprintf("\n===== Results saved to %s =====\n\n", out_file))

# ---- Pretty-print summary table -------------------------------------------
cat(sprintf("%-24s %7s %10s %10s %10s %10s  %s\n",
            "Compound", "Assays", "AC50_10p", "AED_med", "AED_95pct",
            "AED_5pct", "AC50 src"))
cat(paste(rep("-", 95), collapse = ""), "\n")

for (j in seq_len(nrow(aed_df))) {
  cat(sprintf("%-24s %7d %10.4f %10.4f %10.4f %10.4f  %s\n",
              substr(aed_df$Compound[j], 1, 24),
              aed_df$n_active_assays[j],
              aed_df$AC50_10pct_uM[j],
              aed_df$AED_median[j],
              aed_df$AED_95pct[j],
              aed_df$AED_5pct[j],
              aed_df$AC50_source[j]))
}

cat(paste(rep("-", 95), collapse = ""), "\n")
cat(sprintf("\nTotal chemicals: %d\n", nrow(aed_df)))
cat(sprintf("AED_95pct range: %.4f -- %.4f mg/kg/day\n",
            min(aed_df$AED_95pct), max(aed_df$AED_95pct)))
cat(sprintf("Median AED_95pct across pilot set: %.4f mg/kg/day\n",
            median(aed_df$AED_95pct)))

cat("\nInterpretation:\n")
cat("  AED_95pct = dose (mg/kg/day) that would produce the target plasma\n")
cat("  concentration in 95%% of the population (i.e. covers the most\n")
cat("  sensitive 5%%). Lower AED_95pct = higher potency concern.\n")
cat("\nDone.\n")
