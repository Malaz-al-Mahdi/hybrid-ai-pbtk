###############################################################################
# 03_httk_pbtk_simulation.R
# ---------------------------------------------------------------------------
# Reads the RF-imputed parameter table and runs httk PBTK simulations for
# each pilot chemical.  Two simulation tracks:
#
#   A) "httk_native"  – uses httk's built-in Clint (ground truth baseline)
#   B) "rf_imputed"   – overrides Clint with the RF prediction
#
# Comparison of Cmax, AUC, and css (steady-state concentration) between the
# two tracks quantifies how sensitive PBTK output is to the ML-predicted
# clearance.
#
# Outputs:
#   results/pbtk_comparison.csv    – per-chemical TK summary statistics
#   results/pbtk_comparison.png    – visual comparison plots
#   results/pbtk_curves/           – individual concentration-time plots
###############################################################################

# Prefer the per-user Windows R library when available.
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

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
project_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

# ---- 0. Setup --------------------------------------------------------------

data_dir    <- file.path(project_dir, "data")
results_dir <- file.path(project_dir, "results")
curves_dir  <- file.path(results_dir, "pbtk_curves")
dir.create(curves_dir, showWarnings = FALSE, recursive = TRUE)

imputed_file <- file.path(data_dir, "pilot_chemicals_imputed.csv")
if (!file.exists(imputed_file)) {
  stop("pilot_chemicals_imputed.csv not found. Run step 02 first.")
}

imputed <- read.csv(imputed_file, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d chemicals from pilot_chemicals_imputed.csv\n",
            nrow(imputed)))

# ---- 1. Helper: run PBTK with optional Clint override ---------------------

run_pbtk_safe <- function(cas, clint_override = NULL, label = "native",
                          days = 28, doses.per.day = 1,
                          daily.dose = 1) {  # mg/kg/day
  tryCatch({
    # Build parameter list – use 3compartmentss (faster) or pbtk
    params <- parameterize_pbtk(chem.cas = cas)

    if (!is.null(clint_override)) {
      params$Clint <- clint_override
    }

    out <- solve_pbtk(
      parameters   = params,
      days         = days,
      doses.per.day = doses.per.day,
      daily.dose   = daily.dose,
      plots        = FALSE,
      suppress.messages = TRUE
    )

    out_df <- as.data.frame(out)

    # Extract key TK statistics
    Cplasma <- out_df$Cplasma
    time     <- out_df$time

    list(
      CAS   = cas,
      label = label,
      Cmax  = max(Cplasma, na.rm = TRUE),
      Tmax  = time[which.max(Cplasma)],
      AUC   = sum(diff(time) * (head(Cplasma, -1) + tail(Cplasma, -1)) / 2),
      Css   = mean(tail(Cplasma, round(length(Cplasma) * 0.1))),
      timeseries = out_df
    )

  }, error = function(e) {
    message(sprintf("  [%s] PBTK failed for %s: %s", label, cas, e$message))
    return(NULL)
  })
}

# ---- 2. Run simulations for each chemical ----------------------------------

comparison_rows <- list()

for (i in seq_len(nrow(imputed))) {
  cas      <- imputed$CAS[i]
  compound <- imputed$Compound[i]
  clint_rf <- imputed$Clint_RF[i]
  clint_orig <- imputed$Clint[i]

  cat(sprintf("\n[%02d/%02d] %s (%s)\n", i, nrow(imputed), compound, cas))

  # Track A: httk native parameters
  res_native <- run_pbtk_safe(cas, clint_override = NULL, label = "httk_native")

  # Track B: RF-imputed Clint
  res_rf <- NULL
  if (!is.na(clint_rf)) {
    res_rf <- run_pbtk_safe(cas, clint_override = clint_rf, label = "rf_imputed")
  }

  # Collect summary
  if (!is.null(res_native)) {
    comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
      CAS       = cas,
      Compound  = compound,
      Track     = "httk_native",
      Clint_used = clint_orig,
      Cmax      = res_native$Cmax,
      AUC       = res_native$AUC,
      Css       = res_native$Css,
      stringsAsFactors = FALSE
    )
  }
  if (!is.null(res_rf)) {
    comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
      CAS       = cas,
      Compound  = compound,
      Track     = "rf_imputed",
      Clint_used = clint_rf,
      Cmax      = res_rf$Cmax,
      AUC       = res_rf$AUC,
      Css       = res_rf$Css,
      stringsAsFactors = FALSE
    )
  }

  # ---- Individual concentration-time plot ----------------------------------
  if (!is.null(res_native) || !is.null(res_rf)) {
    safe_name <- gsub("[^A-Za-z0-9_-]", "_", compound)
    png(file.path(curves_dir, sprintf("%s_%s.png", safe_name, cas)),
        width = 800, height = 400)
    par(mar = c(4, 4, 3, 1))

    ylim_max <- 0
    if (!is.null(res_native))
      ylim_max <- max(ylim_max, max(res_native$timeseries$Cplasma, na.rm = TRUE))
    if (!is.null(res_rf))
      ylim_max <- max(ylim_max, max(res_rf$timeseries$Cplasma, na.rm = TRUE))

    plot(NULL, xlim = c(0, 28), ylim = c(0, ylim_max * 1.1),
         xlab = "Time (days)", ylab = "Cplasma (uM)",
         main = sprintf("%s (%s)", compound, cas))

    if (!is.null(res_native))
      lines(res_native$timeseries$time, res_native$timeseries$Cplasma,
            col = "steelblue", lwd = 2)
    if (!is.null(res_rf))
      lines(res_rf$timeseries$time, res_rf$timeseries$Cplasma,
            col = "tomato", lwd = 2, lty = 2)

    legend("topright",
           legend = c(
             sprintf("httk native (Clint=%.2f)", clint_orig),
             sprintf("RF imputed  (Clint=%.2f)", ifelse(is.na(clint_rf), 0, clint_rf))
           ),
           col = c("steelblue", "tomato"), lwd = 2, lty = c(1, 2),
           cex = 0.8)
    dev.off()
  }
}

# ---- 3. Compile & export comparison table ----------------------------------

if (length(comparison_rows) == 0) {
  writeLines(
    c(
      "No PBTK simulations completed successfully.",
      "Check package dependencies and httk runtime errors above."
    ),
    con = file.path(results_dir, "pbtk_run_diagnostic.txt")
  )
  stop("No PBTK simulations completed successfully. See results/pbtk_run_diagnostic.txt.")
}

comp_df <- do.call(rbind, comparison_rows)
write.csv(comp_df, file.path(results_dir, "pbtk_comparison.csv"),
          row.names = FALSE)
cat(sprintf("\nSaved results/pbtk_comparison.csv (%d rows)\n", nrow(comp_df)))

# ---- 4. Summary comparison plot -------------------------------------------

# Reshape to wide for paired comparisons
native_df <- comp_df[comp_df$Track == "httk_native", ]
rf_df     <- comp_df[comp_df$Track == "rf_imputed", ]

merged <- merge(native_df, rf_df,
                by = c("CAS", "Compound"),
                suffixes = c("_native", "_rf"))

if (nrow(merged) > 0) {
  png(file.path(results_dir, "pbtk_comparison.png"),
      width = 1200, height = 400)
  par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))

  # Cmax comparison
  lim <- range(c(merged$Cmax_native, merged$Cmax_rf), na.rm = TRUE)
  plot(merged$Cmax_native, merged$Cmax_rf,
       xlab = "Cmax (httk native)", ylab = "Cmax (RF imputed)",
       main = "Cmax comparison", pch = 19, col = "steelblue",
       xlim = lim, ylim = lim)
  abline(0, 1, lty = 2)

  # AUC comparison
  lim <- range(c(merged$AUC_native, merged$AUC_rf), na.rm = TRUE)
  plot(merged$AUC_native, merged$AUC_rf,
       xlab = "AUC (httk native)", ylab = "AUC (RF imputed)",
       main = "AUC comparison", pch = 19, col = "darkgreen",
       xlim = lim, ylim = lim)
  abline(0, 1, lty = 2)

  # Css comparison
  lim <- range(c(merged$Css_native, merged$Css_rf), na.rm = TRUE)
  plot(merged$Css_native, merged$Css_rf,
       xlab = "Css (httk native)", ylab = "Css (RF imputed)",
       main = "Css comparison", pch = 19, col = "tomato",
       xlim = lim, ylim = lim)
  abline(0, 1, lty = 2)

  dev.off()
  cat("Saved results/pbtk_comparison.png\n")
}

# ---- 5. Print fold-change summary -----------------------------------------

if (nrow(merged) > 0) {
  merged$FC_Cmax <- merged$Cmax_rf / merged$Cmax_native
  merged$FC_AUC  <- merged$AUC_rf  / merged$AUC_native

  cat("\n========== Fold-Change Summary (RF / native) ==========\n")
  cat(sprintf("%-25s  FC_Cmax   FC_AUC\n", "Compound"))
  cat(paste(rep("-", 55), collapse = ""), "\n")
  for (j in seq_len(nrow(merged))) {
    cat(sprintf("%-25s  %7.3f   %7.3f\n",
                substr(merged$Compound[j], 1, 25),
                merged$FC_Cmax[j],
                merged$FC_AUC[j]))
  }

  cat(paste(rep("-", 55), collapse = ""), "\n")
  cat(sprintf("%-25s  %7.3f   %7.3f\n", "MEDIAN",
              median(merged$FC_Cmax, na.rm = TRUE),
              median(merged$FC_AUC,  na.rm = TRUE)))
  cat(sprintf("%-25s  %7.3f   %7.3f\n", "MEAN",
              mean(merged$FC_Cmax, na.rm = TRUE),
              mean(merged$FC_AUC,  na.rm = TRUE)))
}

cat("\nDone. All results in results/ directory.\n")
