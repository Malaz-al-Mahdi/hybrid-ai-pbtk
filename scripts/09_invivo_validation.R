###############################################################################
# 09_invivo_validation.R
# ---------------------------------------------------------------------------
# Validation of AI-assisted PBTK/TK predictions against published in-vivo
# pharmacokinetic data.
#
# Scientific rationale
# ~~~~~~~~~~~~~~~~~~~~
# The credibility of any predictive TK model requires head-to-head comparison
# with measured in-vivo data.  httk bundles curated literature PK values from
# Wetmore et al. (2012) and other sources, enabling systematic validation.
#
# Metrics computed
# ~~~~~~~~~~~~~~~~
#   • Pearson r and R² on log10(Css) scale
#   • Root Mean Squared Error (RMSE) on log10 scale
#   • Geometric Mean Ratio (GMR)  = geometric mean(predicted / observed)
#   • Fraction within 2-, 3-, 10-fold of observed
#
# Models compared
# ~~~~~~~~~~~~~~~
#   A) 3compartmentss  – fast screening model (used in Step 5/6)
#   B) PBTK            – physiologically based model (used in Step 3)
#
# Outputs
# ~~~~~~~
#   results/invivo_validation.csv          Full predicted vs. observed table
#   results/invivo_validation_scatter.png  Log-log correlation plot
#   results/invivo_validation_residuals.png Residual analysis
#   results/invivo_validation_metrics.csv  Summary statistics
###############################################################################

suppressPackageStartupMessages({
  library(httk)
  library(ggplot2)
  library(dplyr)
})

ROOT    <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."),
                         mustWork = FALSE)
if (!nchar(ROOT) || ROOT == ".") ROOT <- getwd()
RESULTS <- file.path(ROOT, "results")
DATA    <- file.path(ROOT, "data")
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

cat("========================================================\n")
cat("Step 9 – In-vivo Validation of PBTK/TK Predictions\n")
cat("========================================================\n\n")

# ── Helper: ensure a package is installed ────────────────────────────────────
ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing '", pkg, "' …")
    install.packages(pkg, repos = "https://cloud.r-project.org",
                     quiet = TRUE)
  }
  library(pkg, character.only = TRUE, quietly = TRUE)
}
ensure_pkg("httk")

# ── 1. Load in-vivo literature data ──────────────────────────────────────────
cat("── 1. Loading in-vivo literature data …\n")

# httk::Wetmore2012 contains measured plasma Css (µg/L) and literature
# pharmacokinetic parameters for environmental chemicals.
lit_available <- tryCatch(
  data("Wetmore2012", package = "httk", envir = environment()),
  error = function(e) NULL
)

# Also check alternative data sets
invivo_df <- NULL

for (ds_name in c("Wetmore2012", "invivo.data", "chem.invivo.PK.data",
                  "chem.invivo.PK.aggregate.data")) {
  tryCatch({
    e <- new.env()
    data(list = ds_name, package = "httk", envir = e)
    obj <- get(ds_name, envir = e)
    if (is.data.frame(obj) && nrow(obj) > 0) {
      invivo_df <- obj
      cat("  Loaded dataset:", ds_name, "–", nrow(obj), "rows\n")
      break
    }
  }, error = function(e) NULL)
}

# Fall back: use get_lit_css() on pilot chemicals
pilot_csv <- file.path(DATA, "pilot_chemicals_imputed.csv")
if (!is.null(invivo_df) && nrow(invivo_df) > 0) {
  cat("  Using built-in in-vivo dataset.\n\n")
} else if (file.exists(pilot_csv)) {
  cat("  No built-in in-vivo dataset found – using get_lit_css() on pilot chemicals.\n\n")
  pilot <- read.csv(pilot_csv, stringsAsFactors = FALSE)
  cas_list <- pilot$CAS[!is.na(pilot$CAS)]

  # get_lit_css returns a data.frame or NULL per chemical
  lit_rows <- lapply(cas_list, function(cas) {
    tryCatch({
      css_lit <- get_lit_css(chem.cas = cas, suppress.messages = TRUE)
      if (!is.null(css_lit) && length(css_lit) > 0) {
        data.frame(CAS = cas, Css_lit_ugL = as.numeric(css_lit[1]),
                   stringsAsFactors = FALSE)
      } else NULL
    }, error = function(e) NULL)
  })
  lit_rows <- do.call(rbind, Filter(Negate(is.null), lit_rows))

  if (!is.null(lit_rows) && nrow(lit_rows) > 0) {
    invivo_df <- lit_rows
    cat("  Retrieved", nrow(invivo_df), "literature Css values via get_lit_css().\n\n")
  } else {
    cat("  WARNING: no literature Css available.  Generating synthetic validation\n")
    cat("           from reference 3compartmentss predictions (self-consistency check).\n\n")
    invivo_df <- NULL
  }
} else {
  invivo_df <- NULL
}

# ── 2. Identify chemicals and predict Css ────────────────────────────────────
cat("── 2. Predicting Css for literature chemicals …\n")

# Determine which CAS numbers are available in httk
all_cas <- get_cheminfo(suppress.messages = TRUE)

# Collect chemicals to validate
if (!is.null(invivo_df)) {
  # Identify the CAS column
  cas_col <- intersect(c("CAS", "CASRN", "cas"), colnames(invivo_df))
  if (length(cas_col) == 0) {
    cat("  WARNING: cannot identify CAS column in literature dataset.\n")
    cas_col <- colnames(invivo_df)[1]
  } else {
    cas_col <- cas_col[1]
  }
  val_cas <- unique(as.character(invivo_df[[cas_col]]))
  val_cas <- val_cas[val_cas %in% all_cas]
  cat("  Chemicals with HTTK parameters:", length(val_cas), "\n")
} else {
  # Synthetic self-consistency check: use pilot chemicals
  pilot <- read.csv(pilot_csv, stringsAsFactors = FALSE)
  val_cas <- pilot$CAS[pilot$CAS %in% all_cas]
  cat("  Using pilot chemicals for self-consistency check:", length(val_cas), "\n")
}

if (length(val_cas) == 0) {
  cat("ERROR: no chemicals available for validation.  Check httk installation.\n")
  quit(status = 1)
}

# Predict Css with two models
predict_css <- function(cas, model_name) {
  tryCatch(
    calc_css(
      chem.cas         = cas,
      model            = model_name,
      daily.dose       = 1,
      doses.per.day    = 1,
      output.units     = "uM",
      suppress.messages = TRUE
    )$css,
    error = function(e) NA_real_
  )
}

predictions <- do.call(rbind, lapply(val_cas, function(cas) {
  cmpd <- tryCatch(
    get_chem_id(chem.cas = cas, suppress.messages = TRUE)$chem.name,
    error = function(e) cas
  )
  data.frame(
    CAS        = cas,
    Compound   = cmpd,
    Css_3comp_uM = predict_css(cas, "3compartmentss"),
    Css_pbtk_uM  = predict_css(cas, "pbtk"),
    stringsAsFactors = FALSE
  )
}))

cat("  Predictions complete:", sum(!is.na(predictions$Css_3comp_uM)),
    "/ ", nrow(predictions), "for 3compartmentss\n")
cat("  Predictions complete:", sum(!is.na(predictions$Css_pbtk_uM)),
    "/ ", nrow(predictions), "for pbtk\n\n")

# ── 3. Merge with in-vivo / literature data ──────────────────────────────────
cat("── 3. Merging predictions with literature values …\n")

if (!is.null(invivo_df)) {
  # Identify Css column in literature
  css_candidates <- c("Css_lit_ugL", "Css", "css", "Css_uM", "css_uM",
                      "CSS", "plasma_css", "mean_css", "AUC", "Cmax")
  css_col <- intersect(css_candidates, colnames(invivo_df))[1]

  if (is.na(css_col)) css_col <- colnames(invivo_df)[ncol(invivo_df)]
  cat("  Using literature column:", css_col, "\n")

  merged <- merge(predictions, invivo_df[, c(cas_col, css_col)],
                  by.x = "CAS", by.y = cas_col, all.x = FALSE)
  colnames(merged)[colnames(merged) == css_col] <- "Css_lit"

  # Convert units if needed (rough heuristic: if max > 1000, likely µg/L)
  merged$Css_lit <- as.numeric(merged$Css_lit)
  merged <- merged[!is.na(merged$Css_lit) & merged$Css_lit > 0, ]

  # For plotting we need comparable units – convert predictions to µg/L
  # httk calc_css returns µM; convert: µg/L = µM * MW / 1000
  get_mw <- function(cas) {
    tryCatch(
      get_chem_id(chem.cas = cas, suppress.messages = TRUE)$MW,
      error = function(e) 300
    )
  }
  merged$MW <- sapply(merged$CAS, get_mw)
  merged$Css_3comp_ugL <- merged$Css_3comp_uM * merged$MW / 1000
  merged$Css_pbtk_ugL  <- merged$Css_pbtk_uM  * merged$MW / 1000

  # If literature already in µM, keep as is and use µM for all
  if (max(merged$Css_lit, na.rm = TRUE) < 100) {
    # Likely µM
    merged$Css_lit_uM <- merged$Css_lit
    merged$Css_3comp_cmp <- merged$Css_3comp_uM
    merged$Css_pbtk_cmp  <- merged$Css_pbtk_uM
    unit_label <- "µM"
  } else {
    # Likely µg/L
    merged$Css_lit_uM <- merged$Css_lit
    merged$Css_3comp_cmp <- merged$Css_3comp_ugL
    merged$Css_pbtk_cmp  <- merged$Css_pbtk_ugL
    unit_label <- "µg/L"
  }
} else {
  # Self-consistency: compare 3compartmentss vs. pbtk
  merged <- predictions[
    !is.na(predictions$Css_3comp_uM) & !is.na(predictions$Css_pbtk_uM), ]
  merged$Css_lit_uM    <- merged$Css_pbtk_uM   # "reference"
  merged$Css_3comp_cmp <- merged$Css_3comp_uM
  merged$Css_pbtk_cmp  <- merged$Css_pbtk_uM
  unit_label <- "µM"
  cat("  Self-consistency mode: 3compartmentss vs. pbtk as reference\n")
}

merged <- merged[is.finite(merged$Css_lit_uM) & merged$Css_lit_uM > 0 &
                   is.finite(merged$Css_3comp_cmp) & merged$Css_3comp_cmp > 0, ]
cat("  Matched chemicals for validation:", nrow(merged), "\n\n")

# ── 4. Compute validation metrics ────────────────────────────────────────────
cat("── 4. Computing validation statistics …\n")

compute_metrics <- function(obs, pred, model_name) {
  log_obs  <- log10(obs)
  log_pred <- log10(pred)
  ratio    <- pred / obs

  r2  <- cor(log_obs, log_pred)^2
  rmse <- sqrt(mean((log_pred - log_obs)^2))
  gmr  <- exp(mean(log(ratio)))
  fold2 <- mean(ratio >= 0.5 & ratio <= 2.0)
  fold3 <- mean(ratio >= 1/3 & ratio <= 3.0)
  fold10 <- mean(ratio >= 0.1 & ratio <= 10.0)

  cat(sprintf("  %-18s  R²=%.3f  RMSE(log)=%.3f  GMR=%.3f  %%2-fold=%.0f%%  %%3-fold=%.0f%%  %%10-fold=%.0f%%\n",
              model_name, r2, rmse, gmr, fold2*100, fold3*100, fold10*100))

  data.frame(
    Model    = model_name,
    N        = length(obs),
    R2       = round(r2,   3),
    RMSE_log = round(rmse, 3),
    GMR      = round(gmr,  3),
    Pct_2fold = round(fold2  * 100, 1),
    Pct_3fold = round(fold3  * 100, 1),
    Pct_10fold = round(fold10 * 100, 1)
  )
}

metrics_list <- list()
if (any(!is.na(merged$Css_3comp_cmp) & merged$Css_3comp_cmp > 0)) {
  sub3 <- merged[!is.na(merged$Css_3comp_cmp) & merged$Css_3comp_cmp > 0, ]
  metrics_list[["3compartmentss"]] <- compute_metrics(
    sub3$Css_lit_uM, sub3$Css_3comp_cmp, "3compartmentss"
  )
}
if (!is.null(invivo_df) && any(!is.na(merged$Css_pbtk_cmp) & merged$Css_pbtk_cmp > 0)) {
  subP <- merged[!is.na(merged$Css_pbtk_cmp) & merged$Css_pbtk_cmp > 0, ]
  metrics_list[["pbtk"]] <- compute_metrics(
    subP$Css_lit_uM, subP$Css_pbtk_cmp, "pbtk"
  )
}

metrics_df <- do.call(rbind, metrics_list)
metrics_csv <- file.path(RESULTS, "invivo_validation_metrics.csv")
write.csv(metrics_df, metrics_csv, row.names = FALSE)
cat("  Saved", metrics_csv, "\n\n")

# ── 5. Export full prediction table ──────────────────────────────────────────
out_df <- merged[, c("CAS", "Compound",
                      "Css_lit_uM", "Css_3comp_cmp", "Css_pbtk_cmp")]
colnames(out_df) <- c("CAS", "Compound",
                       paste0("Css_literature_", unit_label),
                       paste0("Css_3compartmentss_", unit_label),
                       paste0("Css_pbtk_", unit_label))
out_df$log10_lit   <- round(log10(out_df[[3]]), 3)
out_df$log10_3comp <- round(log10(out_df[[4]]), 3)
out_df$log10_pbtk  <- round(log10(out_df[[5]]), 3)
out_df$fold_3comp  <- round(out_df[[4]] / out_df[[3]], 3)
out_df$fold_pbtk   <- round(out_df[[5]] / out_df[[3]], 3)

val_csv <- file.path(RESULTS, "invivo_validation.csv")
write.csv(out_df, val_csv, row.names = FALSE)
cat("  Saved", val_csv, "\n\n")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
cat("── 6. Generating validation plots …\n")

# Helper: fold-error colour coding
fold_colour <- function(ratio) {
  ifelse(ratio >= 0.5 & ratio <= 2.0, "#2196F3",    # within 2-fold: blue
  ifelse(ratio >= 1/3 & ratio <= 3.0, "#4CAF50",    # within 3-fold: green
  ifelse(ratio >= 0.1 & ratio <= 10.0,"#FF9800",    # within 10-fold: orange
                                        "#F44336"))) # outside 10-fold: red
}

# ── Plot A: log-log scatter (3compartmentss vs. literature) ──────────────────
png(file.path(RESULTS, "invivo_validation_scatter.png"),
    width = 2400, height = 1200, res = 200)

par(mfrow = c(1, ifelse(!is.null(invivo_df), 2, 1)), mar = c(5,5,4,2))

# 3compartmentss
sub3 <- merged[!is.na(merged$Css_3comp_cmp) & merged$Css_3comp_cmp > 0 &
                 is.finite(merged$Css_lit_uM) & merged$Css_lit_uM > 0, ]
if (nrow(sub3) > 0) {
  lims <- range(log10(c(sub3$Css_lit_uM, sub3$Css_3comp_cmp)))
  lims <- c(floor(lims[1]) - 0.5, ceiling(lims[2]) + 0.5)
  col_pts <- fold_colour(sub3$Css_3comp_cmp / sub3$Css_lit_uM)
  plot(log10(sub3$Css_lit_uM), log10(sub3$Css_3comp_cmp),
       pch = 21, bg = col_pts, col = "black", cex = 1.4,
       xlim = lims, ylim = lims,
       xlab = paste0("log₁₀(Css literature)  [", unit_label, "]"),
       ylab = paste0("log₁₀(Css predicted)  [", unit_label, "]"),
       main = "3compartmentss vs. In-vivo Literature")
  abline(0, 1, lwd = 2)
  abline(log10(2),  1, lty = 2, col = "steelblue")
  abline(-log10(2), 1, lty = 2, col = "steelblue")
  abline(log10(3),  1, lty = 3, col = "darkgreen")
  abline(-log10(3), 1, lty = 3, col = "darkgreen")
  m3 <- metrics_list[["3compartmentss"]]
  if (!is.null(m3)) {
    legend("topleft", bty = "n", cex = 0.8,
           legend = c(
             sprintf("R² = %.3f", m3$R2),
             sprintf("RMSE(log) = %.3f", m3$RMSE_log),
             sprintf("GMR = %.3f", m3$GMR),
             sprintf("Within 2-fold: %.0f%%", m3$Pct_2fold)
           ))
  }
  legend("bottomright", bty = "n", cex = 0.75,
         legend = c("within 2-fold","within 3-fold","within 10-fold","outside"),
         pch = 21, pt.bg = c("#2196F3","#4CAF50","#FF9800","#F44336"),
         pt.cex = 1.2)
}

# PBTK (if in-vivo data available)
if (!is.null(invivo_df)) {
  subP <- merged[!is.na(merged$Css_pbtk_cmp) & merged$Css_pbtk_cmp > 0 &
                   is.finite(merged$Css_lit_uM) & merged$Css_lit_uM > 0, ]
  if (nrow(subP) > 0) {
    lims <- range(log10(c(subP$Css_lit_uM, subP$Css_pbtk_cmp)))
    lims <- c(floor(lims[1]) - 0.5, ceiling(lims[2]) + 0.5)
    col_pts <- fold_colour(subP$Css_pbtk_cmp / subP$Css_lit_uM)
    plot(log10(subP$Css_lit_uM), log10(subP$Css_pbtk_cmp),
         pch = 21, bg = col_pts, col = "black", cex = 1.4,
         xlim = lims, ylim = lims,
         xlab = paste0("log₁₀(Css literature)  [", unit_label, "]"),
         ylab = paste0("log₁₀(Css predicted)  [", unit_label, "]"),
         main = "PBTK vs. In-vivo Literature")
    abline(0, 1, lwd = 2)
    abline(log10(2),  1, lty = 2, col = "steelblue")
    abline(-log10(2), 1, lty = 2, col = "steelblue")
    abline(log10(3),  1, lty = 3, col = "darkgreen")
    abline(-log10(3), 1, lty = 3, col = "darkgreen")
    mP <- metrics_list[["pbtk"]]
    if (!is.null(mP)) {
      legend("topleft", bty = "n", cex = 0.8,
             legend = c(
               sprintf("R² = %.3f", mP$R2),
               sprintf("RMSE(log) = %.3f", mP$RMSE_log),
               sprintf("GMR = %.3f", mP$GMR),
               sprintf("Within 2-fold: %.0f%%", mP$Pct_2fold)
             ))
    }
  }
}

dev.off()
cat("  Saved", file.path(RESULTS, "invivo_validation_scatter.png"), "\n")

# ── Plot B: Residual analysis ─────────────────────────────────────────────────
png(file.path(RESULTS, "invivo_validation_residuals.png"),
    width = 2000, height = 1000, res = 200)

par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

# Histogram of log10 fold-errors (3compartmentss)
if (nrow(sub3) > 0) {
  log_ratio <- log10(sub3$Css_3comp_cmp / sub3$Css_lit_uM)
  hist(log_ratio, breaks = 20, col = "#2196F380", border = "white",
       xlab = "log₁₀(Predicted / Observed)",
       main = "Residual Distribution – 3compartmentss",
       xlim = c(-2, 2))
  abline(v = 0, lwd = 2, col = "red")
  abline(v = c(log10(2), -log10(2)), lty = 2, col = "steelblue")
  abline(v = c(log10(3), -log10(3)), lty = 3, col = "darkgreen")
  legend("topright", bty = "n", cex = 0.75,
         legend = c("y = 0", "±2-fold", "±3-fold"),
         lty = c(1,2,3), col = c("red","steelblue","darkgreen"), lwd = 2)
  mtext(sprintf("Bias (GMR): %.3f  |  Spread (RMSE_log): %.3f",
                10^mean(log_ratio), sqrt(mean(log_ratio^2))),
        side = 3, line = 0, cex = 0.7)
}

# Q-Q plot of log-errors
if (nrow(sub3) > 0) {
  qqnorm(log_ratio, pch = 21, bg = "#2196F3", col = "black",
         main = "Normal Q-Q – log₁₀(Pred/Obs)\n3compartmentss")
  qqline(log_ratio, col = "red", lwd = 1.5)
}

dev.off()
cat("  Saved", file.path(RESULTS, "invivo_validation_residuals.png"), "\n\n")

# ── 7. Summary ────────────────────────────────────────────────────────────────
cat("========================================================\n")
cat("Validation metrics summary:\n")
cat("========================================================\n")
print(metrics_df, row.names = FALSE)

cat("\nOutputs saved to results/\n")
cat("  invivo_validation.csv             – full predicted vs. observed\n")
cat("  invivo_validation_metrics.csv     – R², RMSE, GMR, fold-error fractions\n")
cat("  invivo_validation_scatter.png     – log-log scatter plots\n")
cat("  invivo_validation_residuals.png   – residual histogram + Q-Q\n")
cat("\nDone.\n")
