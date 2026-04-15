###############################################################################
# 05_full_rtk_aed_ber.R
# ---------------------------------------------------------------------------
# Complete Reverse Toxicokinetics and Risk Prioritization workflow:
#
#   1. Builds the full chemical universe that is parameterizable with
#      3compartmentss and mapped to ToxCast.
#   2. Trains a Random Forest on the broader Wambaugh et al. (2019) HTTK
#      training set and imputes missing/zero Clint values for all eligible
#      chemicals.
#   3. Retrieves bioactivity concentrations from ToxCast and derives a
#      conservative AC50 5th percentile target concentration.
#   4. Computes Administered Equivalent Doses (AED) with calc_mc_oral_equiv()
#      using the best available Clint value for each chemical.
#   5. Merges real-world exposure estimates with priority NHANES biomonitoring
#      intake rates, then SEEM3 predictions, then example.seem as a fallback.
#   6. Calculates Bioactivity-Exposure Ratios (BER) and exports rankings for
#      risk prioritization.
#
# Outputs:
#   data/all_777_chemicals.csv      - eligible chemicals with Clint/exposure data
#   results/aed_ber_full.csv        - AED + BER for all successfully simulated chemicals
#   results/aed_ber_summary.csv     - top-priority chemicals ranked by BER
#   results/ber_ranking_plot.png    - BER waterfall ranking plot
###############################################################################

# ---- Library bootstrap -----------------------------------------------------
r_ver <- paste(
  R.version$major,
  strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1],
  sep = "."
)
r_user_lib <- Sys.getenv("R_LIBS_USER", unset = "")
if (!nzchar(r_user_lib)) {
  r_user_lib <- file.path(
    Sys.getenv("USERPROFILE"),
    "Documents",
    "R",
    "win-library",
    r_ver
  )
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
ensure_pkg("httk")
ensure_pkg("randomForest")

library(httk, lib.loc = .libPaths())
library(randomForest, lib.loc = .libPaths())

# ---- Resolve project paths -------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
project_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
data_dir <- file.path(project_dir, "data")
results_dir <- file.path(project_dir, "results")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Helpers ---------------------------------------------------------------
first_numeric <- function(x) {
  if (length(x) == 0 || is.null(x) || is.na(x[1])) {
    return(NA_real_)
  }
  if (is.numeric(x)) {
    return(as.numeric(x[1]))
  }

  txt <- trimws(as.character(x[1]))
  if (!nzchar(txt)) {
    return(NA_real_)
  }

  txt <- gsub(";", ",", txt, fixed = TRUE)
  txt <- gsub("\\s+", "", txt)
  vals <- suppressWarnings(as.numeric(unlist(strsplit(txt, ",", fixed = TRUE))))
  vals <- vals[!is.na(vals)]

  if (length(vals) == 0) {
    NA_real_
  } else {
    vals[1]
  }
}

num_or_na <- function(x) {
  suppressWarnings(as.numeric(x))
}

choose_existing_col <- function(df, candidates) {
  hits <- candidates[candidates %in% names(df)]
  if (length(hits) == 0) {
    return(NULL)
  }
  hits[1]
}

get_httk_data <- function(name) {
  tryCatch(get(name, envir = asNamespace("httk"), inherits = FALSE),
           error = function(e) NULL)
}

classify_ber <- function(ber) {
  if (is.na(ber)) {
    return("no_exposure_data")
  }
  if (ber < 1) {
    return("HIGH (BER<1)")
  }
  if (ber < 10) {
    return("MODERATE (1<BER<10)")
  }
  if (ber < 100) {
    return("LOW (10<BER<100)")
  }
  "MINIMAL (BER>100)"
}

impute_with_medians <- function(df, feature_cols, medians) {
  out <- df
  for (col in feature_cols) {
    bad <- is.na(out[[col]]) | !is.finite(out[[col]])
    out[[col]][bad] <- medians[[col]]
  }
  out
}

# ---- Keep a copy of the original HTTK database -----------------------------
httk_table_backup <- chem.physical_and_invitro.data

###############################################################################
# STEP 1: Build the eligible chemical universe
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 1: Building eligible chemical universe\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

all_3css <- suppressWarnings(
  get_cheminfo(info = "all", median.only = TRUE, model = "3compartmentss")
)
cat(sprintf("  3compartmentss-parametrisable chemicals: %d\n", nrow(all_3css)))

tc_flags <- vapply(
  all_3css$CAS,
  function(cas) tryCatch(is.toxcast(cas), error = function(e) FALSE),
  logical(1)
)
eligible <- all_3css[tc_flags, ]

cat(sprintf("  Also mapped to ToxCast:                %d\n\n", nrow(eligible)))

if (nrow(eligible) == 0) {
  stop("No eligible chemicals found for the full RTK pipeline.")
}

eligible$MW_num <- num_or_na(eligible$MW)
eligible$logP_num <- num_or_na(eligible$logP)
eligible$Fup_num <- vapply(eligible$Human.Funbound.plasma, first_numeric, numeric(1))
eligible$Rblood2plasma_num <- vapply(eligible$Human.Rblood2plasma, first_numeric, numeric(1))
eligible$Clint_native <- vapply(eligible$Human.Clint, first_numeric, numeric(1))

###############################################################################
# STEP 2: RF-impute missing/zero Clint using Wambaugh et al. (2019)
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 2: RF imputation for all-chemical Clint coverage\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

rf_training_raw <- get_httk_data("wambaugh2019")
if (is.null(rf_training_raw)) {
  stop("httk dataset 'wambaugh2019' is unavailable; cannot train the RF imputation model.")
}

rf_features <- c("MW", "logP", "Fup")
EPSILON <- 1e-3

rf_train <- data.frame(
  CAS = rf_training_raw$CAS,
  MW = num_or_na(rf_training_raw$MW),
  logP = num_or_na(rf_training_raw$logP),
  Fup = vapply(rf_training_raw$Human.Funbound.plasma, first_numeric, numeric(1)),
  Clint = vapply(rf_training_raw$Human.Clint, first_numeric, numeric(1)),
  stringsAsFactors = FALSE
)
rf_train <- rf_train[!is.na(rf_train$Clint) & rf_train$Clint > 0, ]

if (nrow(rf_train) < 25) {
  stop("Too few Wambaugh training rows available for reliable RF Clint imputation.")
}

rf_feature_medians <- lapply(rf_features, function(col) {
  median(rf_train[[col]][is.finite(rf_train[[col]])], na.rm = TRUE)
})
names(rf_feature_medians) <- rf_features

rf_train_x <- impute_with_medians(rf_train[, rf_features], rf_features, rf_feature_medians)
rf_train_y <- log10(rf_train$Clint + EPSILON)

set.seed(42)
rf_model <- randomForest(
  x = rf_train_x,
  y = rf_train_y,
  ntree = 500,
  mtry = max(1, floor(sqrt(length(rf_features)))),
  importance = TRUE
)

eligible_rf_x <- data.frame(
  MW = eligible$MW_num,
  logP = eligible$logP_num,
  Fup = eligible$Fup_num
)
eligible_rf_x <- impute_with_medians(eligible_rf_x, rf_features, rf_feature_medians)

eligible$Clint_rf <- pmax(10^predict(rf_model, newdata = eligible_rf_x) - EPSILON, 0)
eligible$needs_clint_impute <- is.na(eligible$Clint_native) | eligible$Clint_native <= 0
eligible$Clint_used <- ifelse(
  eligible$needs_clint_impute,
  eligible$Clint_rf,
  eligible$Clint_native
)
eligible$Clint_source <- ifelse(
  eligible$needs_clint_impute,
  "RF_imputed_wambaugh2019",
  "httk_measured"
)

cat(sprintf("  RF training chemicals used:             %d\n", nrow(rf_train)))
cat(sprintf("  Eligible chemicals needing imputation:  %d\n", sum(eligible$needs_clint_impute)))
cat(sprintf("  Chemicals with measured Clint retained: %d\n\n",
            sum(!eligible$needs_clint_impute)))

# Override only the chemicals that need imputation inside the in-memory httk DB.
override_rows <- eligible[
  eligible$needs_clint_impute & !is.na(eligible$Clint_used) & eligible$Clint_used > 0,
  c("Compound", "CAS", "DTXSID", "Clint_used")
]

if (nrow(override_rows) > 0) {
  override_rows$Clint_pValue <- 0
  chem.physical_and_invitro.data <- add_chemtable(
    override_rows,
    current.table = chem.physical_and_invitro.data,
    data.list = list(
      Compound = "Compound",
      CAS = "CAS",
      DTXSID = "DTXSID",
      Clint = "Clint_used",
      Clint.pValue = "Clint_pValue"
    ),
    species = "Human",
    reference = "RF Clint imputation from wambaugh2019",
    overwrite = TRUE
  )
}

###############################################################################
# STEP 3: Retrieve ToxCast bioactivity thresholds
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 3: Retrieving conservative ToxCast AC50 targets\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

tc_data <- httk::example.toxcast
eligible$n_tc_assays <- 0L
eligible$AC50_5pct_uM <- NA_real_
eligible$AC50_med_uM <- NA_real_
eligible$AC50_source <- NA_character_

for (i in seq_len(nrow(eligible))) {
  dtxsid <- eligible$DTXSID[i]
  if (is.na(dtxsid) || !nzchar(dtxsid)) {
    eligible$AC50_5pct_uM[i] <- 1.0
    eligible$AC50_med_uM[i] <- 1.0
    eligible$AC50_source[i] <- "default_1uM"
    next
  }

  hits <- tc_data[tc_data$dsstox_substance_id == dtxsid & tc_data$hitc == 1, ]
  if (nrow(hits) == 0) {
    eligible$AC50_5pct_uM[i] <- 1.0
    eligible$AC50_med_uM[i] <- 1.0
    eligible$AC50_source[i] <- "default_1uM"
    next
  }

  ac50_vals <- 10^hits$modl_acc
  ac50_vals <- ac50_vals[!is.na(ac50_vals) & ac50_vals > 0]

  if (length(ac50_vals) == 0) {
    eligible$AC50_5pct_uM[i] <- 1.0
    eligible$AC50_med_uM[i] <- 1.0
    eligible$AC50_source[i] <- "default_1uM"
    next
  }

  eligible$n_tc_assays[i] <- length(ac50_vals)
  eligible$AC50_5pct_uM[i] <- as.numeric(quantile(ac50_vals, 0.05))
  eligible$AC50_med_uM[i] <- median(ac50_vals)
  eligible$AC50_source[i] <- "example.toxcast"
}

cat(sprintf("  Chemicals with measured ToxCast hits:   %d\n",
            sum(eligible$AC50_source == "example.toxcast")))
cat(sprintf("  Chemicals using 1 uM fallback:          %d\n\n",
            sum(eligible$AC50_source == "default_1uM")))

###############################################################################
# STEP 4: Merge real-world exposure data (NHANES -> SEEM3 -> example.seem)
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 4: Loading exposure data for BER\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

nhanes_data <- get_httk_data("wambaugh2019.nhanes")
seem3_data <- get_httk_data("wambaugh2019.seem3")
example_seem <- get_httk_data("example.seem")

if (!is.null(nhanes_data)) {
  nhanes_data <- as.data.frame(nhanes_data)
}
if (!is.null(seem3_data)) {
  seem3_data <- as.data.frame(seem3_data)
}
if (!is.null(example_seem)) {
  example_seem <- as.data.frame(example_seem)
}

eligible$Exposure_median_mg_kg_day <- NA_real_
eligible$Exposure_l95_mg_kg_day <- NA_real_
eligible$Exposure_u95_mg_kg_day <- NA_real_
eligible$Exposure_source <- NA_character_
eligible$Exposure_pathway <- NA_character_

seem3_low_col <- if (!is.null(seem3_data)) {
  choose_existing_col(seem3_data, c("seem3.l95", "seem3_l95", "lP.min"))
} else {
  NULL
}
seem3_high_col <- if (!is.null(seem3_data)) {
  choose_existing_col(seem3_data, c("seem3.u95", "seem3_u95", "lP.max"))
} else {
  NULL
}
seem3_pathway_col <- if (!is.null(seem3_data)) {
  choose_existing_col(seem3_data, c("Pathway", "pathway"))
} else {
  NULL
}

example_low_col <- if (!is.null(example_seem)) {
  choose_existing_col(example_seem, c("seem3.l95", "seem3_l95"))
} else {
  NULL
}
example_high_col <- if (!is.null(example_seem)) {
  choose_existing_col(example_seem, c("seem3.u95", "seem3_u95"))
} else {
  NULL
}
example_pathway_col <- if (!is.null(example_seem)) {
  choose_existing_col(example_seem, c("Pathway", "pathway"))
} else {
  NULL
}

for (i in seq_len(nrow(eligible))) {
  cas <- eligible$CAS[i]

  if (!is.null(nhanes_data)) {
    nhanes_row <- nhanes_data[nhanes_data$CASRN == cas, , drop = FALSE]
    if (nrow(nhanes_row) > 0) {
      eligible$Exposure_median_mg_kg_day[i] <- first_numeric(nhanes_row$lP)
      eligible$Exposure_l95_mg_kg_day[i] <- first_numeric(nhanes_row$lP.min)
      eligible$Exposure_u95_mg_kg_day[i] <- first_numeric(nhanes_row$lP.max)
      eligible$Exposure_source[i] <- "NHANES_biomonitoring"
      next
    }
  }

  if (!is.null(seem3_data)) {
    seem_row <- seem3_data[seem3_data$CAS == cas, , drop = FALSE]
    if (nrow(seem_row) > 0 && "seem3" %in% names(seem3_data)) {
      eligible$Exposure_median_mg_kg_day[i] <- first_numeric(seem_row$seem3)
      if (!is.null(seem3_low_col)) {
        eligible$Exposure_l95_mg_kg_day[i] <- first_numeric(seem_row[[seem3_low_col]])
      }
      if (!is.null(seem3_high_col)) {
        eligible$Exposure_u95_mg_kg_day[i] <- first_numeric(seem_row[[seem3_high_col]])
      }
      if (!is.null(seem3_pathway_col)) {
        eligible$Exposure_pathway[i] <- as.character(seem_row[[seem3_pathway_col]][1])
      }
      eligible$Exposure_source[i] <- "SEEM3_modeled"
      next
    }
  }

  if (!is.null(example_seem)) {
    example_row <- example_seem[example_seem$CAS == cas, , drop = FALSE]
    if (nrow(example_row) > 0 && "seem3" %in% names(example_seem)) {
      eligible$Exposure_median_mg_kg_day[i] <- first_numeric(example_row$seem3)
      if (!is.null(example_low_col)) {
        eligible$Exposure_l95_mg_kg_day[i] <- first_numeric(example_row[[example_low_col]])
      }
      if (!is.null(example_high_col)) {
        eligible$Exposure_u95_mg_kg_day[i] <- first_numeric(example_row[[example_high_col]])
      }
      if (!is.null(example_pathway_col)) {
        eligible$Exposure_pathway[i] <- as.character(example_row[[example_pathway_col]][1])
      }
      eligible$Exposure_source[i] <- "example.seem"
    }
  }
}

cat(sprintf("  NHANES direct exposures available:      %d\n",
            sum(eligible$Exposure_source == "NHANES_biomonitoring", na.rm = TRUE)))
cat(sprintf("  SEEM3 modeled exposures available:      %d\n",
            sum(eligible$Exposure_source == "SEEM3_modeled", na.rm = TRUE)))
cat(sprintf("  example.seem fallback exposures:        %d\n",
            sum(eligible$Exposure_source == "example.seem", na.rm = TRUE)))
cat(sprintf("  Chemicals still without exposure data:  %d\n\n",
            sum(is.na(eligible$Exposure_source))))

write.csv(eligible, file.path(data_dir, "all_777_chemicals.csv"), row.names = FALSE)
cat(sprintf("  Saved data/all_777_chemicals.csv (%d rows)\n\n", nrow(eligible)))

###############################################################################
# STEP 5: Monte Carlo reverse dosimetry -> AED
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 5: Monte Carlo reverse dosimetry (AED)\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

set.seed(42)
aed_rows <- list()

for (i in seq_len(nrow(eligible))) {
  cas <- eligible$CAS[i]
  compound <- eligible$Compound[i]

  if (i %% 50 == 1 || i == nrow(eligible)) {
    cat(sprintf("[%3d/%d] %s ...\n", i, nrow(eligible), compound))
  }

  if (is.na(eligible$Clint_used[i]) || eligible$Clint_used[i] <= 0) {
    message(sprintf("  [%s] Skipping AED; no usable Clint after imputation.", cas))
    next
  }

  mc_samples <- tryCatch(
    calc_mc_oral_equiv(
      conc = eligible$AC50_5pct_uM[i],
      chem.cas = cas,
      species = "Human",
      which.quantile = 0.95,
      suppress.messages = TRUE,
      return.samples = TRUE,
      input.units = "uM",
      output.units = "mgpkgpday"
    ),
    error = function(e) {
      message(sprintf("  [%s] AED calculation failed: %s", cas, e$message))
      NULL
    }
  )

  if (is.null(mc_samples)) {
    next
  }

  aed_rows[[length(aed_rows) + 1]] <- data.frame(
    CAS = cas,
    Compound = compound,
    DTXSID = eligible$DTXSID[i],
    Clint_native = eligible$Clint_native[i],
    Clint_rf = eligible$Clint_rf[i],
    Clint_used = eligible$Clint_used[i],
    Clint_source = eligible$Clint_source[i],
    AC50_5pct_uM = eligible$AC50_5pct_uM[i],
    AC50_med_uM = eligible$AC50_med_uM[i],
    AC50_source = eligible$AC50_source[i],
    n_tc_assays = eligible$n_tc_assays[i],
    AED_median = median(mc_samples),
    AED_5pct = as.numeric(quantile(mc_samples, 0.05)),
    AED_95pct = as.numeric(quantile(mc_samples, 0.95)),
    AED_mean = mean(mc_samples),
    stringsAsFactors = FALSE
  )
}

if (length(aed_rows) == 0) {
  chem.physical_and_invitro.data <- httk_table_backup
  stop("No AED calculations succeeded.")
}

aed_df <- do.call(rbind, aed_rows)
rownames(aed_df) <- NULL

cat(sprintf("\n  AED calculated for %d chemicals.\n\n", nrow(aed_df)))

###############################################################################
# STEP 6: BER calculation and prioritization
###############################################################################

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 6: BER calculation and prioritization\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

row_idx <- match(aed_df$CAS, eligible$CAS)
aed_df$Exposure_median_mg_kg_day <- eligible$Exposure_median_mg_kg_day[row_idx]
aed_df$Exposure_l95_mg_kg_day <- eligible$Exposure_l95_mg_kg_day[row_idx]
aed_df$Exposure_u95_mg_kg_day <- eligible$Exposure_u95_mg_kg_day[row_idx]
aed_df$Exposure_source <- eligible$Exposure_source[row_idx]
aed_df$Exposure_pathway <- eligible$Exposure_pathway[row_idx]

# Primary BER uses the protective AED_95pct relative to the median real-world exposure.
aed_df$BER <- ifelse(
  !is.na(aed_df$Exposure_median_mg_kg_day) & aed_df$Exposure_median_mg_kg_day > 0,
  aed_df$AED_95pct / aed_df$Exposure_median_mg_kg_day,
  NA_real_
)

# Optional BER bounds using exposure and AED uncertainty envelopes.
aed_df$BER_low <- ifelse(
  !is.na(aed_df$Exposure_u95_mg_kg_day) & aed_df$Exposure_u95_mg_kg_day > 0,
  aed_df$AED_5pct / aed_df$Exposure_u95_mg_kg_day,
  NA_real_
)
aed_df$BER_high <- ifelse(
  !is.na(aed_df$Exposure_l95_mg_kg_day) & aed_df$Exposure_l95_mg_kg_day > 0,
  aed_df$AED_95pct / aed_df$Exposure_l95_mg_kg_day,
  NA_real_
)
aed_df$concern <- vapply(aed_df$BER, classify_ber, character(1))

order_idx <- order(is.na(aed_df$BER), aed_df$BER, aed_df$AED_95pct)
aed_df <- aed_df[order_idx, ]
rownames(aed_df) <- NULL

write.csv(aed_df, file.path(results_dir, "aed_ber_full.csv"), row.names = FALSE)
cat(sprintf("  Saved results/aed_ber_full.csv (%d rows)\n", nrow(aed_df)))

ranked_ber <- aed_df[!is.na(aed_df$BER), ]
summary_n <- min(30, nrow(ranked_ber))
ber_summary <- ranked_ber[seq_len(summary_n), , drop = FALSE]
write.csv(ber_summary, file.path(results_dir, "aed_ber_summary.csv"), row.names = FALSE)
cat(sprintf("  Saved results/aed_ber_summary.csv (%d rows)\n\n", nrow(ber_summary)))

cat(sprintf("  Chemicals with BER coverage:            %d\n", nrow(ranked_ber)))
cat(sprintf("  BER < 1  (HIGH concern):                %d\n", sum(ranked_ber$BER < 1, na.rm = TRUE)))
cat(sprintf("  BER < 10 (MODERATE+ concern):           %d\n", sum(ranked_ber$BER < 10, na.rm = TRUE)))
cat(sprintf("  BER < 100:                              %d\n", sum(ranked_ber$BER < 100, na.rm = TRUE)))
if (nrow(ranked_ber) > 0) {
  cat(sprintf("  BER range: %.2e -- %.2e\n\n",
              min(ranked_ber$BER, na.rm = TRUE),
              max(ranked_ber$BER, na.rm = TRUE)))
}

if (nrow(ber_summary) > 0) {
  cat("  Top chemicals by BER (lowest = highest concern):\n")
  cat(sprintf("  %-24s %10s %12s %12s %18s\n",
              "Compound", "AED_95pct", "Exposure", "BER", "Exposure source"))
  cat(paste("  ", paste(rep("-", 92), collapse = ""), "\n"))
  for (j in seq_len(min(20, nrow(ber_summary)))) {
    cat(sprintf("  %-24s %10.4f %12.2e %12.2f %18s\n",
                substr(ber_summary$Compound[j], 1, 24),
                ber_summary$AED_95pct[j],
                ber_summary$Exposure_median_mg_kg_day[j],
                ber_summary$BER[j],
                substr(ber_summary$Exposure_source[j], 1, 18)))
  }
  cat("\n")
}

tryCatch({
  plot_df <- head(ranked_ber, 40)
  if (nrow(plot_df) == 0) {
    stop("No BER-ranked chemicals available for plotting.")
  }

  png(file.path(results_dir, "ber_ranking_plot.png"), width = 1100, height = 650)
  par(mar = c(11, 5, 3, 2))

  plot_colors <- ifelse(
    plot_df$BER < 1,
    "firebrick",
    ifelse(plot_df$BER < 10, "darkorange",
           ifelse(plot_df$BER < 100, "goldenrod", "steelblue"))
  )

  barplot(
    log10(plot_df$BER),
    names.arg = substr(plot_df$Compound, 1, 16),
    col = plot_colors,
    las = 2,
    cex.names = 0.65,
    ylab = "log10(BER)",
    main = "Bioactivity-Exposure Ratio (BER) ranking\nLower BER indicates higher concern"
  )
  abline(h = 0, lty = 2, col = "firebrick", lwd = 2)    # BER = 1
  abline(h = 1, lty = 2, col = "darkorange", lwd = 1)   # BER = 10
  abline(h = 2, lty = 2, col = "goldenrod", lwd = 1)    # BER = 100
  legend(
    "topright",
    legend = c("BER < 1", "1-10", "10-100", ">100"),
    fill = c("firebrick", "darkorange", "goldenrod", "steelblue"),
    cex = 0.8
  )
  dev.off()
  cat("  Saved results/ber_ranking_plot.png\n")
}, error = function(e) {
  message(sprintf("  Plot skipped: %s", e$message))
})

# Restore the original HTTK database state before exiting.
chem.physical_and_invitro.data <- httk_table_backup

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Pipeline complete.\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
