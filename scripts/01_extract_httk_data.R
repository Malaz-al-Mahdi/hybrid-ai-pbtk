###############################################################################
# 01_extract_httk_data.R
# ---------------------------------------------------------------------------
# Extracts physicochemical + TK data for ~20 pilot chemicals from the httk
# package.  Outputs two CSVs:
#   data/pilot_chemicals_full.csv   – all columns, complete cases (training)
#   data/pilot_chemicals_masked.csv – same rows but with Clint set to NA for
#                                     leave-one-out style external validation
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

httk_installed <- any(file.exists(file.path(.libPaths(), "httk")))
if (!httk_installed) {
  install.packages("httk", repos = "https://cloud.r-project.org", lib = r_user_lib)
}
library(httk, lib.loc = .libPaths())

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
project_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
data_dir <- file.path(project_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 1.  Define the 20 pilot chemicals (CAS numbers) ----------------------
#
# Selection rationale:
#   - Bisphenol A (required)
#   - Mix of pharmaceuticals, pesticides, industrial chemicals
#   - All must have Clint AND Fup data in httk so we can validate RF later
#   - Diverse MW / logP range for a meaningful feature space

pilot_cas <- c(

  "80-05-7",      # Bisphenol A

  "34256-82-1",   # Acetochlor
  "99-71-8",      # 4-sec-Butylphenol
  "58-08-2",      # Caffeine
  "298-46-4",     # Carbamazepine
  "2921-88-2",    # Chlorpyrifos
  "138261-41-3",  # Imidacloprid
  "87-86-5",      # Pentachlorophenol
  "62-44-2",      # Phenacetin
  "57-41-0",      # Phenytoin
  "94-75-7",      # 2,4-D
  "1912-24-9",    # Atrazine
  "330-54-1",     # Diuron
  "1071-83-6",    # Glyphosate
  "15307-79-6",   # Diclofenac sodium
  "137-26-8",     # Thiram
  "52-68-6",      # Trichlorfon
  "2104-64-5",    # EPN
  "62-73-7",      # Dichlorvos
  "56-72-4"       # Coumaphos
)

# ---- 2.  Query httk for complete pilot data --------------------------------
#
# `get_cheminfo(info = "all", median.only = TRUE)` already returns the exact
# summary variables needed for this pilot study. Using it directly is more
# robust than rebuilding each row via `parameterize_steadystate()`.

all_info <- suppressWarnings(get_cheminfo(info = "all", median.only = TRUE))

# Keep requested CAS numbers in the original order.
pilot_df <- all_info[match(pilot_cas, all_info$CAS), ]
pilot_df <- pilot_df[!is.na(pilot_df$CAS), ]

cat(sprintf(
  "Requested %d pilot chemicals, found %d with complete httk data.\n",
  length(pilot_cas), nrow(pilot_df)
))

missing_cas <- setdiff(pilot_cas, pilot_df$CAS)
if (length(missing_cas) > 0) {
  message("Chemicals skipped because current httk complete-case filters exclude them:")
  message(paste("  -", missing_cas, collapse = "\n"))
}

if (nrow(pilot_df) == 0) {
  stop("No pilot chemicals found with complete httk data.")
}

# ---- 3.  Standardize columns for downstream ML / PBTK ----------------------

enriched_df <- data.frame(
  CAS           = pilot_df$CAS,
  Compound      = pilot_df$Compound,
  MW            = as.numeric(pilot_df$MW),
  logP          = as.numeric(pilot_df$logP),
  Fup           = as.numeric(pilot_df$Human.Funbound.plasma),
  Clint         = as.numeric(pilot_df$Human.Clint),
  Rblood2plasma = as.numeric(pilot_df$Human.Rblood2plasma),
  stringsAsFactors = FALSE
)

cat(sprintf("Enriched data for %d chemicals.\n", nrow(enriched_df)))
print(enriched_df)

# ---- 4.  Export full dataset -----------------------------------------------
write.csv(enriched_df,
          file = file.path(data_dir, "pilot_chemicals_full.csv"),
          row.names = FALSE)

cat("Saved  data/pilot_chemicals_full.csv\n")

# ---- 5.  Create masked version (Clint = NA) for ML prediction test ---------
masked_df        <- enriched_df
masked_df$Clint  <- NA

write.csv(masked_df,
          file = file.path(data_dir, "pilot_chemicals_masked.csv"),
          row.names = FALSE)

cat("Saved  data/pilot_chemicals_masked.csv\n")
cat("Done. Proceed to 02_rf_predict_clint.py\n")
