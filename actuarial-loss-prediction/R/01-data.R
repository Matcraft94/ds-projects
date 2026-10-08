# Shared data layer — single source of truth for every experiment.
# All functions are pure w.r.t. their inputs; statistics are train-only.

library(tidyverse)

DATA_PATH <- "Data/actuarial_loss/train.csv"
SPLIT_SEED <- 42

load_raw <- function(path = DATA_PATH) {
  readr::read_csv(path, show_col_types = FALSE) %>% select(-ClaimNumber)
}

add_derived <- function(raw) {
  raw %>%
    mutate(
      Days_To_Report = as.numeric(difftime(DateReported, DateTimeOfAccident, units = "days")),
      AccidentYear = lubridate::year(DateTimeOfAccident)
    )
}

# The split used by EVERY experiment and the published re-run check.
# Stratification variable computed but the sample is plain (matches all runs
# in this repo; changing this changes every number downstream).
make_split <- function(prep, seed = SPLIT_SEED) {
  set.seed(seed)
  idx <- sample(nrow(prep), size = 0.8 * nrow(prep))
  train <- prep[idx, ]
  test  <- prep[setdiff(seq_len(nrow(prep)), idx), ]
  # rows without a measurable target cannot be scored; drop and report
  train <- train %>% filter(!is.na(UltimateIncurredClaimCost))
  test  <- test  %>% filter(!is.na(UltimateIncurredClaimCost))
  list(train = train, test = test, n_dropped = sum(is.na(prep$UltimateIncurredClaimCost)))
}

clean_text <- function(x) {
  x <- tolower(x)
  x <- gsub("[0-9]+", "", x)
  x <- gsub("[[:punct:]]", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}
