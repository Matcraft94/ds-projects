# Evaluation: dollar-space metrics, quintile/decile segment errors.

library(tidyverse)

metrics_dollars <- function(pred, actual) {
  e <- pred - actual
  tibble(
    RMSE  = sqrt(mean(e^2)),
    MAE   = mean(abs(e)),
    MAPE  = mean(abs(e) / pmax(actual, 1)) * 100,
    R2    = 1 - sum(e^2) / sum((actual - mean(actual))^2),
    RMSLE = sqrt(mean((log1p(pmax(pred, 0)) - log1p(actual))^2))
  )
}

r2_log <- function(pred, actual) {
  v <- var(log1p(actual))
  1 - mean((log1p(pmax(pred, 0)) - log1p(actual))^2) / v
}

seg_mae <- function(pred, actual, groups) {
  tapply(abs(pred - actual), groups, mean)
}

save_round <- function(round_id, arm_metrics, segments, dir = "results") {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  metrics_csv <- file.path(dir, sprintf("metrics_%s.csv", round_id))
  segments_csv <- file.path(dir, sprintf("segments_%s.csv", round_id))
  write_csv(arm_metrics, metrics_csv)
  map2_dfr(names(segments), segments,
           ~ tibble(arm = .x, segment = names(.y), mae = as.numeric(.y))) %>%
    write_csv(segments_csv)
  invisible(list(metrics = metrics_csv, segments = segments_csv))
}
