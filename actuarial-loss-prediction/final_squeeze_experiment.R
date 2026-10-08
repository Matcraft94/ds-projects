# Final squeeze round (2026-10-07): OOF-TE, median objective, monotone,
# tuned grid, ensemble — plus a stronger ceiling estimate.
# All arms on the FE2 feature set, same split/seed as every round.

suppressMessages({
  library(tidyverse); library(tidymodels); library(tidytext)
  library(SnowballC); library(stopwords); library(xgboost)
})
set.seed(42)
t0 <- proc.time()

raw <- read_csv("Data/actuarial_loss/train.csv", show_col_types = FALSE) %>% select(-ClaimNumber)
prep <- raw %>%
  mutate(Days_To_Report = as.numeric(difftime(DateReported, DateTimeOfAccident, units = "days")),
         AccidentYear = lubridate::year(DateTimeOfAccident))
set.seed(42)
idx <- sample(nrow(prep), size = 0.8 * nrow(prep))
train <- prep[idx, ]; test <- prep[setdiff(seq_len(nrow(prep)), idx), ]
train <- train %>% filter(!is.na(UltimateIncurredClaimCost))
test  <- test  %>% filter(!is.na(UltimateIncurredClaimCost))
y_tr <- train$UltimateIncurredClaimCost; y_te <- test$UltimateIncurredClaimCost
q_te <- ntile(y_te, 5)
n_tr <- nrow(train)
cat("train:", n_tr, " test:", nrow(test), "\n")

clean_text <- function(x) {
  x <- tolower(x); x <- gsub("[0-9]+", "", x)
  x <- gsub("[[:punct:]]", " ", x); x <- gsub("\\s+", " ", x); trimws(x)
}
train_txt <- train %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
test_txt  <- test  %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
tok <- function(d) d %>% unnest_tokens(word, txt) %>%
  anti_join(data.frame(word = stopwords::stopwords("en")), by = "word") %>%
  mutate(stem = SnowballC::wordStem(word))
tr_u <- tok(train_txt); te_u <- tok(test_txt)
tr_b <- train_txt %>% unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>% filter(!is.na(bigram))
te_b <- test_txt  %>% unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>% filter(!is.na(bigram))
vocab_u <- tr_u %>% count(stem, sort = TRUE) %>% filter(n >= 50) %>% slice_head(n = 200) %>% pull(stem)
vocab_b <- tr_b %>% count(bigram, sort = TRUE) %>% filter(n >= 50) %>% slice_head(n = 100) %>% pull(bigram)

yl_tr <- log1p(y_tr); gm <- mean(yl_tr); K <- 50

# ---- OOF target encoding (5 folds; maps fitted on out-of-fold data only) ----
set.seed(42); folds <- sample(rep(1:5, length.out = n_tr))
te_col <- function(tr_tokens, te_tokens, tr_ids, te_ids, colname, vocab) {
  mk_map <- function(ids_keep) {
    tr_tokens %>% filter(.data[[colname]] %in% vocab, claim_id %in% ids_keep) %>%
      left_join(tibble(claim_id = ids_keep, yl = yl_tr[ids_keep]), by = "claim_id") %>%
      group_by(.data[[colname]]) %>% summarise(m = mean(yl), .groups = "drop") %>%
      mutate(n = tr_tokens %>% filter(.data[[colname]] %in% vocab, claim_id %in% ids_keep) %>%
               count(.data[[colname]]) %>% pull(n)) %>%
      mutate(te = (n * m + K * gm) / (n + K))
  }
  oof <- numeric(n_tr)
  for (f in 1:5) {
    in_f  <- which(folds == f)
    out_f <- setdiff(seq_len(n_tr), in_f)
    mp <- mk_map(out_f)
    oof[in_f] <- tr_tokens %>% filter(claim_id %in% in_f, .data[[colname]] %in% vocab) %>%
      inner_join(mp, by = colname) %>% group_by(claim_id) %>% summarise(v = mean(te), .groups = "drop") %>%
      {.$v[match(in_f, .$claim_id)]}
  }
  oof[is.na(oof)] <- gm
  mp_full <- mk_map(seq_len(n_tr))
  tst <- te_tokens %>% filter(.data[[colname]] %in% vocab) %>%
    inner_join(mp_full, by = colname) %>% group_by(claim_id) %>% summarise(v = mean(te), .groups = "drop")
  te_v <- rep(gm, nrow(test_txt)); te_v[tst$claim_id] <- tst$v
  list(tr = oof, te = te_v)
}
teu <- te_col(tr_u, te_u, seq_len(n_tr), NULL, "stem", vocab_u)
teb <- te_col(tr_b, te_b, seq_len(n_tr), NULL, "bigram", vocab_b)

# Initial-percentile TE, OOF for train
ecdf_train <- function(ids) ecdf(train$InitialIncurredCalimsCost[ids])
init_pctl_map <- function(ids) {
  p <- ecdf_train(ids)(train$InitialIncurredCalimsCost[ids])
  tibble(yl = yl_tr[ids]) %>% mutate(b = pmin(ceiling(p * 100), 100)) %>%
    group_by(b) %>% summarise(te = mean(yl), .groups = "drop")
}
ip_oof <- numeric(n_tr)
for (f in 1:5) {
  in_f <- which(folds == f); out_f <- setdiff(seq_len(n_tr), in_f)
  mp <- init_pctl_map(out_f)
  p <- ecdf_train(out_f)(train$InitialIncurredCalimsCost[in_f])
  ip_oof[in_f] <- mp$te[pmin(ceiling(p * 100), 100)]
}
ip_full <- init_pctl_map(seq_len(n_tr))
p_te <- ecdf(train$InitialIncurredCalimsCost)(test$InitialIncurredCalimsCost)
ip_te <- ip_full$te[pmin(ceiling(p_te * 100), 100)]

# ---- features ----
tab_feats <- function(d, txt_df) d %>% transmute(
  Age, DependentChildren, DependentsOther, WeeklyWages, HoursWorkedPerWeek,
  DaysWorkedPerWeek, InitialIncurredCalimsCost, Days_To_Report, AccidentYear,
  WeeklyWagesPerHour = if_else(HoursWorkedPerWeek > 0, WeeklyWages / HoursWorkedPerWeek, 0),
  DependentsTotal = DependentChildren + DependentsOther,
  log_initial = log1p(InitialIncurredCalimsCost),
  init_zero = as.numeric(InitialIncurredCalimsCost == 0),
  init_x_delay = log1p(InitialIncurredCalimsCost) * log1p(pmax(Days_To_Report, 0)),
  word_count = strsplit(txt_df$txt, " ") |> map_int(length))
tr_num <- tab_feats(train, train_txt); te_num <- tab_feats(test, test_txt)
# two variants: in-sample TE (as round 3) and OOF TE
tr_is <- tr_num %>% mutate(init_pct_te = ip_full$te[pmin(ceiling(ecdf(train$InitialIncurredCalimsCost)(train$InitialIncurredCalimsCost) * 100), 100)],
                           te_stem = teu$tr, te_bigram = teb$tr)
te_f <- te_num %>% mutate(init_pct_te = ip_te, te_stem = teu$te, te_bigram = teb$te)
tr_oof <- tr_num %>% mutate(init_pct_te = ip_oof, te_stem = teu$tr, te_bigram = teb$tr)

DF_tr <- train %>% select(Gender, MaritalStatus, PartTimeFullTime) %>% mutate(across(everything(), ~ replace_na(.x, "NA")))
DF_te <- test %>% select(Gender, MaritalStatus, PartTimeFullTime) %>% mutate(across(everything(), ~ replace_na(.x, "NA")))
for (v in names(DF_tr)) {
  lv <- levels(factor(DF_tr[[v]])); DF_tr[[v]] <- factor(DF_tr[[v]], lv); DF_te[[v]] <- factor(DF_te[[v]], lv)
}
mm_tr <- model.matrix(~ . - 1, DF_tr); mm_te <- model.matrix(~ . - 1, DF_te)
colnames(mm_tr) <- paste0("c_", colnames(mm_tr)); colnames(mm_te) <- paste0("c_", colnames(mm_te))
meds <- map_dbl(tr_is, ~ median(.x, na.rm = TRUE))
fill_med <- function(d) map2_dfc(d, meds, ~ if_else(is.na(.x), .y, .x))
X_is  <- cbind(as.matrix(fill_med(tr_is)), mm_tr); X_te <- cbind(as.matrix(fill_med(te_f)), mm_te)
X_oof <- cbind(as.matrix(fill_med(tr_oof)), mm_tr)
stopifnot(all(colnames(X_is) == colnames(X_te)), all(colnames(X_oof) == colnames(X_te)))

# ---- ceiling v2: Initial-percentile x stem-TE-bucket interaction ----
g1 <- ceiling(ecdf(train$InitialIncurredCalimsCost)(train$InitialIncurredCalimsCost) * 20)
g2 <- ceiling(rank(tr_is$te_stem) / n_tr * 20)
v_total <- var(log1p(y_tr))
ceiling2 <- 1 - (sum(tapply(log1p(y_tr), paste(g1, g2), var) * as.numeric(table(paste(g1, g2))), na.rm = TRUE) /
                 (n_tr * v_total))
cat(sprintf("\nCEILING v2: Initial(20 pct-buckets) alone: %.3f | x stemTE(20): %.3f\n",
            1 - sum(tapply(log1p(y_tr), g1, var) * as.numeric(table(g1))) / (n_tr * v_total), ceiling2))

# ---- arms ----
metrics_dollars <- function(pred) {
  e <- pred - y_te
  tibble(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)),
         MAPE = mean(abs(e) / pmax(y_te, 1)) * 100,
         R2 = 1 - sum(e^2) / sum((y_te - mean(y_te))^2),
         RMSLE = sqrt(mean((log1p(pmax(pred, 0)) - log1p(y_te))^2)),
         R2log = 1 - mean((log1p(pmax(pred, 0)) - log1p(y_te))^2) / v_total)
}
res <- list(); preds <- list()
base_p <- function(extra = NULL) {
  p <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 4,
            lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
  if (!is.null(extra)) for (k in names(extra)) p[[k]] <- extra[[k]]
  p
}
fit_pred <- function(Xtr, params, rounds = 3000) {
  set.seed(42)
  m <- xgb.train(params, xgb.DMatrix(Xtr, label = yl_tr), nrounds = rounds, verbose = 0)
  pmax(expm1(predict(m, X_te)), 0)
}
cat("\n== ARM 1: FE2 reference (in-sample TE) ==\n")
preds$A1 <- fit_pred(X_is, base_p()); res$A1 <- metrics_dollars(preds$A1) %>% mutate(arm = "A1_FE2_ref")
cat("== ARM 2: OOF target encoding ==\n")
preds$A2 <- fit_pred(X_oof, base_p()); res$A2 <- metrics_dollars(preds$A2) %>% mutate(arm = "A2_OOF_TE")
cat("== ARM 3: median objective (quantile 0.5 in log space) ==\n")
qobj <- tryCatch({
  set.seed(42)
  m <- xgb.train(list(objective = "reg:quantileerror", quantile_alpha = 0.5, eta = 0.03,
                      max_depth = 4, lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8),
                 xgb.DMatrix(X_oof, label = yl_tr), nrounds = 3000, verbose = 0)
  pmax(expm1(predict(m, X_te)), 0)
}, error = function(e) { cat("quantile obj failed:", conditionMessage(e), "\n"); NULL })
if (!is.null(qobj)) { preds$A3 <- qobj; res$A3 <- metrics_dollars(qobj) %>% mutate(arm = "A3_median") }
cat("== ARM 4: monotone constraint on log_initial ==\n")
mc <- setNames(rep(0, ncol(X_oof)), colnames(X_oof)); mc["log_initial"] <- 1; mc["init_pct_te"] <- 1
preds$A4 <- fit_pred(X_oof, base_p(list(monotone_constraints = mc))); res$A4 <- metrics_dollars(preds$A4) %>% mutate(arm = "A4_monotone")
cat("== ARM 5: tuned grid (internal validation, log-RMSE) ==\n")
set.seed(42); vi <- sample(n_tr, 0.15 * n_tr)
dsub <- xgb.DMatrix(X_oof[-vi, ], label = yl_tr[-vi]); dval <- xgb.DMatrix(X_oof[vi, ], label = yl_tr[vi])
best <- NULL
for (eta in c(0.02, 0.03, 0.05)) for (dp in c(3, 4, 6)) {
  set.seed(42)
  m <- xgb.train(base_p(list(eta = eta, max_depth = dp)), dsub, nrounds = 3000, verbose = 0)
  pv <- predict(m, dval)
  sc <- sqrt(mean((pv - yl_tr[vi])^2))
  cat(sprintf("  eta=%.2f depth=%d -> val_logRMSE=%.5f\n", eta, dp, sc))
  if (is.null(best) || sc < best$sc) best <- list(sc = sc, it = 3000, eta = eta, dp = dp)
}
cat(sprintf("  tuned: eta=%.2f depth=%d rounds=%d val=%.5f\n", best$eta, best$dp, best$it, best$sc))
preds$A5 <- fit_pred(X_oof, base_p(list(eta = best$eta, max_depth = best$dp)), rounds = best$it)
res$A5 <- metrics_dollars(preds$A5) %>% mutate(arm = "A5_tuned")
cat("== ARM 6: ensemble (mean of A2..A5) ==\n")
ens_names <- intersect(c("A2", "A3", "A4", "A5"), names(preds))
preds$A6 <- Reduce(`+`, preds[ens_names]) / length(ens_names)
res$A6 <- metrics_dollars(preds$A6) %>% mutate(arm = "A6_ensemble")

tab <- bind_rows(res)
print(tab, n = 10)
cat("\nMAE by test quintile:\n")
for (nm in names(preds)) cat(sprintf("  %-3s %s\n", nm,
  paste(sprintf("%.0f", tapply(abs(preds[[nm]] - y_te), q_te, mean)), collapse = " ")))
for (nm in names(res)) cat(sprintf('FINAL_METRICS arm=%s rmse=%.4f mae=%.4f mape=%.4f r2=%.6f rmsle=%.6f r2log=%.6f\n',
  nm, res[[nm]]$RMSE, res[[nm]]$MAE, res[[nm]]$MAPE, res[[nm]]$R2, res[[nm]]$RMSLE, res[[nm]]$R2log))
cat(sprintf("elapsed_min=%.1f\n", (proc.time() - t0)[3] / 60))
