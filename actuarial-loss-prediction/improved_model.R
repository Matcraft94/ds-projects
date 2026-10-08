# Improved actuarial model — clean-protocol experiment (2026-10-07)
# Compares, in ONE environment, same data, same seeds:
#   A) baseline-clean : original design (squared error on dollars, count text features)
#                       but with the audit fixes: seeded split FIRST, no target imputation
#   B) improved-log   : log1p target + TF-IDF (unigram stems + bigrams, train-only vocab)
#   C) improved-tweed : Tweedie objective (insurance standard) + TF-IDF
# All arms: mini-grid eta x depth, early stopping on an internal validation split,
# final refit on full train, single evaluation on the untouched test set.
# Metrics reported in DOLLARS: RMSE, MAE, MAPE, R2, RMSLE, MAE by target quintile.

suppressMessages({
  library(tidyverse); library(tidymodels); library(tidytext)
  library(SnowballC); library(stopwords); library(xgboost); library(moments)
})
tidymodels_prefer()
set.seed(42)

cat("== LOAD ==\n")
raw <- read_csv("Data/actuarial_loss/train.csv", show_col_types = FALSE) %>% select(-ClaimNumber)
cat("rows:", nrow(raw), "\n")

# ---------- shared preprocessing (no target info used) ----------
prep <- raw %>%
  mutate(
    Days_To_Report = as.numeric(difftime(DateReported, DateTimeOfAccident, units = "days")),
    AccidentYear = lubridate::year(DateTimeOfAccident)
  )

n_na_target <- sum(is.na(prep$UltimateIncurredClaimCost))
cat("rows with NA target (dropped from train/test scoring):", n_na_target, "\n")

# ---------- 1. SPLIT FIRST (seeded, stratified on cost decile) ----------
split_var <- prep %>%
  mutate(cost_bin = ntile(UltimateIncurredClaimCost, 10)) %>%
  pull(cost_bin)
set.seed(42)
idx    <- sample(nrow(prep), size = 0.8 * nrow(prep))
tr_rows <- idx; te_rows <- setdiff(seq_len(nrow(prep)), idx)
train <- prep[tr_rows, ]; test <- prep[te_rows, ]
cat("train:", nrow(train), " test:", nrow(test), "\n")

# rows with NA target cannot be scored -> drop for modelling (kept counted)
train <- train %>% filter(!is.na(UltimateIncurredClaimCost))
test  <- test  %>% filter(!is.na(UltimateIncurredClaimCost))

# ---------- 2. TEXT FEATURES ----------
clean_text <- function(x) {
  x <- tolower(x); x <- gsub("[0-9]+", "", x)
  x <- gsub("[[:punct:]]", " ", x); x <- gsub("\\s+", " ", x); trimws(x)
}
train_txt <- train %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
test_txt  <- test  %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))

tok <- function(d) d %>% unnest_tokens(word, txt) %>%
  anti_join(data.frame(word = stopwords::stopwords("en")), by = "word") %>%
  mutate(stem = SnowballC::wordStem(word))

tr_u <- tok(train_txt)
vocab_u <- tr_u %>% count(stem, sort = TRUE) %>% filter(n >= 50) %>% slice_head(n = 200) %>% pull(stem)
tr_b <- train_txt %>% unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>% filter(!is.na(bigram))
vocab_b <- tr_b %>% count(bigram, sort = TRUE) %>% filter(n >= 50) %>% slice_head(n = 100) %>% pull(bigram)

# tf-idf (idf computed on TRAIN only)
doc_n <- nrow(train_txt)
idf_u <- tr_u %>% filter(stem %in% vocab_u) %>% distinct(claim_id, stem) %>%
  count(stem) %>% rename(df_doc = n) %>% mutate(idf = log(doc_n / df_doc))
idf_b <- tr_b %>% filter(bigram %in% vocab_b) %>% distinct(claim_id, bigram) %>%
  count(bigram) %>% rename(df_doc = n) %>% mutate(idf = log(doc_n / df_doc))

pad_rows <- function(wide, txt_df) {
  full <- tibble(claim_id = seq_len(nrow(txt_df)))
  out <- full %>% left_join(wide, by = "claim_id") %>% select(-claim_id)
  out[is.na(out)] <- 0
  as.data.frame(out)
}
tfidf_mat <- function(txt_df, vocab_u, vocab_b, idf_u, idf_b) {
  ttl <- txt_df %>% group_by(claim_id) %>% summarise(ntot = sum(strsplit(txt, " ") |> map_int(length)), .groups = "drop")
  u <- tok(txt_df) %>% filter(stem %in% vocab_u) %>%
    count(claim_id, stem, name = "n") %>% left_join(ttl, by = "claim_id") %>%
    left_join(idf_u, by = "stem") %>% mutate(v = (n / ntot) * idf) %>%
    pivot_wider(id_cols = claim_id, names_from = stem, values_from = v, values_fill = 0, names_prefix = "CD_")
  b <- txt_df %>% unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>% filter(!is.na(bigram), bigram %in% vocab_b) %>%
    count(claim_id, bigram, name = "n") %>% left_join(ttl, by = "claim_id") %>%
    left_join(idf_b, by = "bigram") %>% mutate(v = (n / ntot) * idf) %>%
    pivot_wider(id_cols = claim_id, names_from = bigram, values_from = v, values_fill = 0, names_prefix = "CB_")
  out <- u %>% full_join(b, by = "claim_id")
  pad_rows(out, txt_df)
}
cnt_mat <- function(txt_df, vocab_u) {  # original-style raw counts
  out <- tok(txt_df) %>% filter(stem %in% vocab_u) %>% count(claim_id, stem, name = "n") %>%
    pivot_wider(id_cols = claim_id, names_from = stem, values_from = n, values_fill = 0, names_prefix = "CD_")
  pad_rows(out, txt_df)
}

T_tfidf_tr <- tfidf_mat(train_txt, vocab_u, vocab_b, idf_u, idf_b)
T_tfidf_te <- tfidf_mat(test_txt,  vocab_u, vocab_b, idf_u, idf_b)
T_cnt_tr   <- cnt_mat(train_txt, vocab_u)
T_cnt_te   <- cnt_mat(test_txt,  vocab_u)
cat("tfidf features:", ncol(T_tfidf_tr), " count features:", ncol(T_cnt_tr), "\n")

# ---------- 3. TABULAR FEATURES (medians/modes from TRAIN) ----------
tab_feats <- function(d) d %>% transmute(
  Age, DependentChildren, DependentsOther, WeeklyWages, HoursWorkedPerWeek,
  DaysWorkedPerWeek, InitialIncurredCalimsCost, Days_To_Report, AccidentYear,
  WeeklyWagesPerHour = if_else(HoursWorkedPerWeek > 0, WeeklyWages / HoursWorkedPerWeek, 0),
  DependentsTotal = DependentChildren + DependentsOther
)
tr_num <- tab_feats(train); te_num <- tab_feats(test)
meds <- map_dbl(tr_num, ~ median(.x, na.rm = TRUE))
tr_num <- map2_dfc(tr_num, meds, ~ if_else(is.na(.x), .y, .x))
te_num <- map2_dfc(te_num,  meds, ~ if_else(is.na(.x), .y, .x))

onehot <- function(d, lv) {
  dd <- as.data.frame(lapply(d, function(x) factor(x, levels = lv[[length(lv)]])))
  # d columns are in order of `cats`; lv is a parallel list
  for (j in seq_along(d)) dd[[j]] <- factor(d[[j]], levels = lv[[j]])
  model.matrix(~ . - 1, data = dd)
}
cats <- c("Gender", "MaritalStatus", "PartTimeFullTime")
cat_lv <- map(train[cats], ~ levels(factor(na.omit(.x))))
tr_cat <- onehot(map2(train[cats], cat_lv, ~ if_else(is.na(.x), .y[1], .x)), cat_lv)
te_cat <- onehot(map2(test[cats],  cat_lv, ~ if_else(is.na(.x), .y[1], .x)), cat_lv)
colnames(tr_cat) <- paste0("c_", colnames(tr_cat)); colnames(te_cat) <- colnames(tr_cat)

X_base_tr <- cbind(as.matrix(tr_num), tr_cat)
X_base_te <- cbind(as.matrix(te_num), te_cat)

# ---------- 4. MODEL ARMS ----------
y_tr <- train$UltimateIncurredClaimCost; y_te <- test$UltimateIncurredClaimCost
q_te <- ntile(y_te, 5)

metrics_dollars <- function(pred, actual) {
  e <- pred - actual
  tibble(
    RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)),
    MAPE = mean(abs(e) / pmax(actual, 1)) * 100,
    R2 = 1 - sum(e^2) / sum((actual - mean(actual))^2),
    RMSLE = sqrt(mean((log1p(pmax(pred, 0)) - log1p(actual))^2))
  )
}
seg_mae <- function(pred, actual, q) tapply(abs(pred - actual), q, mean)

run_arm <- function(X_tr, X_te, y_lab_tr, back_trafo, objective, tag, tweedie_p = NULL) {
  # internal validation for early stopping + config choice
  set.seed(42); vi <- sample(nrow(X_tr), 0.1 * nrow(X_tr))
  dsub <- xgb.DMatrix(X_tr[-vi, ], label = y_lab_tr[-vi])
  dval <- xgb.DMatrix(X_tr[vi, ],  label = y_lab_tr[vi])
  best <- NULL
  for (eta in c(0.03, 0.1)) for (depth in c(4, 6, 8)) {
    p <- list(objective = objective, eta = eta, max_depth = depth,
              lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
    if (!is.null(tweedie_p)) p$tweedie_variance_power <- tweedie_p
    set.seed(42)
    m <- xgb.train(p, dsub, nrounds = 3000, evals = list(val = dval),
                   early_stopping_rounds = 60, verbose = 0)
    it <- m$best_iteration; sc <- m$best_score
    if (is.null(it) || length(it) == 0 || is.na(it) || it < 10) {
      # no early stopping triggered (or degenerate): use the full 3000 rounds and
      # take the val score at the last round from the evaluation log
      ev <- m$evaluation_log
      it <- nrounds_full <- if (!is.null(ev) && nrow(ev) > 0) nrow(ev) else 3000
      sc <- if (!is.null(ev) && ncol(ev) >= 2) ev[[ncol(ev)]][nrow(ev)] else Inf
    }
    if (is.null(best) || sc < best$val) best <- list(val = sc, iter = it, eta = eta, depth = depth)
  }
  p <- list(objective = objective, eta = best$eta, max_depth = best$depth,
            lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
  if (!is.null(tweedie_p)) p$tweedie_variance_power <- tweedie_p
  set.seed(42)
  mfull <- xgb.train(p, xgb.DMatrix(X_tr, label = y_lab_tr), nrounds = best$iter, verbose = 0)
  pred <- back_trafo(predict(mfull, X_te))
  list(metrics = metrics_dollars(pred, y_te) %>% mutate(arm = tag, eta = best$eta,
       depth = best$depth, rounds = best$iter), seg = seg_mae(pred, y_te, q_te),
       model = mfull, pred = pred)
}

cat("\n== ARM A: baseline-clean (counts, dollars) ==\n")
XA_tr <- cbind(X_base_tr, as.matrix(T_cnt_tr)); XA_te <- cbind(X_base_te, as.matrix(T_cnt_te))
armA <- run_arm(XA_tr, XA_te, y_tr, function(z) z, "reg:squarederror", "A_baseline_clean")

cat("== ARM B: log1p + tfidf ==\n")
XB_tr <- cbind(X_base_tr, as.matrix(T_tfidf_tr)); XB_te <- cbind(X_base_te, as.matrix(T_tfidf_te))
armB <- run_arm(XB_tr, XB_te, log1p(y_tr), function(z) expm1(z), "reg:squarederror", "B_log_tfidf")

cat("== ARM C: tweedie + tfidf ==\n")
armC <- run_arm(XB_tr, XB_te, y_tr, function(z) z, "reg:tweedie", "C_tweedie_tfidf", tweedie_p = 1.5)

# ---------- 5. REPORT ----------
res <- bind_rows(armA$metrics, armB$metrics, armC$metrics)
print(res %>% select(arm, eta, depth, rounds, RMSE, MAE, MAPE, R2, RMSLE))
cat("\nMAE by test-target quintile (Q1=cheapest .. Q5=most expensive):\n")
print(cbind(A = armA$seg, B = armB$seg, C = armC$seg))

best <- res %>% arrange(MAE) %>% slice(1)
cat(sprintf('\nBEST_BY_MAE: %s  RMSE=%.2f MAE=%.2f MAPE=%.2f R2=%.3f RMSLE=%.4f\n',
            best$arm, best$RMSE, best$MAE, best$MAPE, best$R2, best$RMSLE))

for (a in list(armA, armB, armC)) {
  imp <- xgb.importance(model = a$model) %>% slice_head(n = 8)
  cat(sprintf('\nFINAL_METRICS arm=%s rmse=%.4f mae=%.4f mape=%.4f r2=%.6f rmsle=%.6f top=%s\n',
              a$metrics$arm, a$metrics$RMSE, a$metrics$MAE, a$metrics$MAPE,
              a$metrics$R2, a$metrics$RMSLE, paste(imp$Feature, collapse = "|")))
}
