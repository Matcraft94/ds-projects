# Round 1 (2026-10-07): clean-protocol 3-arm comparison.
# A: original design + audit fixes (seeded split first, no target imputation)
# B: log1p target + TF-IDF          C: Tweedie + TF-IDF
# Refactored onto R/ (single source of truth). Callable from _targets.R or standalone.

run_round1 <- function(train_df, test_df, tfidf, counts, dummies, tab_med,
                       y_train, y_test, q_test, seed = 42) {
  suppressMessages({library(tidyverse); library(xgboost)})
  y_tr <- y_train; y_te <- y_test
  X_base_tr <- cbind(as.matrix(tab_med$train), dummies$train)
  X_base_te <- cbind(as.matrix(tab_med$test), dummies$test)

  run_arm <- function(X_tr, X_te, y_lab_tr, back_trafo, objective, tag, tweedie_p = NULL) {
    set.seed(seed); vi <- sample(nrow(X_tr), 0.1 * nrow(X_tr))
    dsub <- xgb.DMatrix(X_tr[-vi, ], label = y_lab_tr[-vi])
    dval <- xgb.DMatrix(X_tr[vi, ],  label = y_lab_tr[vi])
    best <- NULL
    for (eta in c(0.03, 0.1)) for (depth in c(4, 6, 8)) {
      p <- list(objective = objective, eta = eta, max_depth = depth,
                lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
      if (!is.null(tweedie_p)) p$tweedie_variance_power <- tweedie_p
      set.seed(seed)
      m <- xgb.train(p, dsub, nrounds = 3000, evals = list(val = dval), verbose = 0)
      pv <- predict(m, dval)
      sc <- sqrt(mean((pv - y_lab_tr[vi])^2))
      if (is.null(best) || sc < best$val) best <- list(val = sc, iter = 3000, eta = eta, depth = depth)
    }
    p <- list(objective = objective, eta = best$eta, max_depth = best$depth,
              lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
    if (!is.null(tweedie_p)) p$tweedie_variance_power <- tweedie_p
    set.seed(seed)
    mfull <- xgb.train(p, xgb.DMatrix(X_tr, label = y_lab_tr), nrounds = best$iter, verbose = 0)
    pred <- back_trafo(predict(mfull, X_te))
    list(pred = pred,
         metrics = metrics_dollars(pred, y_te) %>%
           mutate(arm = tag, eta = best$eta, depth = best$depth, rounds = best$iter))
  }

  XA_tr <- cbind(X_base_tr, as.matrix(counts$train)); XA_te <- cbind(X_base_te, as.matrix(counts$test))
  XB_tr <- cbind(X_base_tr, as.matrix(tfidf$train));  XB_te <- cbind(X_base_te, as.matrix(tfidf$test))

  armA <- run_arm(XA_tr, XA_te, y_tr, function(z) z, "reg:squarederror", "A_baseline_clean")
  armB <- run_arm(XB_tr, XB_te, log1p(y_tr), function(z) expm1(z), "reg:squarederror", "B_log_tfidf")
  armC <- run_arm(XB_tr, XB_te, y_tr, function(z) z, "reg:tweedie", "C_tweedie_tfidf", tweedie_p = 1.5)

  metrics <- bind_rows(armA$metrics, armB$metrics, armC$metrics)
  segments <- list(A_baseline_clean = seg_mae(armA$pred, y_te, q_test),
                   B_log_tfidf = seg_mae(armB$pred, y_te, q_test),
                   C_tweedie_tfidf = seg_mae(armC$pred, y_te, q_test))
  save_round("round1_improved", metrics, segments)
  metrics
}

# ---- standalone execution ----
if (sys.nframe() == 0L) {
  for (f in list.files("R", full.names = TRUE)) source(f)
  raw <- load_raw(); prep <- add_derived(raw); split <- make_split(prep)
  train_txt <- split$train %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
  test_txt  <- split$test  %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
  tr_tokens <- tokenize_claims(train_txt); te_tokens <- tokenize_claims(test_txt)
  tr_bigrams <- bigrams_of(train_txt); te_bigrams <- bigrams_of(test_txt)
  vocab_u <- build_vocab(tr_tokens, "stem"); vocab_b <- build_vocab(tr_bigrams, "bigram")
  tfidf <- make_tfidf(tr_tokens, te_tokens, tr_bigrams, te_bigrams, vocab_u, vocab_b, train_txt, test_txt)
  counts <- make_counts(tr_tokens, te_tokens, vocab_u, train_txt, test_txt)
  dummies <- cat_dummies(split$train, split$test)
  tab_base_tr <- tab_features(split$train, train_txt, extras = FALSE)
  tab_base_te <- tab_features(split$test, test_txt, extras = FALSE)
  tab_med <- impute_median(tab_base_tr, tab_base_te)
  m <- run_round1(split$train, split$test, tfidf, counts, dummies, tab_med,
                  split$train$UltimateIncurredClaimCost,
                  split$test$UltimateIncurredClaimCost,
                  ntile(split$test$UltimateIncurredClaimCost, 5))
  print(m, n = 3)
}
