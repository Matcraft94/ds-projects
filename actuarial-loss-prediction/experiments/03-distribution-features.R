# Round 3 (2026-10-07): distribution study + target-encoded features.
# Part A: where the error lives (Initial-decile study, variance ceiling).
# Part B: TE stems/bigrams (in-sample), Initial-percentile TE, log/flag/interaction
#         extras; REF/FE1/FE2/FE3 arms. Refactored onto R/.

run_round3 <- function(train_df, test_df, train_txt, test_txt,
                       tr_tokens, te_tokens, tr_bigrams, te_bigrams,
                       vocab_u, vocab_b, tfidf, dummies, y_train, y_test,
                       q_test, init_dec_test, seed = 42) {
  suppressMessages({library(tidyverse); library(xgboost)})
  y_tr <- y_train; y_te <- y_test

  ## ---- Part A: distribution study (train) ----
  dec10 <- train_df %>% mutate(d = ntile(InitialIncurredCalimsCost, 10))
  study <- dec10 %>% group_by(d) %>%
    summarise(n = n(), init_med = median(InitialIncurredCalimsCost),
              y_med = median(UltimateIncurredClaimCost), y_mean = mean(UltimateIncurredClaimCost),
              ratio_med = median(UltimateIncurredClaimCost / pmax(InitialIncurredCalimsCost, 1)),
              p_blowup = mean(UltimateIncurredClaimCost > 3 * pmax(InitialIncurredCalimsCost, 1)),
              sd_log = sd(log1p(UltimateIncurredClaimCost)), .groups = "drop")
  total_var <- var(log1p(y_tr))
  ceiling_fn <- function(nb) {
    d <- ntile(train_df$InitialIncurredCalimsCost, nb)
    w <- tapply(log1p(y_tr), d, var) * as.numeric(table(d))
    1 - sum(w, na.rm = TRUE) / (length(y_tr) * total_var)
  }
  ceiling_dec <- ceiling_fn(10); ceiling_pct <- ceiling_fn(100)

  ## ---- Part B: features ----
  teu <- target_encode_text(tr_tokens, te_tokens, vocab_u, log1p(y_tr),
                            col = "stem", method = "insample")
  teb <- target_encode_text(tr_bigrams, te_bigrams, vocab_b, log1p(y_tr),
                            col = "bigram", method = "insample")
  ipte <- target_encode_initial(train_df$InitialIncurredCalimsCost, log1p(y_tr),
                                test_df$InitialIncurredCalimsCost, method = "insample")
  tab_tr <- tab_features(train_df, train_txt, extras = TRUE) %>%
    mutate(init_pct_te = ipte$train, te_stem = teu$train, te_bigram = teb$train)
  tab_te <- tab_features(test_df, test_txt, extras = TRUE) %>%
    mutate(init_pct_te = ipte$test, te_stem = teu$test, te_bigram = teb$test)
  tabm <- impute_median(tab_tr, tab_te)
  X_new_tr <- cbind(as.matrix(tabm$train), dummies$train)
  X_new_te <- cbind(as.matrix(tabm$test), dummies$test)

  new_cols <- c("log_initial", "init_zero", "init_x_delay", "word_count",
                "init_pct_te", "te_stem", "te_bigram")
  Xo_tr <- X_new_tr[, setdiff(colnames(X_new_tr), new_cols)]
  Xo_te <- X_new_te[, setdiff(colnames(X_new_te), new_cols)]

  xgb_log <- function(Xtr, Xte) {
    p <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 4,
              lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
    set.seed(seed)
    m <- xgb.train(p, xgb.DMatrix(Xtr, label = log1p(y_tr)), nrounds = 3000, verbose = 0)
    pmax(expm1(predict(m, Xte)), 0)
  }

  out <- list(); preds <- list()
  add <- function(tag, pred) {
    preds[[tag]] <<- pred
    out[[tag]] <<- metrics_dollars(pred, y_te) %>% mutate(arm = tag)
  }
  add("REF_tfidf", xgb_log(cbind(Xo_tr, as.matrix(tfidf$train)),
                           cbind(Xo_te, as.matrix(tfidf$test))))
  add("FE1_tfidf_new", xgb_log(cbind(X_new_tr, as.matrix(tfidf$train)),
                               cbind(X_new_te, as.matrix(tfidf$test))))
  add("FE2_new_only", xgb_log(X_new_tr, X_new_te))

  # FE3: hurdle with the full feature set
  thr <- 50000
  Xr_tr <- cbind(X_new_tr, as.matrix(tfidf$train)); Xr_te <- cbind(X_new_te, as.matrix(tfidf$test))
  set.seed(seed)
  m_cl <- xgb.train(list(objective = "binary:logistic", eta = 0.05, max_depth = 6,
                         subsample = 0.8, colsample_bytree = 0.8),
                    xgb.DMatrix(Xr_tr, label = as.numeric(y_tr > thr)), nrounds = 500, verbose = 0)
  ph <- predict(m_cl, Xr_te)
  ei <- which(y_tr > thr); ci <- which(y_tr <= thr)
  set.seed(seed)
  m_exp <- xgb.train(list(objective = "reg:gamma", eta = 0.05, max_depth = 6,
                          subsample = 0.8, colsample_bytree = 0.8),
                     xgb.DMatrix(Xr_tr[ei, ], label = y_tr[ei]), nrounds = 800, verbose = 0)
  set.seed(seed)
  m_chp <- xgb.train(list(objective = "reg:squarederror", eta = 0.05, max_depth = 6,
                          subsample = 0.8, colsample_bytree = 0.8),
                     xgb.DMatrix(Xr_tr[ci, ], label = log1p(y_tr[ci])), nrounds = 800, verbose = 0)
  add("FE3_hurdle_new", ph * pmax(predict(m_exp, Xr_te), 0) +
        (1 - ph) * pmax(expm1(predict(m_chp, Xr_te)), 0))

  metrics <- bind_rows(out)
  segments <- imap(preds, ~ seg_mae(.x, y_te, q_test))
  save_round("round3_features", metrics, segments)
  list(metrics = metrics,
       study = study,
       ceiling = c(deciles = ceiling_dec, percentiles = ceiling_pct,
                   total_sd_log = sqrt(total_var)),
       segments_by_initial_decile = imap(preds, ~ seg_mae(.x, y_te, init_dec_test)))
}

if (sys.nframe() == 0L) {
  for (f in list.files("R", full.names = TRUE)) source(f)
  raw <- load_raw(); prep <- add_derived(raw); split <- make_split(prep)
  train_txt <- split$train %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
  test_txt  <- split$test  %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
  tr_tokens <- tokenize_claims(train_txt); te_tokens <- tokenize_claims(test_txt)
  tr_bigrams <- bigrams_of(train_txt); te_bigrams <- bigrams_of(test_txt)
  vocab_u <- build_vocab(tr_tokens, "stem"); vocab_b <- build_vocab(tr_bigrams, "bigram")
  tfidf <- make_tfidf(tr_tokens, te_tokens, tr_bigrams, te_bigrams, vocab_u, vocab_b, train_txt, test_txt)
  dummies <- cat_dummies(split$train, split$test)
  r <- run_round3(split$train, split$test, train_txt, test_txt,
                  tr_tokens, te_tokens, tr_bigrams, te_bigrams, vocab_u, vocab_b,
                  tfidf, dummies,
                  split$train$UltimateIncurredClaimCost, split$test$UltimateIncurredClaimCost,
                  ntile(split$test$UltimateIncurredClaimCost, 5),
                  ntile(split$test$InitialIncurredCalimsCost, 10))
  print(r$ceiling); print(r$metrics, n = 4)
}
