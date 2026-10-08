# Round 4 (2026-10-07): final squeeze — OOF-TE, median objective, monotone
# constraints, tuned grid, ensemble; interaction ceiling v2.
# Refactored onto R/ (target_encode_text/initial support method="oof").

run_round4 <- function(train_df, test_df, train_txt, test_txt,
                       tr_tokens, te_tokens, tr_bigrams, te_bigrams,
                       vocab_u, vocab_b, dummies, y_train, y_test, q_test,
                       seed = 42) {
  suppressMessages({library(tidyverse); library(xgboost)})
  y_tr <- y_train; y_te <- y_test
  yl_tr <- log1p(y_tr)
  n_tr <- nrow(train_df)
  v_total <- var(yl_tr)

  teu_is <- target_encode_text(tr_tokens, te_tokens, vocab_u, yl_tr, "stem", method = "insample")
  teb_is <- target_encode_text(tr_bigrams, te_bigrams, vocab_b, yl_tr, "bigram", method = "insample")
  teu_oof <- target_encode_text(tr_tokens, te_tokens, vocab_u, yl_tr, "stem", method = "oof")
  teb_oof <- target_encode_text(tr_bigrams, te_bigrams, vocab_b, yl_tr, "bigram", method = "oof")
  ip_is <- target_encode_initial(train_df$InitialIncurredCalimsCost, yl_tr,
                                 test_df$InitialIncurredCalimsCost, method = "insample")
  ip_oof <- target_encode_initial(train_df$InitialIncurredCalimsCost, yl_tr,
                                  test_df$InitialIncurredCalimsCost, method = "oof")

  tab_tr <- tab_features(train_df, train_txt, extras = TRUE)
  tab_te <- tab_features(test_df, test_txt, extras = TRUE)
  tab_is <- tab_tr %>% mutate(init_pct_te = ip_is$train, te_stem = teu_is$train, te_bigram = teb_is$train)
  tab_oof <- tab_tr %>% mutate(init_pct_te = ip_oof$train, te_stem = teu_oof$train, te_bigram = teb_oof$train)
  tab_te_f <- tab_te %>% mutate(init_pct_te = ip_is$test, te_stem = teu_is$test, te_bigram = teb_is$test)
  tabm_is <- impute_median(tab_is, tab_te_f); tabm_oof <- impute_median(tab_oof, tab_te_f)
  X_is  <- cbind(as.matrix(tabm_is$train), dummies$train)
  X_oof <- cbind(as.matrix(tabm_oof$train), dummies$train)
  X_te  <- cbind(as.matrix(tabm_is$test), dummies$test)
  stopifnot(all(colnames(X_is) == colnames(X_te)),
            all(colnames(X_oof) == colnames(X_te)))

  # ceiling v2: Initial(20 pct buckets) x stemTE(20 buckets)
  g1 <- ceiling(ecdf(train_df$InitialIncurredCalimsCost)(train_df$InitialIncurredCalimsCost) * 20)
  g2 <- ceiling(rank(tab_is$te_stem) / n_tr * 20)
  ceil1 <- 1 - sum(tapply(yl_tr, g1, var) * as.numeric(table(g1)), na.rm = TRUE) / (n_tr * v_total)
  ceil2 <- 1 - sum(tapply(yl_tr, paste(g1, g2), var) * as.numeric(table(paste(g1, g2))), na.rm = TRUE) / (n_tr * v_total)

  base_p <- function(extra = NULL) {
    p <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 4,
              lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
    if (!is.null(extra)) for (k in names(extra)) p[[k]] <- extra[[k]]
    p
  }
  fit_pred <- function(Xtr, params, rounds = 3000) {
    set.seed(seed)
    m <- xgb.train(params, xgb.DMatrix(Xtr, label = yl_tr), nrounds = rounds, verbose = 0)
    pmax(expm1(predict(m, X_te)), 0)
  }

  out <- list(); preds <- list()
  add <- function(tag, pred) {
    preds[[tag]] <<- pred
    out[[tag]] <<- metrics_dollars(pred, y_te) %>%
      mutate(arm = tag, R2log = r2_log(pred, y_te))
  }

  add("A1_FE2_ref", fit_pred(X_is, base_p()))
  add("A2_OOF_TE",  fit_pred(X_oof, base_p()))
  qobj <- tryCatch(
    fit_pred(X_oof, base_p(list(objective = "reg:quantileerror", quantile_alpha = 0.5))),
    error = function(e) { cat("quantile obj failed:", conditionMessage(e), "\n"); NULL })
  if (!is.null(qobj)) add("A3_median", qobj)
  mc <- setNames(rep(0, ncol(X_oof)), colnames(X_oof))
  mc["log_initial"] <- 1; mc["init_pct_te"] <- 1
  add("A4_monotone", fit_pred(X_oof, base_p(list(monotone_constraints = unname(mc)))))
  # A5: tuned grid (manual internal validation on log-RMSE)
  set.seed(seed); vi <- sample(n_tr, 0.15 * n_tr)
  dsub <- xgb.DMatrix(X_oof[-vi, ], label = yl_tr[-vi])
  dval <- xgb.DMatrix(X_oof[vi, ], label = yl_tr[vi])
  best <- NULL
  for (eta in c(0.02, 0.03, 0.05)) for (dp in c(3, 4, 6)) {
    set.seed(seed)
    m <- xgb.train(base_p(list(eta = eta, max_depth = dp)), dsub, nrounds = 3000, verbose = 0)
    pv <- predict(m, dval)
    sc <- sqrt(mean((pv - yl_tr[vi])^2))
    if (is.null(best) || sc < best$sc) best <- list(sc = sc, it = 3000, eta = eta, dp = dp)
  }
  add("A5_tuned", fit_pred(X_oof, base_p(list(eta = best$eta, max_depth = best$dp)), rounds = best$it))
  ens <- intersect(c("A2_OOF_TE", "A3_median", "A4_monotone", "A5_tuned"), names(preds))
  add("A6_ensemble", Reduce(`+`, preds[ens]) / length(ens))

  metrics <- bind_rows(out)
  segments <- imap(preds, ~ seg_mae(.x, y_te, q_test))
  save_round("round4_squeeze", metrics, segments)
  list(metrics = metrics, segments = segments,
       ceiling = c(initial20 = ceil1, initial_x_stemTE = ceil2),
       tuned = best)
}

if (sys.nframe() == 0L) {
  for (f in list.files("R", full.names = TRUE)) source(f)
  raw <- load_raw(); prep <- add_derived(raw); split <- make_split(prep)
  train_txt <- split$train %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
  test_txt  <- split$test  %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))
  tr_tokens <- tokenize_claims(train_txt); te_tokens <- tokenize_claims(test_txt)
  tr_bigrams <- bigrams_of(train_txt); te_bigrams <- bigrams_of(test_txt)
  vocab_u <- build_vocab(tr_tokens, "stem"); vocab_b <- build_vocab(tr_bigrams, "bigram")
  dummies <- cat_dummies(split$train, split$test)
  r <- run_round4(split$train, split$test, train_txt, test_txt,
                  tr_tokens, te_tokens, tr_bigrams, te_bigrams, vocab_u, vocab_b, dummies,
                  split$train$UltimateIncurredClaimCost, split$test$UltimateIncurredClaimCost,
                  ntile(split$test$UltimateIncurredClaimCost, 5))
  print(r$ceiling); print(r$metrics, n = 6)
}
