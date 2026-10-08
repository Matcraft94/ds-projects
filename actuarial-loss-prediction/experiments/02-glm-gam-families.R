# Round 2 (2026-10-07): GLM / GAM / GAMM / hurdle family comparison.
# Refactored onto R/. Winsorized numerics for the linear arms (trees are immune).

run_round2 <- function(train_df, test_df, tfidf, dummies, tab_med,
                       y_train, y_test, q_test, seed = 42) {
  suppressMessages({
    library(tidyverse); library(xgboost); library(glmnet); library(mgcv)
  })
  y_tr <- y_train; y_te <- y_test
  w <- winsorize(tab_med$train, tab_med$test)
  cats <- c("Gender", "MaritalStatus", "PartTimeFullTime")
  DF_tr <- raw_factor_df(train_df, cats)
  DF_te <- raw_factor_df(test_df, cats)

  Xr_tr <- cbind(as.matrix(tab_med$train), dummies$train, as.matrix(tfidf$train))
  Xr_te <- cbind(as.matrix(tab_med$test), dummies$test, as.matrix(tfidf$test))
  Xn_tr <- cbind(as.matrix(w$train), dummies$train, as.matrix(tfidf$train))
  Xn_te <- cbind(as.matrix(w$test), dummies$test, as.matrix(tfidf$test))

  out <- list(); preds <- list()
  add <- function(tag, pred) {
    preds[[tag]] <<- pred
    out[[tag]] <<- metrics_dollars(pred, y_te) %>% mutate(arm = tag)
  }

  # REF: XGB log + tfidf (round-1 arm-B configuration)
  p <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 4,
            lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
  set.seed(seed)
  m_ref <- xgb.train(p, xgb.DMatrix(Xr_tr, label = log1p(y_tr)), nrounds = 3000, verbose = 0)
  add("REF_XGB", pmax(expm1(predict(m_ref, Xr_te)), 0))

  # top-40 text cols by train doc-frequency for the sparse arms
  top40 <- top40_text_cols(tfidf$train)
  TXT40_tr <- tfidf$train[, top40, drop = FALSE]
  TXT40_te <- tfidf$test[, top40, drop = FALSE]

  # GLM elastic net on log1p
  set.seed(seed)
  cvfit <- cv.glmnet(Xn_tr, log1p(y_tr), alpha = 0.5, nfolds = 5)
  add("GLM_ENET", pmax(expm1(predict(cvfit, Xn_te, s = "lambda.min")), 0))

  # GLM Gamma(log) — divergence recorded as NA row (documented failure)
  df_glm_tr <- data.frame(y = pmax(y_tr, 1), DF_tr, TXT40_tr,
                          w$train %>% select(-AccidentYear))
  df_glm_te <- data.frame(DF_te, TXT40_te, w$test %>% select(-AccidentYear))
  pred_glm <- tryCatch({
    fit_glm <- glm(y ~ ., family = Gamma(link = "log"), data = df_glm_tr,
                   control = glm.control(maxit = 200, epsilon = 1e-8))
    pmax(predict(fit_glm, df_glm_te, type = "response"), 0)
  }, error = function(e) NULL,
     warning = function(w) if (grepl("did not converge", conditionMessage(w))) NULL else {
       fit_glm <- suppressWarnings(glm(y ~ ., family = Gamma(link = "log"), data = df_glm_tr,
                                       control = glm.control(maxit = 200, epsilon = 1e-8)))
       pmax(predict(fit_glm, df_glm_te, type = "response"), 0)
     })
  if (is.null(pred_glm)) {
    out[["GLM_GAMMA"]] <- tibble(RMSE = NA_real_, MAE = NA_real_, MAPE = NA_real_,
                                 R2 = NA_real_, RMSLE = NA_real_, arm = "GLM_GAMMA_diverged")
  } else add("GLM_GAMMA", pred_glm)

  # GAM (bam, splines) and GAMM (+ year RE)
  gam_form <- y ~ s(Age, k = 10) + s(WeeklyWages, k = 10) + s(InitialIncurredCalimsCost, k = 10) +
    s(Days_To_Report, k = 10) + s(WeeklyWagesPerHour, k = 10) + s(HoursWorkedPerWeek, k = 10) +
    Gender + MaritalStatus + PartTimeFullTime + DependentsTotal + DaysWorkedPerWeek +
    DependentChildren + DependentsOther
  df_gam_tr <- data.frame(y = pmax(y_tr, 1), DF_tr, w$train, TXT40_tr)
  df_gam_te <- data.frame(DF_te, w$test, TXT40_te)
  fit_gam <- bam(gam_form, family = Gamma(link = "log"), data = df_gam_tr, discrete = TRUE)
  add("GAM", pmax(predict(fit_gam, df_gam_te, type = "response"), 0))
  fit_gamm <- bam(update(gam_form, . ~ . + s(AccidentYear, bs = "re")),
                  family = Gamma(link = "log"), data = df_gam_tr, discrete = TRUE)
  add("GAMM", pmax(predict(fit_gamm, df_gam_te, type = "response"), 0))

  # HURDLE: P(cost>50k) x Gamma severity + cheap-claims log regressor
  thr <- 50000
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
  add("HURDLE", ph * pmax(predict(m_exp, Xr_te), 0) +
        (1 - ph) * pmax(expm1(predict(m_chp, Xr_te)), 0))

  metrics <- bind_rows(out)
  segments <- imap(preds, ~ seg_mae(.x, y_te, q_test))
  save_round("round2_families", metrics, segments)
  metrics
}

# ---- helpers ----
raw_factor_df <- function(d, cats) {
  out <- d %>% select(all_of(cats)) %>% mutate(across(everything(), ~ replace_na(.x, "NA")))
  for (v in cats) out[[v]] <- factor(out[[v]])
  out
}
top40_text_cols <- function(tfidf_train) {
  nz <- colnames(tfidf_train)[colSums(tfidf_train != 0) > 0]
  ord <- order(colSums(tfidf_train[, nz, drop = FALSE] != 0), decreasing = TRUE)
  head(nz[ord], 40)
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
  tab_base_tr <- tab_features(split$train, train_txt, extras = FALSE)
  tab_base_te <- tab_features(split$test, test_txt, extras = FALSE)
  tab_med <- impute_median(tab_base_tr, tab_base_te)
  m <- run_round2(split$train, split$test, tfidf, dummies, tab_med,
                  split$train$UltimateIncurredClaimCost,
                  split$test$UltimateIncurredClaimCost,
                  ntile(split$test$UltimateIncurredClaimCost, 5))
  print(m, n = 10)
}
