# GLM / GAM / GAMM / hurdle vs XGBoost — same clean protocol (2026-10-07)
# Same split and features as improved_model.R (seed 42, code path copied verbatim).
# Arms:
#   REF_XGB   : reference = arm B of improved_model.R (log1p XGB + TF-IDF, eta .03 depth 4, 3000 r)
#   GLM_ENET  : elastic-net GLM (gaussian on log1p) — all features, cv-lambda
#   GLM_GAMMA : classic insurance GLM — Gamma(log) on tabular + top-40 text (glm)
#   GAM       : mgcv::bam Gamma(log), splines on continuous + top-40 text linear
#   GAMM      : GAM + s(AccidentYear, bs="re") random effect
#   HURDLE    : two-part XGB: P(cost>50k) logistic + Gamma severity on each part
# All metrics in DOLLARS on the same untouched test set; quintile MAE included.

suppressMessages({
  library(tidyverse); library(tidymodels); library(tidytext)
  library(SnowballC); library(stopwords); library(xgboost)
  library(glmnet); library(mgcv)
})
set.seed(42)

t0 <- proc.time()
cat("== LOAD ==\n")
raw <- read_csv("Data/actuarial_loss/train.csv", show_col_types = FALSE) %>% select(-ClaimNumber)
cat("rows:", nrow(raw), "\n")

prep <- raw %>%
  mutate(
    Days_To_Report = as.numeric(difftime(DateReported, DateTimeOfAccident, units = "days")),
    AccidentYear = lubridate::year(DateTimeOfAccident)
  )
n_na_target <- sum(is.na(prep$UltimateIncurredClaimCost))
cat("rows with NA target:", n_na_target, "\n")

# ---------- SPLIT (verbatim from improved_model.R -> identical test set) ----------
split_var <- prep %>% mutate(cost_bin = ntile(UltimateIncurredClaimCost, 10)) %>% pull(cost_bin)
set.seed(42)
idx <- sample(nrow(prep), size = 0.8 * nrow(prep))
tr_rows <- idx; te_rows <- setdiff(seq_len(nrow(prep)), idx)
train <- prep[tr_rows, ]; test <- prep[te_rows, ]
cat("train:", nrow(train), " test:", nrow(test), "\n")
train <- train %>% filter(!is.na(UltimateIncurredClaimCost))
test  <- test  %>% filter(!is.na(UltimateIncurredClaimCost))

# ---------- TEXT FEATURES (same as improved_model.R) ----------
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
doc_n <- nrow(train_txt)
idf_u <- tr_u %>% filter(stem %in% vocab_u) %>% distinct(claim_id, stem) %>%
  count(stem) %>% rename(df_doc = n) %>% mutate(idf = log(doc_n / df_doc))
idf_b <- tr_b %>% filter(bigram %in% vocab_b) %>% distinct(claim_id, bigram) %>%
  count(bigram) %>% rename(df_doc = n) %>% mutate(idf = log(doc_n / df_doc))

pad_rows <- function(wide, txt_df) {
  full <- tibble(claim_id = seq_len(nrow(txt_df)))
  out <- full %>% left_join(wide, by = "claim_id") %>% select(-claim_id)
  out[is.na(out)] <- 0; as.data.frame(out)
}
tfidf_mat <- function(txt_df) {
  ttl <- txt_df %>% group_by(claim_id) %>% summarise(ntot = sum(strsplit(txt, " ") |> map_int(length)), .groups = "drop")
  u <- tok(txt_df) %>% filter(stem %in% vocab_u) %>% count(claim_id, stem, name = "n") %>%
    left_join(ttl, by = "claim_id") %>% left_join(idf_u, by = "stem") %>%
    mutate(v = (n / ntot) * idf) %>%
    pivot_wider(id_cols = claim_id, names_from = stem, values_from = v, values_fill = 0, names_prefix = "CD_")
  b <- txt_df %>% unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>%
    filter(!is.na(bigram), bigram %in% vocab_b) %>% count(claim_id, bigram, name = "n") %>%
    left_join(ttl, by = "claim_id") %>% left_join(idf_b, by = "bigram") %>%
    mutate(v = (n / ntot) * idf) %>%
    pivot_wider(id_cols = claim_id, names_from = bigram, values_from = v, values_fill = 0, names_prefix = "CB_")
  pad_rows(u %>% full_join(b, by = "claim_id"), txt_df)
}
T_tfidf_tr <- tfidf_mat(train_txt); T_tfidf_te <- tfidf_mat(test_txt)
cat("tfidf features:", ncol(T_tfidf_tr), "\n")

# top-40 stems (by train document frequency) for the sparse-model arms
top40 <- tr_u %>% filter(stem %in% vocab_u) %>% distinct(claim_id, stem) %>%
  count(stem, sort = TRUE) %>% slice_head(n = 40) %>% pull(stem)
TXT40_tr <- T_tfidf_tr[paste0("CD_", top40)]; TXT40_te <- T_tfidf_te[paste0("CD_", top40)]

# ---------- TABULAR ----------
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

DF_tr <- train %>% select(Gender, MaritalStatus, PartTimeFullTime) %>%
  mutate(across(everything(), ~ replace_na(.x, "NA")))
DF_te <- test %>% select(Gender, MaritalStatus, PartTimeFullTime) %>%
  mutate(across(everything(), ~ replace_na(.x, "NA")))
for (v in c("Gender", "MaritalStatus", "PartTimeFullTime")) {
  lv <- levels(factor(DF_tr[[v]])); DF_tr[[v]] <- factor(DF_tr[[v]], lv); DF_te[[v]] <- factor(DF_te[[v]], lv)
}

y_tr <- train$UltimateIncurredClaimCost; y_te <- test$UltimateIncurredClaimCost
q_te <- ntile(y_te, 5)

metrics_dollars <- function(pred, actual) {
  e <- pred - actual
  tibble(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)),
         MAPE = mean(abs(e) / pmax(actual, 1)) * 100,
         R2 = 1 - sum(e^2) / sum((actual - mean(actual))^2),
         RMSLE = sqrt(mean((log1p(pmax(pred, 0)) - log1p(actual))^2)))
}
seg_mae <- function(pred) tapply(abs(pred - y_te), q_te, mean)

# design matrices for linear models (tabular dummies + top40 text)
X_lm_tr <- model.matrix(~ . - 1, data = bind_cols(DF_tr, TXT40_tr, tr_num %>%
  select(-AccidentYear)))  # year handled separately per arm
X_lm_te <- model.matrix(~ . - 1, data = bind_cols(DF_te, TXT40_te, te_num %>%
  select(-AccidentYear)))
storage.mode(X_lm_tr) <- "double"; storage.mode(X_lm_te) <- "double"
# align columns
common <- intersect(colnames(X_lm_tr), colnames(X_lm_te))
X_lm_tr <- X_lm_tr[, common]; X_lm_te <- X_lm_te[, common]

res <- list()

## ---- REF: XGB log1p + tfidf (arm B config) ----
cat("\n== REF_XGB ==\n")
X_full_tr <- cbind(as.matrix(tr_num), X_lm_tr[, grep("^CD_|^Gender|^Marital|^Part", colnames(X_lm_tr))])
# rebuild full tfidf X exactly like improved_model.R: base + all 300 text cols
onehot_m <- function(d) model.matrix(~ . - 1, data = d)
mm_tr <- onehot_m(DF_tr); mm_te <- onehot_m(DF_te)
colnames(mm_tr) <- paste0("c_", colnames(mm_tr)); colnames(mm_te) <- paste0("c_", colnames(mm_te))
X_full_tr <- cbind(as.matrix(tr_num), mm_tr, as.matrix(T_tfidf_tr))
X_full_te <- cbind(as.matrix(te_num), mm_te, as.matrix(T_tfidf_te))
p <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 4,
          lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
set.seed(42)
m_ref <- xgb.train(p, xgb.DMatrix(X_full_tr, label = log1p(y_tr)), nrounds = 3000, verbose = 0)
pred_ref <- pmax(expm1(predict(m_ref, X_full_te)), 0)
res$REF_XGB <- metrics_dollars(pred_ref, y_te) %>% mutate(arm = "REF_XGB"); attr(res$REF_XGB, "seg") <- seg_mae(pred_ref)

# winsorize numerics at train 1%/99% (trees are immune; linear/GAM arms need it)
winsor <- function(tr, te) {
  lo <- map_dbl(tr, ~ quantile(.x, 0.01, na.rm = TRUE))
  hi <- map_dbl(tr, ~ quantile(.x, 0.99, na.rm = TRUE))
  list(
    tr = map2_dfc(tr, seq_along(tr), ~ pmin(pmax(.x, lo[.y]), hi[.y])),
    te = map2_dfc(te, seq_along(te), ~ pmin(pmax(.x, lo[.y]), hi[.y]))
  )
}
w_num <- winsor(tr_num, te_num)
tr_num_w <- w_num$tr; te_num_w <- w_num$te

## ---- GLM elastic net (gaussian on log1p, all features) ----
cat("== GLM_ENET ==\n")
Xn_tr <- cbind(as.matrix(tr_num_w), mm_tr, as.matrix(T_tfidf_tr))
Xn_te <- cbind(as.matrix(te_num_w), mm_te, as.matrix(T_tfidf_te))
set.seed(42)
cvfit <- cv.glmnet(Xn_tr, log1p(y_tr), alpha = 0.5, nfolds = 5)
pred <- pmax(expm1(predict(cvfit, Xn_te, s = "lambda.min")), 0)
res$GLM_ENET <- metrics_dollars(pred, y_te) %>% mutate(arm = "GLM_ENET"); attr(res$GLM_ENET, "seg") <- seg_mae(pred)

## ---- GLM Gamma(log) classic ----
cat("== GLM_GAMMA ==\n")
df_glm_tr <- data.frame(y = pmax(y_tr, 1), DF_tr, TXT40_tr, tr_num_w %>% select(-AccidentYear))
df_glm_te <- data.frame(DF_te, TXT40_te, te_num_w %>% select(-AccidentYear))
fit_glm <- glm(y ~ ., family = Gamma(link = "log"), data = df_glm_tr,
               control = glm.control(maxit = 200, epsilon = 1e-8))
pred <- pmax(predict(fit_glm, df_glm_te, type = "response"), 0)
res$GLM_GAMMA <- metrics_dollars(pred, y_te) %>% mutate(arm = "GLM_GAMMA"); attr(res$GLM_GAMMA, "seg") <- seg_mae(pred)

## ---- GAM (bam, Gamma log, splines + linear text) ----
cat("== GAM ==\n")
gam_form <- y ~ s(Age, k = 10) + s(WeeklyWages, k = 10) + s(InitialIncurredCalimsCost, k = 10) +
  s(Days_To_Report, k = 10) + s(WeeklyWagesPerHour, k = 10) + s(HoursWorkedPerWeek, k = 10) +
  Gender + MaritalStatus + PartTimeFullTime + DependentsTotal + DaysWorkedPerWeek +
  DependentChildren + DependentsOther
df_gam_tr <- data.frame(y = pmax(y_tr, 1), DF_tr, tr_num_w, TXT40_tr)
df_gam_te <- data.frame(DF_te, te_num_w, TXT40_te)
fit_gam <- bam(gam_form, family = Gamma(link = "log"), data = df_gam_tr, discrete = TRUE)
pred <- pmax(predict(fit_gam, df_gam_te, type = "response"), 0)
res$GAM <- metrics_dollars(pred, y_te) %>% mutate(arm = "GAM"); attr(res$GAM, "seg") <- seg_mae(pred)

## ---- GAMM (GAM + year random effect) ----
cat("== GAMM ==\n")
gamm_form <- update(gam_form, . ~ . + s(AccidentYear, bs = "re"))
df_gamm_tr <- data.frame(y = pmax(y_tr, 1), DF_tr, tr_num_w, TXT40_tr)
fit_gamm <- bam(gamm_form, family = Gamma(link = "log"), data = df_gamm_tr, discrete = TRUE)
pred <- pmax(predict(fit_gamm, df_gam_te, type = "response"), 0)
res$GAMM <- metrics_dollars(pred, y_te) %>% mutate(arm = "GAMM"); attr(res$GAMM, "seg") <- seg_mae(pred)

## ---- HURDLE: P(cost > 50k) * E[cost|exp] + (1-p) * E[cost|cheap] ----
cat("== HURDLE ==\n")
thr <- 50000
z_tr <- as.numeric(y_tr > thr)
set.seed(42)
m_cl <- xgb.train(list(objective = "binary:logistic", eta = 0.05, max_depth = 6,
                       subsample = 0.8, colsample_bytree = 0.8),
                  xgb.DMatrix(X_full_tr, label = z_tr), nrounds = 500, verbose = 0)
p_hat <- predict(m_cl, X_full_te)

exp_idx <- which(y_tr > thr); cheap_idx <- which(y_tr <= thr)
set.seed(42)
m_exp <- xgb.train(list(objective = "reg:gamma", eta = 0.05, max_depth = 6,
                        subsample = 0.8, colsample_bytree = 0.8),
                   xgb.DMatrix(X_full_tr[exp_idx, ], label = y_tr[exp_idx]), nrounds = 800, verbose = 0)
set.seed(42)
m_cheap <- xgb.train(list(objective = "reg:squarederror", eta = 0.05, max_depth = 6,
                          subsample = 0.8, colsample_bytree = 0.8),
                     xgb.DMatrix(X_full_tr[cheap_idx, ], label = log1p(y_tr[cheap_idx])), nrounds = 800, verbose = 0)
pred <- p_hat * pmax(predict(m_exp, X_full_te), 0) +
  (1 - p_hat) * pmax(expm1(predict(m_cheap, X_full_te)), 0)
res$HURDLE <- metrics_dollars(pred, y_te) %>% mutate(arm = "HURDLE"); attr(res$HURDLE, "seg") <- seg_mae(pred)

# ---------- REPORT ----------
tab <- bind_rows(res) %>% select(arm, RMSE, MAE, MAPE, R2, RMSLE)
print(tab, n = 20)
cat("\nMAE by test-target quintile:\n")
segs <- map_dfc(res, ~ as.numeric(attr(.x, "seg")))
rownames(segs) <- paste0("Q", 1:5); print(segs)

for (nm in names(res)) {
  r <- res[[nm]]; s <- as.numeric(attr(r, "seg"))
  cat(sprintf('FINAL_METRICS arm=%s rmse=%.4f mae=%.4f mape=%.4f r2=%.6f rmsle=%.6f q5mae=%.2f\n',
              nm, r$RMSE, r$MAE, r$MAPE, r$R2, r$RMSLE, s[5]))
}
cat(sprintf("elapsed_min=%.1f\n", (proc.time() - t0)[3] / 60))
