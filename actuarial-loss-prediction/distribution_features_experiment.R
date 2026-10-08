# Distribution study + targeted feature engineering (2026-10-07)
# Part A: where does the error live? development surprise, predictability ceiling
# Part B: features driven by A: target-encoded stems/bigrams (the original's
#         term_severity idea, never used as a feature), description length,
#         Initial transforms + decile-TE, zero-Initial flag.
# Arms: REF (XGB log + TF-IDF), FE1 (+new feats), FE2 (new feats, no TF-IDF),
#       FE3 (hurdle + TF-IDF + new feats). Same split/seed as all experiments.

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

# ---- same split as every experiment (verbatim) ----
set.seed(42)
idx <- sample(nrow(prep), size = 0.8 * nrow(prep))
train <- prep[idx, ]; test <- prep[setdiff(seq_len(nrow(prep)), idx), ]
train <- train %>% filter(!is.na(UltimateIncurredClaimCost))
test  <- test  %>% filter(!is.na(UltimateIncurredClaimCost))
y_tr <- train$UltimateIncurredClaimCost; y_te <- test$UltimateIncurredClaimCost
q_te <- ntile(y_te, 5)
cat("train:", nrow(train), " test:", nrow(test), "\n")

# ================= A. DISTRIBUTION STUDY =================
cat("\n== A. DISTRIBUTION STUDY (train) ==\n")
dec10 <- train %>% mutate(d = ntile(InitialIncurredCalimsCost, 10))
study <- dec10 %>% group_by(d) %>%
  summarise(n = n(), init_med = median(InitialIncurredCalimsCost),
            y_med = median(UltimateIncurredClaimCost), y_mean = mean(UltimateIncurredClaimCost),
            ratio_med = median(UltimateIncurredClaimCost / pmax(InitialIncurredCalimsCost, 1)),
            p_blowup = mean(UltimateIncurredClaimCost > 3 * pmax(InitialIncurredCalimsCost, 1)),
            sd_log = sd(log1p(UltimateIncurredClaimCost)), .groups = "drop")
print(as.data.frame(study))

total_var <- var(log1p(y_tr))
ceiling_fn <- function(nb) {
  d <- ntile(train$InitialIncurredCalimsCost, nb)
  w <- tapply(log1p(y_tr), d, var) * as.numeric(table(d))
  1 - sum(w) / (length(y_tr) * total_var)
}
cat(sprintf("R2 ceiling (share of log-variance explained) from Initial alone: deciles=%.3f  percentiles=%.3f\n",
            ceiling_fn(10), ceiling_fn(100)))
cat(sprintf("residual SD of log-cost within Initial percentiles: %.3f (total %.3f)\n",
            sqrt(total_var * (1 - ceiling_fn(100))), sqrt(total_var)))

# ================= features (train-only statistics) =================
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
ttl <- function(txt_df) txt_df %>% group_by(claim_id) %>%
  summarise(ntot = sum(strsplit(txt, " ") |> map_int(length)), .groups = "drop")
tfidf_mat <- function(txt_df) {
  L <- ttl(txt_df)
  u <- tok(txt_df) %>% filter(stem %in% vocab_u) %>% count(claim_id, stem, name = "n") %>%
    left_join(L, by = "claim_id") %>% left_join(idf_u, by = "stem") %>%
    mutate(v = (n / ntot) * idf) %>%
    pivot_wider(id_cols = claim_id, names_from = stem, values_from = v, values_fill = 0, names_prefix = "CD_")
  b <- txt_df %>% unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>%
    filter(!is.na(bigram), bigram %in% vocab_b) %>% count(claim_id, bigram, name = "n") %>%
    left_join(L, by = "claim_id") %>% left_join(idf_b, by = "bigram") %>%
    mutate(v = (n / ntot) * idf) %>%
    pivot_wider(id_cols = claim_id, names_from = bigram, values_from = v, values_fill = 0, names_prefix = "CB_")
  pad_rows(u %>% full_join(b, by = "claim_id"), txt_df)
}
T_tr <- tfidf_mat(train_txt); T_te <- tfidf_mat(test_txt)

# ---- target encoding (smoothed mean log-cost; the original's unused term_severity) ----
gm <- mean(log1p(y_tr)); K <- 50
te_u_map <- tr_u %>% filter(stem %in% vocab_u) %>%
  left_join(tibble(claim_id = seq_len(nrow(train)), yl = log1p(y_tr)), by = "claim_id") %>%
  group_by(stem) %>% summarise(n = n(), m = mean(yl), .groups = "drop") %>%
  mutate(te = (n * m + K * gm) / (n + K))
te_b_map <- tr_b %>% filter(bigram %in% vocab_b) %>%
  left_join(tibble(claim_id = seq_len(nrow(train)), yl = log1p(y_tr)), by = "claim_id") %>%
  group_by(bigram) %>% summarise(n = n(), m = mean(yl), .groups = "drop") %>%
  mutate(te = (n * m + K * gm) / (n + K))
te_apply <- function(tokens, map, col) {
  tokens %>% filter(.data[[col]] %in% map[[col]]) %>% left_join(map, by = col) %>%
    group_by(claim_id) %>% summarise(te_mean = mean(te, na.rm = TRUE), te_n = dplyr::n(), .groups = "drop")
}
teu_tr <- te_apply(tr_u, te_u_map, "stem") %>% pad_rows(., train_txt) %>% mutate(te_n = 0) # pad keeps row count
teu_tr <- train_txt %>% select(claim_id) %>% left_join(te_apply(tr_u, te_u_map, "stem"), by = "claim_id") %>%
  mutate(te_mean = replace_na(te_mean, gm), te_n = replace_na(te_n, 0)) %>% select(-claim_id)
teu_te <- test_txt %>% select(claim_id) %>% left_join(te_apply(te_u, te_u_map, "stem"), by = "claim_id") %>%
  mutate(te_mean = replace_na(te_mean, gm), te_n = replace_na(te_n, 0)) %>% select(-claim_id)
teb_tr <- train_txt %>% select(claim_id) %>% left_join(te_apply(tr_b, te_b_map, "bigram"), by = "claim_id") %>%
  mutate(te_mean = replace_na(te_mean, gm), te_n = replace_na(te_n, 0)) %>% select(-claim_id)
teb_te <- test_txt %>% select(claim_id) %>% left_join(te_apply(te_b, te_b_map, "bigram"), by = "claim_id") %>%
  mutate(te_mean = replace_na(te_mean, gm), te_n = replace_na(te_n, 0)) %>% select(-claim_id)

# ---- tabular + distribution-driven extras ----
tab_feats <- function(d, txt_df) d %>% transmute(
  Age, DependentChildren, DependentsOther, WeeklyWages, HoursWorkedPerWeek,
  DaysWorkedPerWeek, InitialIncurredCalimsCost, Days_To_Report, AccidentYear,
  WeeklyWagesPerHour = if_else(HoursWorkedPerWeek > 0, WeeklyWages / HoursWorkedPerWeek, 0),
  DependentsTotal = DependentChildren + DependentsOther,
  log_initial = log1p(InitialIncurredCalimsCost),
  init_zero = as.numeric(InitialIncurredCalimsCost == 0),
  init_x_delay = log1p(InitialIncurredCalimsCost) * log1p(pmax(Days_To_Report, 0)),
  word_count = strsplit(txt_df$txt, " ") |> map_int(length)
)
tr_num <- tab_feats(train, train_txt); te_num <- tab_feats(test, test_txt)
# Initial-percentile target encoding (captures the full nonlinear Initial->cost curve)
init_ecdf <- ecdf(train$InitialIncurredCalimsCost)
init_pctl_te <- train %>% transmute(p = init_ecdf(InitialIncurredCalimsCost), yl = log1p(UltimateIncurredClaimCost)) %>%
  mutate(b = ceiling(p * 100)) %>% group_by(b) %>% summarise(te = mean(yl), .groups = "drop")
tr_num$init_pct_te <- init_pctl_te$te[ceiling(init_ecdf(train$InitialIncurredCalimsCost) * 100)]
te_p <- init_ecdf(test$InitialIncurredCalimsCost)
te_num$init_pct_te <- init_pctl_te$te[pmin(ceiling(te_p * 100), 100)]
tr_num$te_stem <- teu_tr$te_mean; te_num$te_stem <- teu_te$te_mean
tr_num$te_bigram <- teb_tr$te_mean; te_num$te_bigram <- teb_te$te_mean
meds <- map_dbl(tr_num, ~ median(.x, na.rm = TRUE))
tr_num <- map2_dfc(tr_num, meds, ~ if_else(is.na(.x), .y, .x))
te_num <- map2_dfc(te_num,  meds, ~ if_else(is.na(.x), .y, .x))

DF_tr <- train %>% select(Gender, MaritalStatus, PartTimeFullTime) %>% mutate(across(everything(), ~ replace_na(.x, "NA")))
DF_te <- test %>% select(Gender, MaritalStatus, PartTimeFullTime) %>% mutate(across(everything(), ~ replace_na(.x, "NA")))
for (v in names(DF_tr)) {
  lv <- levels(factor(DF_tr[[v]])); DF_tr[[v]] <- factor(DF_tr[[v]], lv); DF_te[[v]] <- factor(DF_te[[v]], lv)
}
mm_tr <- model.matrix(~ . - 1, DF_tr); mm_te <- model.matrix(~ . - 1, DF_te)
colnames(mm_tr) <- paste0("c_", colnames(mm_tr)); colnames(mm_te) <- paste0("c_", colnames(mm_te))

X_tab_tr <- cbind(as.matrix(tr_num), mm_tr); X_tab_te <- cbind(as.matrix(te_num), mm_te)

metrics_dollars <- function(pred, actual) {
  e <- pred - actual
  tibble(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)),
         MAPE = mean(abs(e) / pmax(actual, 1)) * 100,
         R2 = 1 - sum(e^2) / sum((actual - mean(actual))^2),
         RMSLE = sqrt(mean((log1p(pmax(pred, 0)) - log1p(actual))^2)))
}
res <- list(); preds <- list()

xgb_log <- function(Xtr, Xte, rounds = 3000) {
  p <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 4,
            lambda = 0.01, alpha = 0.01, subsample = 0.8, colsample_bytree = 0.8)
  set.seed(42)
  m <- xgb.train(p, xgb.DMatrix(Xtr, label = log1p(y_tr)), nrounds = rounds, verbose = 0)
  pmax(expm1(predict(m, Xte)), 0)
}

cat("\n== ARMS ==\n")
cat("REF: xgb log + tfidf\n")
Xr_tr <- cbind(X_tab_tr, as.matrix(T_tr)); Xr_te <- cbind(X_tab_te, as.matrix(T_te))
p_REF <- xgb_log(Xr_tr, Xr_te); preds$REF <- p_REF
res$REF <- metrics_dollars(p_REF, y_te) %>% mutate(arm = "REF_tfidf")

cat("FE1: xgb log + tfidf + new feats\n")
# new feats already inside X_tab (te_stem, te_bigram, init_pct_te, log_initial, ...) -> same as REF arm
# FE1 differs: ALSO keep tfidf -> identical to REF given new feats are in X_tab.
# so FE1 is the combined model (the X_tab already includes them); REF above also used them.
# To make REF honest (no new feats), rebuild REF without them:
new_cols <- c("log_initial", "init_zero", "init_x_delay", "word_count", "init_pct_te", "te_stem", "te_bigram")
Xo_tr <- X_tab_tr[, setdiff(colnames(X_tab_tr), new_cols)]
Xo_te <- X_tab_te[, setdiff(colnames(X_tab_te), new_cols)]
p_REF <- xgb_log(cbind(Xo_tr, as.matrix(T_tr)), cbind(Xo_te, as.matrix(T_te))); preds$REF <- p_REF
res$REF <- metrics_dollars(p_REF, y_te) %>% mutate(arm = "REF_tfidf")

p_FE1 <- xgb_log(Xr_tr, Xr_te); preds$FE1 <- p_FE1
res$FE1 <- metrics_dollars(p_FE1, y_te) %>% mutate(arm = "FE1_tfidf_new")

cat("FE2: xgb log + new feats only (no tfidf)\n")
p_FE2 <- xgb_log(X_tab_tr, X_tab_te); preds$FE2 <- p_FE2
res$FE2 <- metrics_dollars(p_FE2, y_te) %>% mutate(arm = "FE2_new_only")

cat("FE3: hurdle + tfidf + new feats\n")
thr <- 50000
set.seed(42)
m_cl <- xgb.train(list(objective = "binary:logistic", eta = 0.05, max_depth = 6,
                       subsample = 0.8, colsample_bytree = 0.8),
                  xgb.DMatrix(Xr_tr, label = as.numeric(y_tr > thr)), nrounds = 500, verbose = 0)
ph <- predict(m_cl, Xr_te)
ei <- which(y_tr > thr); ci <- which(y_tr <= thr)
set.seed(42)
m_exp <- xgb.train(list(objective = "reg:gamma", eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8),
                   xgb.DMatrix(Xr_tr[ei, ], label = y_tr[ei]), nrounds = 800, verbose = 0)
set.seed(42)
m_chp <- xgb.train(list(objective = "reg:squarederror", eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8),
                   xgb.DMatrix(Xr_tr[ci, ], label = log1p(y_tr[ci])), nrounds = 800, verbose = 0)
p_FE3 <- ph * pmax(predict(m_exp, Xr_te), 0) + (1 - ph) * pmax(expm1(predict(m_chp, Xr_te)), 0)
preds$FE3 <- p_FE3
res$FE3 <- metrics_dollars(p_FE3, y_te) %>% mutate(arm = "FE3_hurdle_new")

# ================= REPORT =================
tab <- bind_rows(res)
print(tab, n = 10)
cat("\nMAE by test quintile:\n")
for (nm in names(preds)) cat(sprintf("  %-4s %s\n", nm,
  paste(sprintf("%.0f", tapply(abs(preds[[nm]] - y_te), q_te, mean)), collapse = " ")))
cat("\nMAE by Initial-decile of test (is the surprise decile fixed?):\n")
init_dec_te <- ntile(test$InitialIncurredCalimsCost, 10)
for (nm in names(preds)) cat(sprintf("  %-4s %s\n", nm,
  paste(sprintf("%.0f", tapply(abs(preds[[nm]] - y_te), init_dec_te, mean)), collapse = " ")))
for (nm in names(res)) cat(sprintf('FINAL_METRICS arm=%s rmse=%.4f mae=%.4f mape=%.4f r2=%.6f rmsle=%.6f\n',
  nm, res[[nm]]$RMSE, res[[nm]]$MAE, res[[nm]]$MAPE, res[[nm]]$R2, res[[nm]]$RMSLE))
cat(sprintf("elapsed_min=%.1f\n", (proc.time() - t0)[3] / 60))
