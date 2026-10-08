# Production pipeline — reproducible via `targets::tar_make()`.
# Single source of truth: R/ functions; experiments consume shared targets.

library(targets)
library(tarchetypes)

tar_option_set(packages = c("tidyverse", "tidymodels", "tidytext", "SnowballC",
                            "stopwords", "xgboost", "glmnet", "mgcv", "rmarkdown"))

for (f in list.files("R", full.names = TRUE)) source(f)

list(
  # ---- data layer ----
  tar_target(raw_csv, "Data/actuarial_loss/train.csv", format = "file"),
  tar_target(raw, load_raw(raw_csv)),
  tar_target(prep, add_derived(raw)),
  tar_target(split, make_split(prep)),
  tar_target(train_df, split$train),
  tar_target(test_df, split$test),
  tar_target(y_train, train_df$UltimateIncurredClaimCost),
  tar_target(y_test, test_df$UltimateIncurredClaimCost),
  tar_target(q_test, ntile(y_test, 5)),

  # ---- text layer ----
  tar_target(train_txt, train_df %>% transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))),
  tar_target(test_txt,  test_df %>%  transmute(claim_id = row_number(), txt = clean_text(ClaimDescription))),
  tar_target(tr_tokens, tokenize_claims(train_txt)),
  tar_target(te_tokens, tokenize_claims(test_txt)),
  tar_target(tr_bigrams, bigrams_of(train_txt)),
  tar_target(te_bigrams, bigrams_of(test_txt)),
  tar_target(vocab_u, build_vocab(tr_tokens, "stem", 50, 200)),
  tar_target(vocab_b, build_vocab(tr_bigrams, "bigram", 50, 100)),

  # ---- feature matrices ----
  tar_target(tfidf, make_tfidf(tr_tokens, te_tokens, tr_bigrams, te_bigrams,
                               vocab_u, vocab_b, train_txt, test_txt)),
  tar_target(counts, make_counts(tr_tokens, te_tokens, vocab_u, train_txt, test_txt)),
  tar_target(dummies, cat_dummies(train_df, test_df)),
  tar_target(tab_base_tr, tab_features(train_df, train_txt, extras = FALSE)),
  tar_target(tab_base_te, tab_features(test_df, test_txt, extras = FALSE)),
  tar_target(tab_med, impute_median(tab_base_tr, tab_base_te)),
  tar_target(yl_train, log1p(y_train)),

  # ---- experiments (each returns metrics and writes results/*.csv) ----
  tar_target(round1, {
    source("experiments/01-improved-model.R")
    run_round1(train_df, test_df, tfidf, counts, dummies, tab_med,
               y_train, y_test, q_test)
  }),
  tar_target(round2, {
    source("experiments/02-glm-gam-families.R")
    run_round2(train_df, test_df, tfidf, dummies, tab_med, y_train, y_test, q_test)
  }),
  tar_target(round3, {
    source("experiments/03-distribution-features.R")
    run_round3(train_df, test_df, train_txt, test_txt,
               tr_tokens, te_tokens, tr_bigrams, te_bigrams,
               vocab_u, vocab_b, tfidf, dummies, y_train, y_test, q_test,
               ntile(test_df$InitialIncurredCalimsCost, 10))
  }),
  tar_target(round4, {
    source("experiments/04-final-squeeze.R")
    run_round4(train_df, test_df, train_txt, test_txt,
               tr_tokens, te_tokens, tr_bigrams, te_bigrams,
               vocab_u, vocab_b, dummies, y_train, y_test, q_test)
  }),

  # ---- living report (renders from the results the pipeline produces) ----
  tar_render("report", "reports/improvement-2026.Rmd")
)
