# Text features: tokenization, TF-IDF matrices, target encodings (in-sample
# and out-of-fold). Vocabularies, IDFs and TE maps always fitted on train only.

library(tidyverse)
library(tidytext)
library(SnowballC)
library(stopwords)

tokenize_claims <- function(txt_df) {
  txt_df %>%
    unnest_tokens(word, txt) %>%
    anti_join(data.frame(word = stopwords::stopwords("en")), by = "word") %>%
    mutate(stem = SnowballC::wordStem(word))
}

bigrams_of <- function(txt_df) {
  txt_df %>%
    unnest_tokens(bigram, txt, token = "ngrams", n = 2) %>%
    filter(!is.na(bigram))
}

build_vocab <- function(tokens, col, min_freq = 50, n_max = 200) {
  tokens %>% count(.data[[col]], sort = TRUE) %>%
    filter(n >= min_freq) %>% slice_head(n = n_max) %>% pull(.data[[col]])
}

pad_rows <- function(wide, txt_df) {
  full <- tibble(claim_id = seq_len(nrow(txt_df)))
  out <- full %>% left_join(wide, by = "claim_id") %>% select(-claim_id)
  out[is.na(out)] <- 0
  as.data.frame(out)
}

make_tfidf <- function(tr_tokens, te_tokens, tr_bigrams, te_bigrams,
                       vocab_u, vocab_b, tr_txt, te_txt) {
  doc_n <- nrow(tr_txt)
  idf_u <- tr_tokens %>% filter(stem %in% vocab_u) %>% distinct(claim_id, stem) %>%
    count(stem, name = "df_doc") %>% mutate(idf = log(doc_n / df_doc))
  idf_b <- tr_bigrams %>% filter(bigram %in% vocab_b) %>% distinct(claim_id, bigram) %>%
    count(bigram, name = "df_doc") %>% mutate(idf = log(doc_n / df_doc))
  ttl <- function(txt_df) txt_df %>%
    group_by(claim_id) %>%
    summarise(ntot = sum(vapply(strsplit(txt, " "), length, integer(1))), .groups = "drop")
  build <- function(tokens, txt_df, idf, col, prefix) {
    L <- ttl(txt_df)
    tokens %>% filter(.data[[col]] %in% idf[[col]]) %>%
      count(claim_id, .data[[col]], name = "n") %>%
      left_join(L, by = "claim_id") %>% left_join(idf, by = col) %>%
      mutate(v = (n / ntot) * idf) %>%
      pivot_wider(id_cols = claim_id, names_from = .data[[col]],
                  values_from = v, values_fill = 0, names_prefix = prefix)
  }
  u_tr <- build(tr_tokens, tr_txt, idf_u, "stem", "CD_")
  u_te <- build(te_tokens, te_txt, idf_u, "stem", "CD_")
  b_tr <- build(tr_bigrams, tr_txt, idf_b, "bigram", "CB_")
  b_te <- build(te_bigrams, te_txt, idf_b, "bigram", "CB_")
  list(
    train = pad_rows(u_tr %>% full_join(b_tr, by = "claim_id"), tr_txt),
    test  = pad_rows(u_te %>% full_join(b_te, by = "claim_id"), te_txt)
  )
}

make_counts <- function(tr_tokens, te_tokens, vocab_u, tr_txt, te_txt) {
  wide <- function(tokens, txt_df) {
    tokens %>% filter(stem %in% vocab_u) %>% count(claim_id, stem, name = "n") %>%
      pivot_wider(id_cols = claim_id, names_from = stem, values_from = n,
                  values_fill = 0, names_prefix = "CD_")
  }
  list(train = pad_rows(wide(tr_tokens, tr_txt), tr_txt),
       test  = pad_rows(wide(te_tokens, te_txt), te_txt))
}

# Smoothed (K) target encoding of text terms against log1p(cost).
# method="insample": map fitted on all train (what round 3 used on train rows)
# method="oof":      5-fold out-of-fold maps for train rows (honest; round 4)
target_encode_text <- function(tr_tokens, te_tokens, vocab, y_train_log,
                               col = "stem", K = 50,
                               method = c("insample", "oof"),
                               folds = 5, seed = 42) {
  method <- match.arg(method)
  gm <- mean(y_train_log)
  n_tr <- length(y_train_log)

  map_of <- function(ids) {
    tr_tokens %>% filter(.data[[col]] %in% vocab, claim_id %in% ids) %>%
      left_join(tibble(claim_id = ids, yl = y_train_log[ids]), by = "claim_id") %>%
      group_by(.data[[col]]) %>%
      summarise(m = mean(yl), n = dplyr::n(), .groups = "drop") %>%
      mutate(te = (n * m + K * gm) / (n + K))
  }
  apply_map_train <- function(mp, ids) {
    agg <- tr_tokens %>% filter(claim_id %in% ids, .data[[col]] %in% mp[[col]]) %>%
      inner_join(mp, by = col) %>%
      group_by(claim_id) %>% summarise(v = mean(te), .groups = "drop")
    v <- rep(gm, length(ids)); v[match(agg$claim_id, ids)] <- agg$v; v
  }
  apply_map_test <- function(mp, tokens, n_rows) {
    agg <- tokens %>% filter(.data[[col]] %in% mp[[col]]) %>%
      inner_join(mp, by = col) %>%
      group_by(claim_id) %>% summarise(v = mean(te), .groups = "drop")
    v <- rep(gm, n_rows); v[agg$claim_id] <- agg$v; v
  }

  if (method == "insample") {
    map_full <- map_of(seq_len(n_tr))
    tr_v <- apply_map_train(map_full, seq_len(n_tr))
  } else {
    set.seed(seed)
    fold_id <- sample(rep(seq_len(folds), length.out = n_tr))
    tr_v <- numeric(n_tr)
    for (f in seq_len(folds)) {
      in_f  <- which(fold_id == f)
      out_f <- setdiff(seq_len(n_tr), in_f)
      tr_v[in_f] <- apply_map_train(map_of(out_f), in_f)
    }
    map_full <- map_of(seq_len(n_tr))
  }
  te_v <- apply_map_test(map_full, te_tokens, nrow(te_tokens %>% distinct(claim_id)))
  list(train = tr_v, test = te_v)
}

# Initial-percentile target encoding (the full nonlinear Initial->cost curve)
target_encode_initial <- function(initial_train, y_train_log, initial_test,
                                  n_bucket = 100, method = c("insample", "oof"),
                                  folds = 5, seed = 42) {
  method <- match.arg(method)
  gm <- mean(y_train_log)
  n_tr <- length(y_train_log)
  map_of <- function(ids) {
    F_ecdf <- ecdf(initial_train[ids])
    tibble(yl = y_train_log[ids]) %>%
      mutate(b = pmin(ceiling(F_ecdf(initial_train[ids]) * n_bucket), n_bucket)) %>%
      group_by(b) %>% summarise(te = mean(yl), .groups = "drop")
  }
  to_te <- function(mp, p) {
    mp$te[pmin(ceiling(p * n_bucket), n_bucket)]
  }
  if (method == "insample") {
    mp <- map_of(seq_len(n_tr))
    F_all <- ecdf(initial_train)
    tr_v <- to_te(mp, F_all(initial_train))
  } else {
    set.seed(seed)
    fold_id <- sample(rep(seq_len(folds), length.out = n_tr))
    tr_v <- numeric(n_tr)
    for (f in seq_len(folds)) {
      in_f <- which(fold_id == f); out_f <- setdiff(seq_len(n_tr), in_f)
      mp <- map_of(out_f)
      tr_v[in_f] <- to_te(mp, ecdf(initial_train[out_f])(initial_train[in_f]))
    }
    mp <- map_of(seq_len(n_tr))
  }
  te_v <- to_te(mp, ecdf(initial_train)(initial_test))
  list(train = tr_v, test = te_v)
}
