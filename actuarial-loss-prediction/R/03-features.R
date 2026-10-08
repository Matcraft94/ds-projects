# Tabular features, winsorization, one-hot dummies, design matrices.

library(tidyverse)

tab_features <- function(d, txt_df = NULL, extras = TRUE) {
  out <- d %>% transmute(
    Age, DependentChildren, DependentsOther, WeeklyWages, HoursWorkedPerWeek,
    DaysWorkedPerWeek, InitialIncurredCalimsCost, Days_To_Report, AccidentYear,
    WeeklyWagesPerHour = if_else(HoursWorkedPerWeek > 0,
                                 WeeklyWages / HoursWorkedPerWeek, 0),
    DependentsTotal = DependentChildren + DependentsOther
  )
  if (extras && !is.null(txt_df)) {
    out <- out %>% mutate(
      log_initial = log1p(InitialIncurredCalimsCost),
      init_zero = as.numeric(InitialIncurredCalimsCost == 0),
      init_x_delay = log1p(InitialIncurredCalimsCost) * log1p(pmax(Days_To_Report, 0)),
      word_count = vapply(strsplit(txt_df$txt, " "), length, integer(1))
    )
  }
  out
}

winsorize <- function(tr, te, probs = c(0.01, 0.99)) {
  lo <- map_dbl(tr, ~ quantile(.x, probs[1], na.rm = TRUE))
  hi <- map_dbl(tr, ~ quantile(.x, probs[2], na.rm = TRUE))
  cl <- function(d) map2_dfc(d, seq_along(d), ~ pmin(pmax(.x, lo[.y]), hi[.y]))
  list(train = cl(tr), test = cl(te))
}

impute_median <- function(tr, te) {
  meds <- map_dbl(tr, ~ median(.x, na.rm = TRUE))
  list(
    train = map2_dfc(tr, meds, ~ if_else(is.na(.x), .y, .x)),
    test  = map2_dfc(te,  meds, ~ if_else(is.na(.x), .y, .x))
  )
}

cat_dummies <- function(train, test, cats = c("Gender", "MaritalStatus", "PartTimeFullTime")) {
  prep <- function(d) d %>% select(all_of(cats)) %>%
    mutate(across(everything(), ~ replace_na(.x, "NA")))
  DF_tr <- prep(train); DF_te <- prep(test)
  for (v in cats) {
    lv <- levels(factor(DF_tr[[v]]))
    DF_tr[[v]] <- factor(DF_tr[[v]], lv)
    DF_te[[v]] <- factor(DF_te[[v]], lv)
  }
  mm <- model.matrix(~ . - 1, data = DF_tr)
  mm_te <- model.matrix(~ . - 1, data = DF_te)
  colnames(mm) <- paste0("c_", colnames(mm))
  colnames(mm_te) <- paste0("c_", colnames(mm_te))
  list(train = mm, test = mm_te)
}

# Full numeric + dummy matrix
design_matrix <- function(tr_num, te_num, tr_mm, te_mm, extra_cols = NULL) {
  X_tr <- cbind(as.matrix(tr_num), tr_mm)
  X_te <- cbind(as.matrix(te_num), te_mm)
  if (!is.null(extra_cols)) {
    stopifnot(all(colnames(X_tr) == colnames(X_te)))
  }
  list(train = X_tr, test = X_te)
}
