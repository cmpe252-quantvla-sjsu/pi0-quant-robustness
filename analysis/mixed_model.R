# Mixed-effects sensitivity model (proposal Eq. 7), fitted with lme4.
#
#   logit P(y = 1) = b0 + b1*Q + b2*S + b3*Q*S + u_task + v_pair
#   u_task ~ N(0, s_u^2), v_pair ~ N(0, s_v^2)
#
# Q = 1 for the quantized setting (0 = BF16), S = 1 for a shifted condition (0 = clean).
# b3 is the change in the quantization effect under shift on the log-odds scale; a negative
# b3 means extra harm under shift (the log-odds counterpart of a positive QRP). The pair
# effect ties a BF16 episode to its matched quantized episode.
#
# Fitted for each quantized setting (VE-W4, W4, W8) on clean versus pooled visual shifts, and
# on clean versus each shift separately. Wald 95% intervals; descriptive sensitivity check,
# no multiplicity correction (H1 is decided by the QRP bootstrap, analysis/qrp.md).
#
# Check: BF16 and Q agree on most matched episodes, so the pair variance is large and the
# Laplace approximation is the weakest part of the fit. We therefore also fit a conditional
# logistic regression with each matched pair as a stratum (survival::clogit), which needs no
# random-effect approximation; its Q x S term estimates the same b3 from discordant pairs only.
#
#   python scripts/mixed_model_data.py configs/qrp/study_manifest.yaml --base-dir . --out analysis/mixed_model_long.csv
#   Rscript analysis/mixed_model.R analysis/mixed_model_long.csv analysis/mixed_model.md analysis/mixed_model.json

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: Rscript analysis/mixed_model.R <long.csv> <out.md> <out.json>")
d <- read.csv(args[1], stringsAsFactors = FALSE)

fit_one <- function(dd) {
  dd$task <- factor(dd$task)
  dd$pair <- factor(dd$pair)
  warn <- character(0)
  m <- withCallingHandlers(
    glmer(success ~ Q * shifted + (1 | task) + (1 | pair), data = dd, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") })
  co <- summary(m)$coefficients
  b <- co["Q:shifted", "Estimate"]; se <- co["Q:shifted", "Std. Error"]
  vc <- as.data.frame(VarCorr(m))
  conv <- m@optinfo$conv$lme4$messages
  cl <- clogit(success ~ Q + Q:shifted + strata(pair), data = dd)
  cco <- summary(cl)$coefficients
  cb <- cco["Q:shifted", "coef"]; cse <- cco["Q:shifted", "se(coef)"]
  sr <- function(q, s) 100 * mean(dd$success[dd$Q == q & dd$shifted == s])
  list(
    n_episodes = nrow(dd) / 2,
    b1_Q = unname(co["Q", "Estimate"]), b1_se = unname(co["Q", "Std. Error"]),
    b3 = b, b3_se = se, b3_lo = b - 1.96 * se, b3_hi = b + 1.96 * se,
    p = unname(2 * pnorm(-abs(b / se))),
    or3 = exp(b), or3_lo = exp(b - 1.96 * se), or3_hi = exp(b + 1.96 * se),
    clogit_b3 = cb, clogit_lo = cb - 1.96 * cse, clogit_hi = cb + 1.96 * cse,
    sd_task = vc$sdcor[vc$grp == "task"], sd_pair = vc$sdcor[vc$grp == "pair"],
    singular = isSingular(m),
    messages = unique(c(warn, if (is.null(conv)) character(0) else conv)),
    sr = list(bf16_clean = sr(0, 0), q_clean = sr(1, 0), bf16_shift = sr(0, 1), q_shift = sr(1, 1))
  )
}

shifts <- sort(unique(d$condition[d$shifted == 1]))
res <- list()
for (q in c("VE-W4", "W4", "W8")) {
  dq <- d[d$comparison == q, ]
  if (nrow(dq) == 0) next
  scopes <- c("pooled visual", shifts)
  for (sc in scopes) {
    dd <- if (sc == "pooled visual") dq else dq[dq$condition %in% c("clean", sc), ]
    r <- fit_one(dd)
    r$setting <- q; r$scope <- sc
    res[[length(res) + 1]] <- r
    cat(sprintf("%-6s %-14s b3 = %+.3f [%+.3f, %+.3f]  p = %.3f%s\n", q, sc, r$b3, r$b3_lo, r$b3_hi, r$p,
                if (length(r$messages) || r$singular) "  (see notes)" else ""))
  }
}

f2 <- function(x) sprintf("%+.2f", x)
L <- c("# Mixed-effects sensitivity model (proposal Eq. 7)", "",
       "Logistic GLMM fitted with lme4 (`glmer`, Laplace, bobyqa): success ~ Q * S + (1 | task) + (1 | pair),",
       "on BF16 versus one quantized setting, clean versus shifted. `b3` is the Q x S interaction on the",
       "log-odds scale (negative = extra harm under shift); OR = exp(b3). Wald 95% intervals, descriptive",
       "(no multiplicity correction). Episodes are the matched pairs used by `analysis/qrp.md`.",
       sprintf("lme4 %s, %s.", as.character(packageVersion("lme4")), R.version.string), "",
       "Check column: the same b3 from a conditional logistic regression with matched pairs as strata",
       "(`survival::clogit`), which needs no random-effect approximation.", "",
       "| setting | clean vs | episodes | SR BF16 clean / shift | SR Q clean / shift | b1 (Q) | b3 (Q x S) [95% CI] | OR | p | sd task | sd pair | check: clogit b3 [95% CI] |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|")
notes <- character(0)
for (r in res) {
  L <- c(L, sprintf("| %s | %s | %d | %.1f / %.1f | %.1f / %.1f | %s | %s [%s, %s] | %.2f | %.3f | %.2f | %.2f | %s [%s, %s] |",
                    r$setting, r$scope, as.integer(r$n_episodes), r$sr$bf16_clean, r$sr$bf16_shift,
                    r$sr$q_clean, r$sr$q_shift, f2(r$b1_Q), f2(r$b3), f2(r$b3_lo), f2(r$b3_hi),
                    r$or3, r$p, r$sd_task, r$sd_pair, f2(r$clogit_b3), f2(r$clogit_lo), f2(r$clogit_hi)))
  if (r$singular || length(r$messages))
    notes <- c(notes, sprintf("- %s, clean vs %s: %s", r$setting, r$scope,
                              paste(c(if (r$singular) "singular fit (a variance estimated at ~0)", r$messages),
                                    collapse = "; ")))
}
if (length(notes)) L <- c(L, "", "Fit notes:", notes)
writeLines(L, args[2], useBytes = TRUE)
write_json(res, args[3], auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("wrote", args[2], "and", args[3], "\n")
