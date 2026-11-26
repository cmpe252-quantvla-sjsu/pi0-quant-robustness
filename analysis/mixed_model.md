# Mixed-effects sensitivity model (proposal Eq. 7)

Logistic GLMM fitted with lme4 (`glmer`, Laplace, bobyqa): success ~ Q * S + (1 | task) + (1 | pair),
on BF16 versus one quantized setting, clean versus shifted. `b3` is the Q x S interaction on the
log-odds scale (negative = extra harm under shift); OR = exp(b3). Wald 95% intervals, descriptive
(no multiplicity correction). Episodes are the matched pairs used by `analysis/qrp.md`.
lme4 2.1.0, R version 4.6.2 (2026-10-29 ucrt).

Check column: the same b3 from a conditional logistic regression with matched pairs as strata
(`survival::clogit`), which needs no random-effect approximation.

| setting | clean vs | episodes | SR BF16 clean / shift | SR Q clean / shift | b1 (Q) | b3 (Q x S) [95% CI] | OR | p | sd task | sd pair | check: clogit b3 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VE-W4 | pooled visual | 1500 | 82.8 / 71.0 | 83.0 / 70.2 | +0.14 | -0.61 [-1.98, +0.77] | 0.55 | 0.386 | 0.00 | 27.35 | -0.52 [-1.59, +0.56] |
| VE-W4 | background | 900 | 82.8 / 79.3 | 83.0 / 79.7 | +0.13 | +0.07 [-1.53, +1.67] | 1.07 | 0.930 | 0.84 | 23.02 | +0.03 [-1.22, +1.29] |
| VE-W4 | layout | 900 | 82.8 / 61.3 | 83.0 / 59.7 | +0.15 | -1.04 [-2.66, +0.59] | 0.35 | 0.212 | 1.39 | 28.68 | -1.88 [-4.15, +0.39] |
| VE-W4 | lighting | 900 | 82.8 / 72.3 | 83.0 / 71.3 | +0.14 | -0.78 [-2.46, +0.89] | 0.46 | 0.360 | 1.08 | 26.89 | -0.78 [-2.39, +0.83] |
| W4 | pooled visual | 1500 | 82.8 / 71.0 | 80.5 / 66.1 | -0.28 | -0.19 [-0.68, +0.30] | 0.83 | 0.441 | 1.06 | 2.49 | -0.31 [-0.82, +0.20] |
| W4 | background | 900 | 82.8 / 79.3 | 80.5 / 77.7 | -0.26 | +0.09 [-0.55, +0.72] | 1.09 | 0.790 | 0.99 | 1.99 | +0.07 [-0.60, +0.75] |
| W4 | layout | 900 | 82.8 / 61.3 | 80.5 / 54.7 | -0.29 | -0.35 [-0.99, +0.29] | 0.70 | 0.281 | 1.29 | 2.68 | -0.66 [-1.41, +0.09] |
| W4 | lighting | 900 | 82.8 / 72.3 | 80.5 / 66.0 | -0.26 | -0.30 [-0.92, +0.31] | 0.74 | 0.330 | 0.95 | 2.17 | -0.40 [-1.06, +0.27] |
| W8 | pooled visual | 960 | 82.8 / 69.2 | 82.3 / 68.9 | -0.32 | +0.18 [-1.30, +1.67] | 1.20 | 0.810 | 0.00 | 22.00 | +0.03 [-1.17, +1.23] |
| W8 | background | 720 | 82.8 / 76.7 | 82.3 / 75.8 | -0.30 | -0.16 [-2.26, +1.93] | 0.85 | 0.878 | 0.71 | 19.82 | -0.24 [-2.15, +1.66] |
| W8 | layout | 720 | 82.8 / 60.8 | 82.3 / 60.8 | -0.31 | +0.31 [-1.99, +2.61] | 1.36 | 0.793 | 0.00 | 20.73 | +0.16 [-1.90, +2.23] |
| W8 | lighting | 720 | 82.8 / 70.0 | 82.3 / 70.0 | -0.30 | +0.30 [-1.60, +2.20] | 1.35 | 0.757 | 0.64 | 19.85 | +0.16 [-1.56, +1.89] |

Fit notes:
- VE-W4, clean vs pooled visual: singular fit (a variance estimated at ~0); boundary (singular) fit: see help('isSingular')
- W8, clean vs pooled visual: singular fit (a variance estimated at ~0); boundary (singular) fit: see help('isSingular')
- W8, clean vs layout: singular fit (a variance estimated at ~0); boundary (singular) fit: see help('isSingular')
