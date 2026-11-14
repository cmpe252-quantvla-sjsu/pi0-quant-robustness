# QRP analysis (10,000 bootstrap replicates, seed 0, resampling: tasks only)

## Success rates (descriptive)

| condition | setting | n | SR % | Wilson 95% | mean steps (successes) |
|---|---|---|---|---|---|
| background | BF16 | 300 | 79.3 | [74.4, 83.5] | 117.5 |
| background | VE-W4 | 300 | 79.7 | [74.8, 83.8] | 118.2 |
| background | W4 | 300 | 77.7 | [72.6, 82.0] | 123.7 |
| background | W8 | 120 | 75.8 | [67.4, 82.6] | 117.1 |
| clean | BF16 | 600 | 82.8 | [79.6, 85.6] | 119.5 |
| clean | VE-W4 | 600 | 83.0 | [79.8, 85.8] | 119.6 |
| clean | W4 | 600 | 80.5 | [77.1, 83.5] | 122.5 |
| clean | W8 | 600 | 82.3 | [79.1, 85.2] | 120.0 |
| layout | BF16 | 300 | 61.3 | [55.7, 66.7] | 121.3 |
| layout | VE-W4 | 300 | 59.7 | [54.0, 65.1] | 121.1 |
| layout | W4 | 300 | 54.7 | [49.0, 60.2] | 122.9 |
| layout | W8 | 120 | 60.8 | [51.9, 69.1] | 121.2 |
| lighting | BF16 | 300 | 72.3 | [67.0, 77.1] | 121.4 |
| lighting | VE-W4 | 300 | 71.3 | [66.0, 76.2] | 120.4 |
| lighting | W4 | 300 | 66.0 | [60.5, 71.1] | 124.4 |
| lighting | W8 | 120 | 70.0 | [61.3, 77.5] | 119.8 |

## Quantization gaps (BF16 - setting, pp; paired task-clustered 95% CI)

| condition/setting | matched n | gap | n10 (only BF16 ok) | n01 (only Q ok) | discordance |
|---|---|---|---|---|---|
| background/VE-W4 | 300 | -0.3 [-3.3, +2.7] | 8 | 9 | 5.7% |
| background/W4 | 300 | +1.7 [-3.0, +6.3] | 28 | 23 | 17.0% |
| background/W8 | 120 | +0.8 [-1.7, +4.2] | 3 | 2 | 4.2% |
| clean/VE-W4 | 600 | -0.2 [-1.8, +1.7] | 11 | 12 | 3.8% |
| clean/W4 | 600 | +2.3 [-0.2, +4.8] | 59 | 45 | 17.3% |
| clean/W8 | 600 | +0.5 [-1.0, +2.2] | 20 | 17 | 6.2% |
| layout/VE-W4 | 300 | +1.7 [+0.3, +3.3] | 6 | 1 | 2.3% |
| layout/W4 | 300 | +6.7 [+2.7, +10.7] | 33 | 13 | 15.3% |
| layout/W8 | 120 | +0.0 [-3.3, +3.3] | 2 | 2 | 3.3% |
| lighting/VE-W4 | 300 | +1.0 [-1.0, +3.0] | 6 | 3 | 3.0% |
| lighting/W4 | 300 | +6.3 [+2.0, +10.7] | 39 | 20 | 19.7% |
| lighting/W8 | 120 | +0.0 [-3.3, +3.3] | 3 | 3 | 5.0% |

## QRP (gap under shift - clean gap, pp)

| setting | contrast | QRP [95% CI] | SE | Wald p |
|---|---|---|---|---|
| VE-W4 | lighting | +1.2 [-1.0, +3.3] | 1.08 | 0.281 |
| VE-W4 | background | -0.2 [-4.3, +3.8] | 2.06 | 0.936 |
| VE-W4 | layout | +1.8 [-0.3, +4.2] | 1.15 | 0.111 |
| VE-W4 | pooled_visual | +0.9 [-1.4, +3.2] | 1.16 | 0.417 |
| W4 | lighting | +4.0 [-1.2, +9.2] | 2.61 | 0.126 |
| W4 | background | -0.7 [-5.2, +3.8] | 2.32 | 0.774 |
| W4 | layout | +4.3 [-0.8, +9.3] | 2.58 | 0.093 |
| W4 | pooled_visual | +2.6 [-1.1, +6.1] | 1.84 | 0.166 |
| W8 | lighting | -0.5 [-4.2, +3.2] | 1.84 | 0.785 |
| W8 | background | +0.3 [-2.8, +3.5] | 1.61 | 0.835 |
| W8 | layout | -0.5 [-4.0, +2.8] | 1.72 | 0.772 |
| W8 | pooled_visual | -0.2 [-2.4, +1.9] | 1.10 | 0.840 |

## Primary hypothesis (H1)

Pooled visual QRP for VE-W4: **+0.9 [-1.4, +3.2]** points

Decision: **NO PRACTICALLY IMPORTANT PENALTY (interval includes 0, stays below +5)**

## Sensitivity analysis (resampling: tasks, then units within task)

Same point estimates; only the bootstrap interval changes.

| setting | pooled visual QRP [95% CI] |
|---|---|
| VE-W4 | +0.9 [-2.1, +3.8] |
| W4 | +2.6 [-2.8, +7.9] |
| W8 | -0.2 [-3.6, +3.2] |

H1 under this scheme: **NO PRACTICALLY IMPORTANT PENALTY (interval includes 0, stays below +5)**

## Sensitivity analysis: tasks without fallback episodes only

Per shift, tasks with any fallback episode are dropped and the clean gap is computed on the same remaining tasks.

| condition | excluded tasks | tasks kept |
|---|---|---|
| lighting | goal 1, goal 6, object 1, object 5, spatial 1 | 25 |
| background | goal 4, goal 6, spatial 8 | 27 |
| layout | object 4, spatial 6 | 28 |

| setting | pooled visual QRP [95% CI] |
|---|---|
| VE-W4 | +1.3 [-1.2, +3.6] |
| W4 | +2.2 [-1.6, +5.9] |
| W8 | -0.7 [-2.9, +1.5] |

H1 on covered tasks only: **NO PRACTICALLY IMPORTANT PENALTY (interval includes 0, stays below +5)**

## Secondary tests (Holm, m = 4, exploratory)

| test | Wald p | Holm threshold | reject |
|---|---|---|---|
| VE-W4 QRP layout | 0.1111 | 0.0125 | no |
| H2: W4 pooled QRP | 0.1659 | 0.0167 | no |
| VE-W4 QRP lighting | 0.2807 | 0.0250 | no |
| VE-W4 QRP background | 0.9356 | 0.0500 | no |
