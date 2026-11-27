# E5 failure modes (unblinded)

One video per failed episode; a BF16 failure that is discordant with both VE-W4 and W4 is
counted in both pairs. Counts of the primary code. `first: WRONG_OBJECT` counts videos whose
note starts with that tag: an earlier attempt went for another object (codebook rule 1).
For a shared video it is counted if either coder wrote the tag.

| condition | pair | failed | n | first: WRONG_OBJECT | NO_REACH | WRONG_OBJECT | GRASP_MISS | SLIP | EXTRA_OBJECT | PLACE_MISS | TIMEOUT | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| background | VE-W4 | BF16 | 9 | 0 | 0 | 0 | 3 | 2 | 2 | 1 | 1 | 0 |
| background | VE-W4 | VE-W4 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 6 | 0 | 0 |
| background | W4 | BF16 | 15 | 0 | 0 | 0 | 6 | 4 | 2 | 3 | 0 | 0 |
| background | W4 | W4 | 15 | 0 | 0 | 0 | 7 | 1 | 0 | 6 | 1 | 0 |
| clean | VE-W4 | BF16 | 12 | 0 | 0 | 0 | 2 | 3 | 1 | 6 | 0 | 0 |
| clean | VE-W4 | VE-W4 | 11 | 0 | 0 | 0 | 2 | 3 | 1 | 5 | 0 | 0 |
| clean | W4 | BF16 | 15 | 0 | 0 | 1 | 5 | 2 | 0 | 7 | 0 | 0 |
| clean | W4 | W4 | 15 | 0 | 0 | 0 | 6 | 5 | 0 | 4 | 0 | 0 |
| layout | VE-W4 | BF16 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| layout | VE-W4 | VE-W4 | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 |
| layout | W4 | BF16 | 13 | 0 | 0 | 1 | 5 | 1 | 1 | 5 | 0 | 0 |
| layout | W4 | W4 | 15 | 0 | 1 | 0 | 6 | 0 | 0 | 8 | 0 | 0 |
| lighting | VE-W4 | BF16 | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| lighting | VE-W4 | VE-W4 | 6 | 0 | 0 | 1 | 1 | 2 | 0 | 2 | 0 | 0 |
| lighting | W4 | BF16 | 15 | 0 | 0 | 0 | 7 | 4 | 1 | 3 | 0 | 0 |
| lighting | W4 | W4 | 15 | 0 | 0 | 2 | 7 | 2 | 1 | 2 | 1 | 0 |

Agreement on 30 shared videos: Cohen's kappa = 0.76
