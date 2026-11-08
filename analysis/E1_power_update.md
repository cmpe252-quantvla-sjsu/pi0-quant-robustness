# E1 power update (proposal 7.4)

## Inputs from the pilot

| Shift | BF16 per suite (Spatial / Object / Goal) | Episodes |
|---|---|---|
| lighting | 72% / 78% / 78% | 120 |
| background | 78% / 85% / 80% | 120 |
| layout | 70% / 70% / 50% | 60 |

E1c lighting, VE-W4 vs BF16: 120 matched pairs, n10 = 4, n01 = 3 (discordance 5.8%); a = 4.4%, b = 10.3%. Clean uses the same rates (no environment-B clean VE-W4 data).

## Projected precision of the pooled VE-W4 QRP (300 simulated studies each, tasks-only bootstrap, B = 1000)

| Scenario | Shifted per shift | Bootstrap SE (pp) | Spread over simulations (pp) | MDE at 80% power (pp) | Power at 5 pp |
|---|---|---|---|---|---|
| E1c | 300 | 1.17 | 1.24 | 3.3 | 99% |
| E1c | 450 | 1.08 | 1.12 | 3.0 | 100% |
| 2x | 300 | 1.67 | 1.63 | 4.7 | 85% |
| 2x | 450 | 1.54 | 1.54 | 4.3 | 90% |
| uneven | 300 | 1.16 | 1.18 | 3.3 | 99% |
| uneven | 450 | 1.09 | 1.16 | 3.1 | 100% |

Bootstrap SE: the SE our analysis would report. Spread over simulations: how much the estimate really varies between repeated studies; the two should be close. MDE = 2.8 x SE (proposal Eq. 8). Rule (proposal 7.4): raise the shifted sample size if the projected SE is above ~2.5 pp.

## Conclusion (Amina, Nov 6; for the design freeze)

- The projected SE of the pooled VE-W4 QRP is **about 1.2 pp at 300 per shift**, well below the
  2.5 pp rule, and only ~1.7 pp even if discordance is twice what E1c showed. The proposal's
  planning figure was ~2.0 pp (MDE 5.6 pp); the projected MDE is now ~3.3 pp.
- Going to 450 per shift barely helps (1.17 -> 1.08 pp): the clean gap enters the pooled QRP
  in full, so the clean grid limits precision. **Recommendation: keep 300 per shift.**
- The bootstrap SE and the real spread over simulated studies agree (1.17 vs 1.24 pp), so the
  tasks-only interval is close to right; it is slightly narrow, as the Oct 30 coverage check found.
- **Assumption:** clean VE-W4 discordance in environment B is assumed equal to E1c's lighting value, not measured (also in docs/REPORT_NOTES.md). Cross-check: VLAQuantBench's
  released clean Spatial logs give 6.5% for VE-W4 vs BF16 (200 episodes), close to E1c's 5.8%.

Reproduce (laptop, no GPU; ~1 min):

```
python scripts/power_update.py --e1 <folder with E1b/ and E1c/> --vqb ../VLAQuantBench --sims 300 --B 1000 --out-md analysis/E1_power_update.md
```
