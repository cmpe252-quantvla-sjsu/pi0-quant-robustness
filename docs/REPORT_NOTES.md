# Notes for the final report

Things we learned while running the study that the final report must state or correct.
Add to this list as you go; each item says where it comes from.

## Corrections to the proposal's wording

| Proposal | Correction | Source |
|---|---|---|
| §4.2: rounded weights "are stored as BF16" | The checkpoint keeps the **vision encoder in float32** (all 162 VE layers), so VE-W4 rounded weights are stored back in float32. Other components run in BF16 | E0b weight check, `RUN_LOG.md` |
| §4.3 / §6.3: the no-op "installs the hooks" | Weight-only settings use no forward hooks; the no-op runs the in-place weight-quantization path with an identity transform | Implementation notes 11–12 |
| §6.4: LIBERO-Plus needs "a different robosuite version" | Both pin robosuite 1.4.0; the clash is other dependency pins (numpy, transformers, gym) and the shared `libero` package name | Implementation note 8 |
| §7.1: bootstrap resamples tasks, then matched pairs within tasks | **Departure:** tasks-only resampling is primary; the two-stage scheme is reported as sensitivity. Coverage simulation: 99% vs. 92–94% for a nominal 95% interval. Decided Oct 30, before any core data | `configs/frozen_design.yaml`, `RUN_LOG.md` Decisions |
| §7.4: SE ≈ 2.3 points from the released logs | That figure used the two-stage scheme; with tasks-only it is ~1.4 points for the same 200 episodes | `scripts/qrp_bootstrap.py` on the released logs |
| §6.5: 12 s per episode, 35 GPU-hours with reruns | Measured 16.5 s per episode on the L4: ~34 GPU-hours before reruns | E0a, E0b |
| §6.2: one difficulty level per dimension | LIBERO-Plus level numbers are not comparable across suites, and many levels lack variants for some tasks. Missing tasks use the nearest level (ties lower, at most 2 tasks per suite), recorded per episode; a covered-tasks-only analysis is the pre-registered sensitivity check. If no single level gives BF16 40–85%, levels are chosen per suite | `configs/frozen_design.yaml`, `docs/variant_counts.md` |
| §6.3: visual shifts get "the base BDDL instruction" | They get the standard instruction (task file name with spaces), the same as clean and E0. The BDDL `(:language ...)` line is worded differently; only language rewrites use BDDL text | `quantvla/libero_plus.py`, `docs/JSONL_SCHEMA.md` |
| §6.4: "one software environment" | Environment B also gives slightly different rollouts from A on clean tasks (17/200 flips vs 1–2 within an environment), so A and B episodes are never paired | `RUN_LOG.md` E1a |
| Resources table: LeRobot 0.5.2 | 0.5.2 does not exist; we use 0.5.1 (torch 2.10.0) | `docs/SETUP.md` |

## Findings worth reporting

- **GPU non-determinism:** with identical model, code and seeds, two BF16 runs on the L4
  differ in ~4% of episodes' step counts and ~0.5–1% of outcomes. Exact episode parity is
  impossible; the no-op check uses a BF16 rerun as the chance level (E0b).
- **Matched noise matters:** with unmatched flow-matching noise, 67/200 outcomes flip between
  two BF16 runs although the success rates agree; with matched noise only 1–2 flip. This is
  why we added the episode-seed patch (0002) and why paired analysis has much less noise.
- **Harness fixes we contributed to VLAQuantBench usage:** the no-op preset, history-independent
  episode seeds, and a default output folder that can't resume the released logs.
