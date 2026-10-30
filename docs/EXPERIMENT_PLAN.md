# Experiment plan

Step-by-step runbook for E0–E7 (proposal Section 5, Table 2). E0 and E1 are pilots: their
episodes are **never** used in a hypothesis test. Every run gets a line in `RUN_LOG.md`.

## Timeline (revised Oct 29)

We are about a week behind the proposal's Table 4, but past the environment gate (E0a).

| Step | Owner | Target |
|---|---|---|
| E0a: BF16 reproduces the published baseline | Smitha | done Oct 29 |
| E0b: no-op parity check | Amina | Oct 30 |
| Variant sampler + LIBERO-Plus runner | Smitha | ~Nov 3 |
| Environment B + E1 pilot | Both | Nov 4–6 |
| Design freeze (`configs/frozen_design.yaml`, tag `design-freeze`) | Both | ~Nov 7 |
| Core runs E2–E4 | Both | Nov 8–18 |
| E5 failure coding; optional E6/E7 | Both | Nov 15–20 |
| **Experiment freeze** (no new rollouts) | Both | **Nov 20** |
| Analysis and report | Both | Nov 21–Dec 5 |
| Final presentation / final report | Both | Dec 9 / Dec 16 |

**Nov 4 checkpoint:** if environment B is not working by Nov 4, cut scope instead of
squeezing the schedule (for example, drop W8 or one suite).

## Compute

E0a measured **16.5 s per episode** on average (median 13.2 s) and 8.4 GB peak VRAM on the
L4, more than the proposal's 12 s planning figure. At 16.5 s, the 6,300 core episodes take
about 29 GPU-hours, plus pilots and reruns. E1 should check whether two episodes can run at
once on the 23 GB card.

## Commands: two things to always do

- Pass `--episodes 20`. The default is 50 per task.
- Pass `--out` with a **file name**, not a folder, following `results/README.md`.

---

## E0: harness validation in environment A (pilot)

**Goal:** the stack reproduces the published BF16 baseline, and the no-op changes nothing.

| Run | What | Episodes | Owner | Status |
|---|---|---|---|---|
| E0a | BF16, clean Spatial, unpatched VLAQuantBench | 500 (first 200 used) | Smitha | **done**: 76.0% (152/200) vs. 78.0% published, inside the 71.8–83.2% target |
| E0b-1 | BF16, clean Spatial, **patched** VLAQuantBench (parity reference for E0b only) | 200 | Amina | todo |
| E0b-2 | NOOP, VE scope, clean Spatial, patched | 200 | Amina | todo |

**Why E0b re-runs BF16:** episodes can only be compared one by one when they get the same
flow-matching noise. E0a ran before the seed patch and with 50 episodes per task, so its
noise differs from any 20-per-task patched run. Comparing E0a with VLAQuantBench's
published BF16 run shows the same effect: the success rates agree, but 24% of matched
episodes flip. E0b-1 is the reference for E0b's parity check only: the core runs happen
in environment B, so their BF16 references are produced there (E1a, E2).

**Steps (Amina, VM, inside tmux, from the team repo folder on the VM):**
1. Add `export PYOPENGL_PLATFORM=egl` to `~/.bashrc`. Check the clone and apply the patches
   (`docs/SETUP.md`, section 4); note the patched commit in `RUN_LOG.md`.
2. Check the NOOP preset touches the model correctly (seconds):
   ```bash
   python scripts/parity_check.py weights --model pi0 --preset NOOP --scope ve
   ```
   Expect: `ve: 162 quantizable layers`, `parameters changed: 0`, `PASS`.
3. E0b-1 (~55 min):
   ```bash
   vqb run --model pi0 --benchmark libero --suite libero_spatial --preset BASELINE --scope e2e \
       --episodes 20 --out results/E0/E0b-BF16-patched-spatial.jsonl
   ```
4. E0b-2 (~55 min):
   ```bash
   vqb run --model pi0 --benchmark libero --suite libero_spatial --preset NOOP --scope ve \
       --episodes 20 --out results/E0/E0b-NOOP-spatial.jsonl
   ```
   Check the header: `quant_report.ve.layers` is 162.
5. Compare them:
   ```bash
   python scripts/parity_check.py rollouts \
       --ref results/E0/E0b-BF16-patched-spatial.jsonl --cell results/E0/E0b-NOOP-spatial.jsonl
   ```
6. Compare E0a with E0b-1 statistically (expect PASS):
   ```bash
   python scripts/parity_check.py rollouts --mode statistical \
       --ref results/E0/E0a-BF16-spatial.jsonl --cell results/E0/E0b-BF16-patched-spatial.jsonl
   ```
7. Stop the VM.

**If step 5 shows mismatches:** GPUs are not always bit-for-bit reproducible, and over a
280-step rollout a tiny difference can flip a few outcomes. Before blaming the no-op, run
BF16 once more (same command as step 3, `--out results/E0/E0b-BF16-patched-rerun-spatial.jsonl`)
and compare the two BF16 runs in exact mode. That gives the mismatch count from chance alone.
- NOOP vs. BF16 mismatches are about as many as BF16 vs. BF16 → the no-op is fine; record
  both counts in `RUN_LOG.md`.
- NOOP mismatches clearly exceed the BF16 rerun's → the no-op changes the model. Stop and investigate.

---

## E1: final-environment checks and calibration (pilot, ~1,150 episodes)

**Goal:** confirm environment B, pick difficulty levels, measure discordance, update power.

| Part | What | Episodes |
|---|---|---|
| E1a | Clean BF16 + NOOP parity in environment B | 400 |
| E1b | BF16 difficulty calibration on lighting, background, layout, language | ~550 |
| E1c | VE-W4 and W8 on one visual shift (discordance) | 200 |

**Difficulty rule (proposal 6.2):** per dimension, choose the lowest tested level where
BF16 succeeds 40–85%. If none, take the closest and flag it. If BF16 is below 20% at every
level of a visual dimension, report it separately and use a reduced-set pooled QRP. Never
use `null` levels (implementation note 6). Use only BF16 results to choose.

**Power update (proposal 7.4):** recompute the expected QRP standard error from E1's task
variation and discordance. If it is above ~2.5 points, increase shifted samples (up to 450
per shift) before the freeze.

**Output:** `configs/frozen_design.yaml` filled in, committed, tagged `design-freeze`.

---

## E2: clean grid (Amina, 2,400 episodes)

BF16, VE-W4, whole-model W4 and W8 on clean Spatial, Object and Goal: 10 tasks × 20
episodes × 3 suites = 600 episodes per setting. Files: `results/E2/clean/E2-<setting>-<suite>.jsonl`.

## E3: visual-shift grid (Smitha, 3,000 episodes)

BF16, VE-W4 and whole-model W4 on lighting, background and layout, ~300 matched episodes per
shift (task proportions as in the clean grid). W8 runs a 100-episode subset of the same
variants per shift. Files: `results/E3/<shift>/E3-<setting>-<suite>.jsonl`.

## E4: language contrast (Amina, 900 episodes)

BF16, VE-W4 and whole-model W4 on 300 language rewrites. π0 gets the rewritten instruction.
Files: `results/E4/language/E4-<setting>-<suite>.jsonl`.

**Before core runs:** audit a sample of the prompts actually passed (implementation note 4).

## E5: blinded failure-mode coding (required)

Per visual shift and language: up to 30 matched pairs where BF16 succeeds and VE-W4 fails,
plus pairs with the opposite result. Videos get random IDs. Code with
`docs/FAILURE_CODEBOOK.md`; both code a shared subset of 30 and report Cohen's κ.

## E6: same-observation action replay (optional)

Replay BF16 observation sequences through BF16 and VE-W4 with the same noise, starting from
VLAQuantBench's `scripts/action_fidelity.py` / `replay_pi05_subsets.py` (proposal Eq. 4).

## E7: AH-W4 and/or VE-W2 (optional, ≤3,000 episodes)

Only with budget left after E2–E4 and reruns.

---

## Analysis

```bash
python scripts/qrp_bootstrap.py configs/qrp/study_manifest.yaml --base-dir . \
    --out-json analysis/qrp.json --out-md analysis/qrp.md
```
Reports success rates with Wilson intervals, matched gaps with discordance counts, per-shift
and pooled QRP with the paired task-clustered bootstrap (10,000 replicates), the H1 decision,
and the Holm-corrected secondary tests. The bootstrap resampling scheme
(`bootstrap_within`) is decided at the design freeze.
