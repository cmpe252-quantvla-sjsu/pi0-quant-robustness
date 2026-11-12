# Run log

One entry per run, newest at the bottom. Log every run, including failed, stopped and pilot
runs, plus any change to the machine or environment. If a result surprises you, write it
down here before trying to explain it.

**Entry template**

```
### <ID> · <short title>
- Date / who: 
- Machine / env / commit: qvla-l4 · vqb_a|vqb_b · VLAQuantBench 4a2cb7c + patches <…> · repo <commit>
- Command:
- Output file:
- Result:
- Checks:
- Deviations / notes:
```

---

## Machine and environment changes

### 2026-10-29 · VM and environment A (Smitha)
- Created `qvla-l4` (L4, 23 GB) in `us-central1-c`. Shared conda in `/opt/miniforge3`, work
  in `/opt/qvla` (group `qvla`). Rendering needs `export PYOPENGL_PLATFORM=egl`.
- Built `vqb_a` from VLAQuantBench's LeRobot setup script. Deviations:
  - LeRobot **0.5.1** instead of the pinned 0.5.2, which does not exist (torch 2.10.0, same
    as VLAQuantBench's records). README updated to 0.5.1 (Oct 31).
  - Installed cmake and build tools for `egl_probe`.
  - MuJoCo re-pinned to **3.3.2** (the setup had installed 3.8.1).
  - Installed the NVIDIA graphics package for EGL rendering (the image only had compute).
  - Added 32 GB swap (loading π0 ran out of RAM).
  - Logged in to Hugging Face for the PaliGemma tokenizer (shared token).
- `vqb inspect` matches VLAQuantBench's published layer counts: VE 162 layers / 411.1M params.
- Lost a morning to L4 stockouts in the zone. If the VM won't start, wait an hour and retry.
- Machine type `g2-standard-8`; image `common-cu129-ubuntu-2204-nvidia-580` (Google Deep
  Learning VM, Ubuntu 22.04); boot disk 150 GB balanced persistent disk (`df` shows 146 GB).
- Exact packages installed (apt):
  - `tmux git libegl1 libgl1 libglib2.0-0 ffmpeg cmake build-essential libegl1-mesa-dev`
  - `libnvidia-gl-580-server=580.178.04-0ubuntu0.22.04.1` (the image ships the compute-only
    `580-server` driver; this adds EGL rendering, matched to driver 580.178.04)
- Hugging Face read token `qvla-l4-read` (Smitha's account, PaliGemma license accepted).
- Swap: 32 GB `/swapfile`, in `/etc/fstab` (π0 load was OOM-killed at 31.5 GB RSS on 32 GB RAM).

### 2026-10-30 · Amina's login and VLAQuantBench patches (Amina)
- Added to `~/.bashrc` (same values as Smitha's, confirmed by her): `umask 002`,
  `HF_HOME=/opt/qvla/hf_cache`, `LIBERO_CONFIG_PATH=/opt/qvla/libero_config`,
  `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`.
- `git config --global --add safe.directory /opt/qvla/third_party/VLAQuantBench` (the clone is
  owned by Smitha's login; git refuses another user's repo without this).
- **Hugging Face token was owner-only** (`-rw-------`), so other logins couldn't load π0
  (`PermissionError` on `/opt/qvla/hf_cache/token`). Fixed with
  `sudo chmod g+r /opt/qvla/hf_cache/token /opt/qvla/hf_cache/stored_tokens` (now `-rw-r-----`:
  group read-only). Running `hf auth login` again resets it to owner-only; redo the chmod then.
- Clone was `4a2cb7c` (detached), only `scripts/env/setup_lerobot.sh` modified. Created branch
  `qvla-patches` and applied patches 0001–0003 with `git am` → **`qvla-patches` @ `c497a0d`**.
  This is the VLAQuantBench version from E0b on. `vqb presets` lists `NOOP`.
- The team repo is not cloned on the VM. Patches and scripts were uploaded to `~` with the
  SSH-in-browser UPLOAD FILE button; results downloaded with DOWNLOAD FILE.
- L4 stockout again 11:50–~15:00; the VM started on a later try.

### 2026-11-04 · `LIBERO_CONFIG_PATH` migration and environment B (Smitha)
- `LIBERO_CONFIG_PATH` is now a conda env variable instead of a `~/.bashrc` line:
  `vqb_a` → `/opt/qvla/libero_config_a` (copy of the old `/opt/qvla/libero_config`, kept as a
  backup), `vqb_b` → `/opt/qvla/libero_config_b`. Removed the line from Smitha's `~/.bashrc`.
  Amina's `~/.bashrc` line is harmless (conda overrides it on activation) but can be removed
  (removed Nov 6, when Amina also added the conda `source` line to her `~/.bashrc`).
  Checked: `vqb_a` still finds the original LIBERO.
- Built `vqb_b` with `conda create --name vqb_b --clone vqb_a`, then:
  - `pip uninstall hf_libero`.
  - LIBERO-Plus cloned to `/opt/qvla/third_party/LIBERO-plus` @ `4976dc3`, installed with
    `pip install -e . --no-deps --config-settings editable_mode=compat`. The normal editable
    install registers nothing, because LIBERO-Plus's top-level `libero/` has no `__init__.py`.
  - apt: `libexpat1 libfontconfig1-dev libmagickwand-dev unzip`.
  - pip: `wand 0.7.2`, `scikit-image 0.26.0`, `usd-core 26.11`, `gym 0.25.2` + `gym-notices 0.1.0`
    (`--no-deps`; LIBERO-Plus imports gym, `hf_libero` did not). `pip freeze` before/after:
    only these were added. numpy 2.2.6, torch 2.10.0+cu128, LeRobot 0.5.1, MuJoCo 3.3.2 unchanged.
    Gym prints an "unmaintained / NumPy 2" warning; LIBERO-Plus only uses it for type names.
  - Assets: `hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset` (6.4 GB). The zip
    keeps the authors' long folder path; moved its `assets/` folder to
    `LIBERO-plus/libero/libero/assets` (9.5 GB, 457k files). Zip deleted afterwards.
- Team repo now cloned on the VM: `/opt/qvla/pi0-quant-robustness` (fine-grained GitHub token,
  this repo only). Test runs used a local branch `vm-test` = `smitha/runner` (`4078f44`) merged
  with `smitha/sampler` (not pushed). `pytest`: 72 passed in `vqb_b`.
- For `--benchmark libero_plus`, VQB can't find the checkpoint automatically: pass
  `--checkpoint lerobot/pi0_libero_finetuned_v044`.
- Disk after setup: 95 GB used, 51 GB free.

### 2026-11-05/06 · robosuite log file and env B lock (Smitha)
- robosuite writes `/tmp/robosuite.log`. Amina's morning runs created it under her login, so
  Smitha's runs failed (`PermissionError: /tmp/robosuite.log`). Deleted it (`sudo rm`) and
  turned off robosuite's file logging in `vqb_b`: ran `robosuite/scripts/setup_macros.py`,
  then set `FILE_LOGGING_LEVEL = None` in `macros_private.py`. Rollouts are unaffected.
  `pip freeze` does not capture this, so it is noted in the lock file. `vqb_a` unchanged.
- Nov 6: `configs/env_b.lock.txt` written (pip freeze, NVIDIA L4 / driver 580.178.04,
  VLAQuantBench `c497a0d`, LIBERO-Plus `4976dc3`, host qvla-l4). Branch `smitha/env-b-lock`.

---

## Runs

### E0a · BF16 on clean Spatial, environment A
- Date / who: 2026-10-29, started 23:03 UTC (16:03 PT) · Smitha
- Machine / env / commit: qvla-l4 (NVIDIA L4) · vqb_a · Python 3.12.16 · VLAQuantBench `4a2cb7c`, **unpatched**
- Command: `vqb run --model pi0 --benchmark libero --suite libero_spatial --preset BASELINE --video-dir /opt/qvla/videos/E0a --out /opt/qvla/runs/E0/E0a-BF16-spatial.jsonl`
  (no `--episodes`, so it ran the default 50 per task; run inside tmux after one dropped-connection restart)
- Output file: `results/E0/E0a-BF16-spatial.jsonl`
- Result: **76.0% (152/200)** on the first 20 episodes per task, inside the 71.8–83.2% target
  (published: 78.0%). 76.6% (383/500) on all 500.
  Per task (first 20): 15, 16, 18, 18, 16, 12, 17, 15, 12, 13.
- Checks: header `episodes_per_task: 50`, `quant_report` 0 layers everywhere, 280 steps, 360 px.
  Mean wall time 16.5 s per episode (median 13.2 s), peak VRAM 8.4 GB.
- Statistical comparison with VLAQuantBench's published BF16 file
  (`parity_check.py rollouts --mode statistical`): **PASS** with the fixed script (Oct 30). The
  first version printed FAIL here, only because of the expected 50-vs-20 `episodes_per_task`
  difference, and printed no verdict line; "PASS" was first written from the numbers by hand.
  Gap +2.0 points, task-clustered
  95% CI [−3.5, +7.5], McNemar p = 0.665. But only 152/200 matched episodes have the same
  outcome (26 vs. 22 discordant). Different machine and different noise per episode, so
  episodes can't be compared one by one across runs like these.
- Deviations / notes: ran 500 episodes because `--episodes` was omitted. `--out` needs a
  file name, not a folder.

### E0b · NOOP parity check, environment A: **PASS**
- Date / who: 2026-10-30, ~15:40–18:55 PT · Amina
- Machine / env / commit: qvla-l4 (NVIDIA L4) · vqb_a · VLAQuantBench `qvla-patches` @ `c497a0d`
- VM output folder: `/opt/qvla/runs/E0b/` (separate from E0, so `vqb summarize` can't merge
  these BASELINE runs with E0a). Repo copies in `results/E0/`.

**Weight check**
- Command: `python ~/parity_check.py weights --model pi0 --preset NOOP --scope ve`
- Result: **PASS**. `ve: 162 quantizable layers`, `162 layers touched`, `parameters changed: 0 of 778`.
- Note: all 162 VE layers are **float32** in this checkpoint (`weight dtypes {'float32': 162}`),
  so VE-W4 rounded weights are stored back in float32, not BF16. Fix the wording in the report
  (proposal 4.2 says the rounded weights "are stored as BF16").

**Runs** (clean Spatial, `--episodes 20`, 200 episodes each)

| Run | Command (all `vqb run --model pi0 --benchmark libero --suite libero_spatial --episodes 20`) | File | Success |
|---|---|---|---|
| E0b-1 | `--preset BASELINE --scope e2e` | `E0b-BF16-patched-spatial.jsonl` | 75.5% (151/200), Wilson [69.1, 80.9] |
| E0b-2 | `--preset NOOP --scope ve` (header: `ve` 162 layers) | `E0b-NOOP-spatial.jsonl` | 75.5% (151/200) |
| E0b-3 | `--preset BASELINE --scope e2e` (chance-level rerun) | `E0b-BF16-patched-rerun-spatial.jsonl` | 76.0% (152/200) |

Each took ~55 min; inference latency ~0.44 s per chunk, peak VRAM 8.4 GB.

**Comparisons** (`parity_check.py rollouts`)

| Comparison | Same success and steps | Same success | Flips (n10 / n01) | Paired gap, 95% CI |
|---|---|---|---|---|
| E0b-1 BF16 vs E0b-2 NOOP (exact) | 191/200 | 198/200 | 1 / 1 | +0.0 [−1.5, +1.5] |
| E0b-1 BF16 vs E0b-3 BF16 rerun (exact) | 192/200 | 199/200 | 0 / 1 | −0.5 [−1.5, +0.0] |
| E0a vs E0b-1 (statistical) | 17/200 | 133/200 | 34 / 33 | +0.5 [−6.0, +6.5], McNemar p = 1.0; PASS with the fixed script |

- **Conclusion: the NOOP preset does not change the model.** The weights are bit-identical, and
  NOOP vs. BF16 differs no more than BF16 vs. itself. The exact check "fails" only because the
  GPU is not bit-for-bit repeatable: with identical model and seeds, ~4% of episodes change
  their step count and ~0.5–1% flip outcome. Several of the same episodes wobble in both
  comparisons (task 0 ep 11, task 2 ep 1, task 4 ep 15, task 6 ep 12).
- **Consequence:** exact episode-by-episode parity is not achievable on this machine; parity
  checks (E1a) use the BF16-rerun mismatch count as the chance level, and analysis stays
  statistical. Outcome flips from GPU noise alone (~1%) are a floor for discordance in the
  E1 power update.
- **The seed patch matters:** with different noise (E0a, unpatched, vs. E0b-1), 67/200 outcomes
  flip although the success rates agree (76.0% vs. 75.5%); with matched noise only 1–2 flip.
- Deviation: the statistical E0a comparison printed no verdict line (it treated the expected
  50-vs-20 `episodes_per_task` difference as a failure). Fixed in `scripts/parity_check.py`.

### Smoke-B · LIBERO-Plus runner smoke test, environment B
- Date / who: 2026-11-04 · Smitha
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench `qvla-patches` @ `c497a0d` · repo `vm-test`
- Command: `python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044
  --benchmark libero_plus --suite libero_spatial --preset BASELINE --tasks 0
  --bench-kwargs condition=<C> sample_file=/opt/qvla/runs/smoke_b/samples/<C>.json`
  (layout also task 6). Sample files: `sample_variants.py sample --suites libero_spatial
  --episodes-per-task 2`, levels lighting 3, background 2, layout 3, language 3.
- Output file: `/opt/qvla/runs/smoke_b/<C>.jsonl` (+ videos), not in the repo.
- Result: lighting 2/2, background 2/2, layout 4/4, language 0/2 (both timed out at 280 steps).
- Checks: no duplicate episodes. Every record has condition, variant_id, plus_id,
  difficulty_level, target_level, fallback, init_state_id and instruction as the sample file
  says. Layout `_add_` / `_level` variants use init state 0; task 6 recorded as fallback (L4,
  target 3). Language episodes got the rewritten instructions; the other conditions the standard one.
- Deviations / notes: a dropped connection killed one attempt; runs since then are in tmux.
  The language failures are only 2 episodes. Watch them in E1.

### E1a · BF16 on clean Spatial, environment B: parity **PASS**
- Date / who: 2026-11-04 · Smitha
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench `qvla-patches` @ `c497a0d` · repo `vm-test`
- Command: `python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044
  --benchmark libero_plus --suite libero_spatial --preset BASELINE --episodes 20
  --out /opt/qvla/runs/E1a/clean-BF16-spatial.jsonl`
- Output file: `/opt/qvla/runs/E1a/clean-BF16-spatial.jsonl`
- Result: **76.0% (152/200)**, Wilson 95% [69.6, 81.4]. Peak VRAM 8.4 GB, inference ~0.45 s per chunk.
- Checks: `parity_check.py rollouts --ref E0b-BF16-patched-spatial.jsonl` (env A, 75.5%):
  - statistical: **PASS**. Paired gap −0.5 pp, task-clustered 95% CI [−2.0, +1.0], McNemar p = 1.000.
  - exact: 183/200 same outcome, 120/200 same outcome and steps; flips 8 / 9. Many episodes
    are off by 1–2 steps: LIBERO-Plus's simulation code is slightly different from LIBERO's.
- BF16 rerun in env B (chance level): `clean-BF16-rerun-spatial.jsonl` → **76.0% (152/200)**.
  Exact compare with the first run: 200/200 same outcome, 196/200 same outcome and steps,
  flips 0 / 0. Inside env B, GPU noise moved 4 episodes by 1–2 steps and changed no outcome.

### E1b-language · BF16 language calibration, level 1, environment B (Amina)
- Date / who: 2026-11-05, 09:29–10:07 PT · Amina
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench 4a2cb7c + patches (qvla-patches @ c497a0d) · repo d08b8e9 (main)
- Sample file (sampler seed 1, 2 per task):
  `python scripts/sample_variants.py sample --liberoplus /opt/qvla/third_party/LIBERO-plus --vqb /opt/qvla/third_party/VLAQuantBench --condition language --level 1 --episodes-per-task 2 --seed 1 --out /opt/qvla/runs/E1b/language/samples/L1-s1.json`
  → 60 episodes. Goal tasks 0 and 6 have no level-1 rewrite and fall back to level 2 (2 tasks, at the cap). Same summary as the laptop run the night before. Committed as `configs/samples/E1/language-L1-s1.json`.
- Command (run seed 1):
  `for S in libero_spatial libero_object libero_goal; do python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 --benchmark libero_plus --suite $S --preset BASELINE --seed 1 --bench-kwargs condition=language sample_file=/opt/qvla/runs/E1b/language/samples/L1-s1.json --out /opt/qvla/runs/E1b/language/BF16-L1-$S.jsonl; done`
- Output files: `results/E1/language/E1-BF16-{spatial,object,goal}-L1.jsonl` (copied from `/opt/qvla/runs/E1b/language/`)
- Result: Spatial 1/20, Object 2/20, Goal 7/20 → **10/60 = 16.7% pooled**.
- Checks: `extra.instruction` holds the BDDL rewrite in all 60 episodes; none equals the standard instruction.
- Notes: below 40% at the lowest level, so language fails the difficulty rule (proposal 6.2): levels 2–5 not run, and **H3 is omitted** as pre-registered. Many level-1 rewrites replace object names with descriptions ("the solid dairy product", "the woven holder") or add chatter; π0 fails even on near-standard wording ("grab the ketchup and set it in the basket").

### E1b-control · Clean BF16 on Object and Goal, environment B (Amina)
- Date / who: 2026-11-05, 10:16–10:43 PT · Amina
- Machine / env / commit: as above
- Commands:
  `python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 --benchmark libero_plus --suite libero_object --preset BASELINE --seed 1 --episodes 2 --bench-kwargs condition=clean --out /opt/qvla/runs/E1b/language/control-BF16-clean-object-s1.jsonl`
  (and the same with `libero_goal` → `control-BF16-clean-goal-s1.jsonl`)
- Output files: `results/E1/clean/E1-BF16-{object,goal}-control.jsonl`
- Result: **Object 17/20, Goal 19/20** (VLAQuantBench's released BF16 with the same initial states, episodes 0–1, but different noise: 18/20 and 18/20).
- Notes: same episodes, initial states and noise as the language run; only the instruction differs. So (1) the language drop is real, not an environment problem, and (2) environment B is fine on Object and Goal (E1a covered Spatial).
- My runs left `/tmp/robosuite.log` owned by my user, which blocked Smitha's later runs; she turned robosuite file logging off in vqb_b (see her entry).

### E1b-visual · BF16 difficulty calibration, visual shifts, environment B
- Date / who: 2026-11-05 (~16:50–18:20 PT) · Smitha
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench `qvla-patches` @ `c497a0d` · repo main @ `d08b8e9`
- Command (in `/opt/qvla/pi0-quant-robustness`, `vqb_b`):
  ```bash
  SV="python scripts/sample_variants.py sample --liberoplus /opt/qvla/third_party/LIBERO-plus --vqb /opt/qvla/third_party/VLAQuantBench --episodes-per-task 2 --seed 1"
  $SV --condition lighting   --level 3 --out $D/samples/lighting-L3-s1.json
  $SV --condition layout     --level 3 --out $D/samples/layout-L3-s1.json
  $SV --condition background --level 1 --out $D/samples/background-L1-s1.json
  for C in lighting-L3 layout-L3 background-L1; do for S in libero_spatial libero_object libero_goal; do
    python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 \
      --benchmark libero_plus --suite $S --preset BASELINE --seed 1 \
      --bench-kwargs condition=${C%-*} sample_file=$D/samples/$C-s1.json --out $D/BF16-$C-$S.jsonl
  done; done
  ```
  with `D=/opt/qvla/runs/E1b/visual`. Background top-up: same, with `--episodes-per-task 4`
  (`background-L1-s1-4pt.json`), `--episodes-from 2`, output `BF16-background-L1-topup-$S.jsonl`.
- Output file: `/opt/qvla/runs/E1b/visual/`; repo copy `results/E1/E1b/visual/` (+ `samples/`)
- Result (BF16, pooled over 3 suites, 60 episodes each):

  | Shift | Spatial | Object | Goal | Pooled |
  |---|---|---|---|---|
  | Lighting L3 | 13/20 | 14/20 | 18/20 | **75.0%** |
  | Layout L3 | 14/20 | 14/20 | 10/20 | **63.3%** |
  | Background L1 | 15/20 | 16/20 | 18/20 | **81.7%** |

  Background L1 was near 85%, so topped up with episodes 2–3 (4/task file, `--episodes-from 2`):
  **80.8% (97/120)**, still in range.
- Checks: at most 2 fallback tasks per suite in every sample file (sampler output).
- Deviations / notes: first attempt failed on `/tmp/robosuite.log` (see machine changes).

### E1c · VE-W4 and W8 on lighting L3, environment B
- Date / who: 2026-11-05 overnight · Smitha
- Machine / env / commit: as E1b-visual
- Command (`E=/opt/qvla/runs/E1c`):
  ```bash
  python scripts/sample_variants.py sample --liberoplus /opt/qvla/third_party/LIBERO-plus --vqb /opt/qvla/third_party/VLAQuantBench --seed 1 --episodes-per-task 4 --condition lighting --level 3 --out $E/lighting-L3-s1-4pt.json
  RUN="python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 --benchmark libero_plus --seed 1"
  LB="condition=lighting sample_file=$E/lighting-L3-s1-4pt.json"
  for S in libero_spatial libero_object libero_goal; do
    $RUN --suite $S --preset BASELINE --episodes-from 2 --bench-kwargs $LB --out $E/BF16-ep23-$S.jsonl
    $RUN --suite $S --preset W4 --scope ve  --bench-kwargs $LB --out $E/VEW4-$S.jsonl
    $RUN --suite $S --preset W8 --scope e2e --bench-kwargs $LB --out $E/W8-$S.jsonl
  done
  ```
  BF16 episodes 0–1 are E1b's lighting files (same sampler seed, so the same variants).
- Output file: `/opt/qvla/runs/E1c/`; repo copy `results/E1/E1c/`
- Result: BF16 **75.8% (91/120)**. VE-W4 **75.0% (90/120)**: discordant 4 / 3 (5.8%).
  W8 **72.5% (87/120)**: discordant 7 / 3 (8.3%).
- Notes: GPU noise alone flips 0/200 in env B (E1a rerun), so VE-W4 does change individual
  outcomes, but they nearly cancel. Pilot only, never in the core manifest. Input to the power update.

### E1-power · Power update (Amina)
- Date / who: 2026-11-06 · Amina (laptop, no GPU)
- Inputs: E1b BF16 rates per shift (Smitha's visual calibration incl. the background top-up; lighting also uses the E1c BF16 episodes 2–3, so 120 episodes), E1c lighting VE-W4 vs BF16 (120 matched pairs: n10 = 4, n01 = 3, discordance 5.8%), VLAQuantBench released clean BF16 per-task rates.
- Command: `python scripts/power_update.py --e1 results/E1 --vqb ../VLAQuantBench --sims 300 --B 1000 --out-md analysis/E1_power_update.md`
- Result: projected SE of the pooled VE-W4 QRP **≈ 1.2 pp at 300 per shift**, 1.1 pp at 450; ~1.7 pp if discordance doubles. Rule threshold 2.5 pp → **keep 300 per shift**. Details: `analysis/E1_power_update.md`.
- Notes: clean VE-W4 discordance in env B is assumed equal to E1c; the released clean Spatial logs give 6.5%, close to it.

### E2 · Clean grid: BF16, VE-W4, W4, W8 × Spatial/Object/Goal, environment B (Amina)
- Date / who: 2026-11-08 22:01 PT – 2026-11-09 09:02 PT · Amina
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench `qvla-patches` @ `c497a0d` · repo
  `design-freeze-1-g4198e04` (code and configs identical to `design-freeze`:
  `git diff --stat design-freeze -- quantvla scripts configs` empty)
- Pre-flight: `nvidia-smi` no other process (GPU free after Smitha's lighting + background),
  MuJoCo 3.3.2.
- Command (in `/opt/qvla/pi0-quant-robustness`, inside tmux):
  ```bash
  R=/opt/qvla/runs/E2/clean; V=/opt/qvla/videos/E2/clean; mkdir -p $R
  for SET in "BF16 BASELINE e2e" "VE-W4 W4 ve" "W4 W4 e2e" "W8 W8 e2e"; do set -- $SET
    for S in spatial object goal; do
      python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 \
        --benchmark libero_plus --suite libero_$S --preset $2 --scope $3 --seed 0 --episodes 20 \
        --bench-kwargs condition=clean --video-dir $V/$1 --out $R/E2-$1-$S.jsonl
    done
  done 2>&1 | tee -a $R/E2-log.txt
  ```
- Output files: `results/E2/clean/E2-<setting>-<suite>.jsonl` (12 files, copied from
  `/opt/qvla/runs/E2/clean/`). Log `E2-log.txt` and videos (`/opt/qvla/videos/E2/clean/<setting>/`)
  stay on the VM.
- Result (success, 200 episodes per suite):

  | Setting | Spatial | Object | Goal | Pooled | vs BF16: n10 / n01 (discordance) |
  |---|---|---|---|---|---|
  | BF16 | 75.5% | 90.5% | 82.5% | **82.8%** (497/600) | |
  | VE-W4 | 75.5% | 91.0% | 82.5% | **83.0%** (498/600) | 11 / 12 (3.8%) |
  | W8 | 72.5% | 91.5% | 83.0% | **82.3%** (494/600) | 20 / 17 (6.2%) |
  | W4 | 69.0% | 91.0% | 81.5% | **80.5%** (483/600) | 59 / 45 (17.3%) |

- Checks: every file has 200 episodes, 20 per task, no duplicate (task, episode); headers
  identical in seed 0, `libero_plus`, `qvla-l4`, NVIDIA L4, `c497a0d`, `episodes_per_task` 20;
  all episodes `condition: clean`. `quant_report`: BF16 0 layers; VE-W4 `ve` 162; W4 and W8
  `ve` 162, `llm` 126, `ah` 131, `mp` 1.
- Notes: BF16 Spatial 75.5% matches E0b (75.5%) and E1a (76.0%). VE-W4 gives the same outcome
  and step count as BF16 in 393/600 episodes. Clean VE-W4 discordance 3.8%, below the 5.8% the
  power update assumed, so its precision estimate was conservative. The `imageio_ffmpeg`
  "resizing from (360, 360) to (368, 368)" warning concerns the saved videos only (padding for
  the encoder), not the policy input.

### E3-layout · Layout L3: BF16, VE-W4, W4 (300) + W8 subset (120), environment B (Amina)
- Date / who: 2026-11-09 11:28–18:34 PT · Amina
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench `qvla-patches` @ `c497a0d` · repo
  `design-freeze-1-g4198e04` (code and configs identical to `design-freeze`, checked before E2)
- Sample files: `configs/samples/core/layout-L3-s0.json` (10/task) and `layout-L3-s0-w8.json`
  (4/task = episodes 0–3 of the same sample), sampler seed 0, frozen with the design.
- Command (in `/opt/qvla/pi0-quant-robustness`, inside tmux, started right after E2):
  ```bash
  R=/opt/qvla/runs/E3/layout; V=/opt/qvla/videos/E3/layout; SF=configs/samples/core; mkdir -p $R
  for SET in "BF16 BASELINE e2e layout-L3-s0" "VE-W4 W4 ve layout-L3-s0" "W4 W4 e2e layout-L3-s0" "W8 W8 e2e layout-L3-s0-w8"; do set -- $SET
    for S in spatial object goal; do
      python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 \
        --benchmark libero_plus --suite libero_$S --preset $2 --scope $3 --seed 0 \
        --bench-kwargs condition=layout sample_file=$SF/$4.json --video-dir $V/$1 --out $R/E3-$1-$S.jsonl
    done
  done 2>&1 | tee -a $R/E3-layout-log.txt
  ```
- Output files: `results/E3/layout/E3-<setting>-<suite>.jsonl` (12 files, copied from
  `/opt/qvla/runs/E3/layout/`). Log `E3-layout-log.txt` and videos
  (`/opt/qvla/videos/E3/layout/<setting>/`) stay on the VM.
- Result (success; BF16, VE-W4, W4: 100 episodes per suite; W8: 40 per suite):

  | Setting | Spatial | Object | Goal | Pooled | vs BF16 on the same episodes: n10 / n01 (discordance) |
  |---|---|---|---|---|---|
  | BF16 | 55.0% | 74.0% | 55.0% | **61.3%** (184/300) | |
  | VE-W4 | 52.0% | 74.0% | 53.0% | **59.7%** (179/300) | 6 / 1 (2.3%) |
  | W4 | 39.0% | 74.0% | 51.0% | **54.7%** (164/300) | 33 / 13 (15.3%) |
  | W8 (subset) | 55.0% | 80.0% | 47.5% | **60.8%** (73/120; BF16 on the same 120: 73) | 2 / 2 (3.3%) |

- Checks: every file has the expected episodes (100 or 40), 10 or 4 per task, no duplicate
  (task, variant, episode), every episode matched to a BF16 episode; all episodes
  `condition: layout`, `target_level` 3; fallback episodes only in Spatial task 6 and Object
  task 4, as in the sampler output (cap 2 per suite). Headers identical in seed 0,
  `libero_plus`, `qvla-l4`, NVIDIA L4, `c497a0d`; `episodes_per_task` 10 (W8: 4);
  `quant_report` as in E2. No errors in the log.
- Notes: BF16 61.3% is close to the E1 pilot (63.3%) and inside the 40–85% target. These are
  raw rates; the QRP and its interval come from `qrp_bootstrap.py` on the full grid.

### E3-lighting-background · Lighting L3 and background L1: BF16, VE-W4, W4 (300) + W8 subset (120), environment B (Smitha)
- Date / who: 2026-11-08, ~09:20–20:30 PT · Smitha
- Machine / env / commit: qvla-l4 · vqb_b · VLAQuantBench `qvla-patches` @ `c497a0d` · repo main @ `05efb22` (tag `design-freeze`)
- Sample files: `configs/samples/core/lighting-L3-s0.json`, `background-L1-s0.json` (10/task) and
  their `-w8` files (4/task = episodes 0–3 of the same sample), sampler seed 0, frozen with the design.
- Command (in `/opt/qvla/pi0-quant-robustness`, inside tmux):
  ```bash
  RUN="python -m quantvla.cli run --model pi0 --checkpoint lerobot/pi0_libero_finetuned_v044 --benchmark libero_plus --seed 0"
  for C in lighting-L3 background-L1; do
    COND=${C%-*}; OUT=results/E3/$COND; mkdir -p $OUT
    for S in libero_spatial libero_object libero_goal; do
      for SET in "BF16 --preset BASELINE --scope e2e" "VE-W4 --preset W4 --scope ve" "W4 --preset W4 --scope e2e"; do
        set -- $SET; NAME=$1; shift
        $RUN --suite $S "$@" --bench-kwargs condition=$COND sample_file=configs/samples/core/$C-s0.json \
          --video-dir /opt/qvla/videos/E3/$COND/$NAME --out $OUT/E3-$NAME-$S.jsonl
      done
      $RUN --suite $S --preset W8 --scope e2e --bench-kwargs condition=$COND sample_file=configs/samples/core/$C-s0-w8.json \
        --video-dir /opt/qvla/videos/E3/$COND/W8 --out $OUT/E3-W8-$S.jsonl
    done
  done
  ```
- Output files: `results/E3/lighting/` and `results/E3/background/`, `E3-<setting>-<suite>.jsonl`
  (24 files). Videos: `/opt/qvla/videos/E3/<condition>/<setting>/` (2,040, 184 MB, kept for E5).
- Result (success; BF16, VE-W4, W4: 100 episodes per suite; W8: 40 per suite):

  **Lighting L3**

  | Setting | Spatial | Object | Goal | Pooled | vs BF16 on the same episodes: n10 / n01 (discordance) |
  |---|---|---|---|---|---|
  | BF16 | 66.0% | 83.0% | 68.0% | **72.3%** (217/300) | |
  | VE-W4 | 65.0% | 81.0% | 68.0% | **71.3%** (214/300) | 6 / 3 (3.0%) |
  | W4 | 55.0% | 74.0% | 69.0% | **66.0%** (198/300) | 39 / 20 (19.7%) |
  | W8 (subset) | 62.5% | 80.0% | 67.5% | **70.0%** (84/120; BF16 on the same 120: 84) | 3 / 3 (5.0%) |

  **Background L1**

  | Setting | Spatial | Object | Goal | Pooled | vs BF16 on the same episodes: n10 / n01 (discordance) |
  |---|---|---|---|---|---|
  | BF16 | 71.0% | 88.0% | 79.0% | **79.3%** (238/300) | |
  | VE-W4 | 72.0% | 89.0% | 78.0% | **79.7%** (239/300) | 8 / 9 (5.7%) |
  | W4 | 65.0% | 89.0% | 79.0% | **77.7%** (233/300) | 28 / 23 (17.0%) |
  | W8 (subset) | 65.0% | 90.0% | 72.5% | **75.8%** (91/120; BF16 on the same 120: 92) | 3 / 2 (4.2%) |

- Checks: every BF16 / VE-W4 / W4 file has 100 episodes and every W8 file 40; every episode
  matched to a BF16 episode; headers identical in seed 0, `libero_plus`, `qvla-l4`, NVIDIA L4,
  `c497a0d`; one video per episode (300 / 120 per folder, none overwritten); no errors.
  BF16 close to the E1 pilot (lighting 75.8%, background 80.8%).
- Deviations / notes: the runs wrote `E3-<setting>-libero_<suite>.jsonl`; renamed to
  `E3-<setting>-<suite>.jsonl` (results/README naming) before merging, contents unchanged.
  Raw rates only; the QRP and its interval come from `qrp_bootstrap.py` on the full grid.

---

## Decisions

### 2026-10-30 · After E0b, before any core data (Amina and Smitha, PR review)
- **Keep the episode-seed patch (0002) for all core runs.** E0b shows why: with unmatched
  noise 67/200 outcomes flip between two BF16 runs, with matched noise 1–2.
- **Bootstrap: tasks-only (`bootstrap_within: fixed`) is primary**, the two-stage scheme of
  proposal §7.1 is reported as sensitivity. Based on the coverage simulation (99% vs. 92–94%
  coverage for a nominal 95% interval). Recorded in `configs/frozen_design.yaml`.
- **`LIBERO_CONFIG_PATH` per conda env** (`/opt/qvla/libero_config_a` for `vqb_a`, `_b` for
  `vqb_b`) instead of `~/.bashrc`; migration steps in `docs/SETUP.md` §1, still to do on the VM.
- `qrp_bootstrap.py` now refuses cells that differ in hostname, GPU, VLAQuantBench commit,
  seed or episodes per task (one-machine rule at analysis time).

### 2026-11-04 · After E1a (Smitha; confirmed by Amina)
- **Never pair env A and env B episodes in one comparison.** Across environments 17/200
  outcomes flip, against 1–2 within env A. All pilot and core runs in env B, as planned.
- The per-env `LIBERO_CONFIG_PATH` from Oct 30 is now done on the VM.

### 2026-11-06 · E1 difficulty levels (Smitha and Amina)
- Lighting L3, layout L3, background L1: one level for all suites (difficulty_rule step 1).
- E1 runs used sampler seed 1 and run seed 1; core runs use 0 for both.

### 2026-11-08 · Design freeze (Amina and Smitha)
- Freeze PR #10 (Amina, branch `amina/design-freeze`), reviewed and merged by Smitha; she
  regenerated the lighting core samples on the VM (byte-identical) and ran the tests.
- Tag **`design-freeze`** on merge commit `05efb22` (PR merged 08:25 PT; tag created ~08:48 PT
  by Smitha). No core run started before it; the first one (lighting BF16) started ~09:20 PT.
- Frozen in `configs/frozen_design.yaml`: levels lighting L3, layout L3, background L1 (one level
  for all suites); 300 per shift (10/task), W8 120 (4/task, episodes 0–3 of the same sample),
  clean 20/task; sampler seed 0 and run seed 0; language dropped (H3 omitted, E4 not run,
  Holm m = 4); core split Amina E2 + layout, Smitha lighting + background; env lock
  `configs/env_b.lock.txt`.
- Core sample files in `configs/samples/core/` (sampler seed 0). Every `-w8` file equals
  episodes 0–3 of its 10/task file (checked for all three shifts).
- Core runs also save videos with `--video-dir` (`/opt/qvla/videos/E2/clean/<setting>/`,
  `/opt/qvla/videos/E3/<condition>/<setting>/`), needed for E5.
- From here on, any change to the frozen design is logged here as a deviation.