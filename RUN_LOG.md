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
    as VLAQuantBench's records). The README's resources table still says 0.5.2.
  - Installed cmake and build tools for `egl_probe`.
  - MuJoCo re-pinned to **3.3.2** (the setup had installed 3.8.1).
  - Installed the NVIDIA graphics package for EGL rendering (the image only had compute).
  - Added 32 GB swap (loading π0 ran out of RAM).
  - Logged in to Hugging Face for the PaliGemma tokenizer (shared token).
- `vqb inspect` matches VLAQuantBench's published layer counts: VE 162 layers / 411.1M params.
- Lost a morning to L4 stockouts in the zone. If the VM won't start, wait an hour and retry.
- *Smitha: add exact package names, machine type and disk size.*

---

## Runs

### E0a · BF16 on clean Spatial, environment A
- Date / who: 2026-10-29 (started 23:03) · Smitha
- Machine / env / commit: qvla-l4 (NVIDIA L4) · vqb_a · Python 3.12.16 · VLAQuantBench `4a2cb7c`, **unpatched**
- Command: **TODO (Smitha: paste the exact command).** No `--episodes`, so it ran the default 50 per task.
- Output file: `results/E0/E0a-BF16-spatial.jsonl`
- Result: **76.0% (152/200)** on the first 20 episodes per task, inside the 71.8–83.2% target
  (published: 78.0%). 76.6% (383/500) on all 500.
  Per task (first 20): 15, 16, 18, 18, 16, 12, 17, 15, 12, 13.
- Checks: header `episodes_per_task: 50`, `quant_report` 0 layers everywhere, 280 steps, 360 px.
  Mean wall time 16.5 s per episode (median 13.2 s), peak VRAM 8.4 GB.
- Statistical comparison with VLAQuantBench's published BF16 file
  (`parity_check.py rollouts --mode statistical`): **PASS**. Gap +2.0 points, task-clustered
  95% CI [−3.5, +7.5], McNemar p = 0.665. But only 152/200 matched episodes have the same
  outcome (26 vs. 22 discordant). Different machine and different noise per episode, so
  episodes can't be compared one by one across runs like these.
- Deviations / notes: ran 500 episodes because `--episodes` was omitted. `--out` needs a
  file name, not a folder.

### E0b · NOOP parity check, environment A (planned)
- Who: Amina
- Plan (see `docs/EXPERIMENT_PLAN.md`, E0): check the clone is `4a2cb7c`, apply patches
  0001–0003; weight check; E0b-1 BF16 (patched) → `E0b-BF16-patched-spatial.jsonl` and E0b-2
  NOOP (VE scope) → `E0b-NOOP-spatial.jsonl`, clean Spatial, `--episodes 20`; exact rollout
  comparison of the two; if there are mismatches, a second BF16 run to measure the chance
  level; statistical comparison of E0a vs. E0b-1. E0a stays the "stack works" result.
