# Implementation notes

What we found reading VLAQuantBench (`4a2cb7c`, Sep 21 2026) and LIBERO-Plus (`4976dc3`, Jan 21 2026) before writing any code. Each note says what it means for our build. Items marked **verify** are read from the code but not yet confirmed by running it.

## LIBERO-Plus

**1. The clean base tasks are not in LIBERO-Plus's suite lists.**
After `pip install -e LIBERO-plus`, `get_benchmark_dict()["libero_spatial"]` has 2,402 tasks, Object has 2,518 and Goal has 2,591. All of them are perturbed variants. None of the 10 original task names appears in the list. So running stock `vqb run --benchmark libero` in the final environment would iterate over variants, not the clean suite.
→ Our LIBERO-Plus class needs a `clean` condition that builds the 10 base tasks itself. The base `.bddl` files and base `.pruned_init` files are still in the repo, so this is straightforward: take the 10 base names from the original LIBERO task map.

**2. Variant init-state files don't exist under their own names.**
VLAQuantBench's runner loads `init_files/<suite>/<task>.pruned_init` directly. A variant like `..._light_3` has no file of that name. LIBERO-Plus resolves it inside `Benchmark.get_task_init_states(i)` by stripping `_table_N`, `_light_N`, `_tb_N`, `_view_…`, `_language_…` back to the base task's file.
→ Our runner should call LIBERO-Plus's `get_task_init_states`, or copy its mapping, and pass `weights_only=False` to `torch.load` (torch ≥ 2.6 needs it; VLAQuantBench already works around the same issue).

**3. Layout variants have only one stored initial state.**
Variants named `_add_N` or `_levelN` load from `init_files/libero_newobj/…` and are reshaped to a single state.
→ For layout, the 20 "initial-state/seed pairs" have to come from seeds, not stored init states. The bootstrap must cluster by variant within task, as the proposal already says for low-count levels. Record `init_state_id = 0` and vary `seed`.

**4. The prompt leak is real.**
For non-language variants, LIBERO-Plus's `Task.language` is built from the filename, so π0 would read "pick up the alphabet soup and place it in the basket light 1". (This is LIBERO-Plus issue #65.)
→ For visual shifts, pass the base task's BDDL instruction. For the language condition, pass the rewritten instruction from the variant's BDDL. Amina's prompt audit checks a sample of both.

**5. Task IDs are 1-based in the classification file.**
`task_classification.json` entry `id` equals list index + 1 in `libero_suite_task_map`, for all three suites (checked).
→ `variant_id` in our records = the variant's **name** (stable), plus `plus_id` = the JSON id. Never store only the index.

**6. Some variants have no difficulty level.**
121 Light variants in `libero_goal` have `difficulty_level: null`.
→ The variant selector must skip or separately bucket `null` levels, never treat them as 0.

**7. Variant counts per (dimension, level) are uneven.**
For example, Spatial lighting has 52 / 104 / 92 / 38 / 6 variants at levels 1–5, Object lighting has 2 / 9 / 71 / 110 / 105, and Spatial layout has only 6 at level 1. The proposal picks one level per dimension from the E1 pilot. Low-count levels will need repeat seeds on the same variants.
→ `scripts/sample_variants.py` should report the variant count per task at each level before E1 so we know which levels are feasible.

## Environment

**8. The robosuite clash is smaller than we wrote, but the package clash is real.**
VLAQuantBench's π0 environment (`scripts/env/setup_lerobot.sh`) installs `lerobot[pi]==0.5.2` and `hf_libero` on Python 3.12. `hf_libero` 0.1.4 pins `robosuite==1.4.0`, the same version LIBERO-Plus pins. The actual conflicts are:
- LIBERO-Plus's `requirements.txt` pins old `numpy==1.22.4`, `transformers==4.21.1` and `gym==0.25.2`, which would break LeRobot (numpy 1.22 doesn't support Python 3.12).
- Both `hf_libero` and LIBERO-Plus install a package called `libero`.
- `~/.libero/config.yaml` has to point to the LIBERO-Plus tree, not the pinned LIBERO clone.

→ **Plan (verify):** clone the lerobot env, `pip uninstall hf_libero`, `pip install -e LIBERO-plus --no-deps`, then add only `wand scikit-image usd-core` plus apt `libmagickwand-dev`. Unzip the 6.4 GB assets into `LIBERO-plus/libero/libero/assets`. Re-pin `mujoco==3.3.2`.
→ For the final report, Section 6.4 should probably say "different dependency pins" rather than "a different robosuite version".

**9. MuJoCo must be 3.3.2.** Versions ≥ 3.4.0 change box–box collision, so one Spatial task's stored initial state no longer settles, which costs 3–9 suite points silently. Check `python -c "import mujoco; print(mujoco.__version__)"` at the start of every session.

**10. LeRobot π0 protocol.** The adapter declares 360×360 cameras and 280 steps on Spatial (instead of 256 and 220). The runner honors this and writes it into every episode's `extra`. Replanning every 10 actions is required; a released config that executes the full 50-step chunk open-loop loses 10–25 points.

## VLAQuantBench

**11. `BASELINE` skips the quantization path entirely.** `parse_spec("BASELINE")` (also `BF16`, `W16`) returns a spec with no weight or activation quantizer, so nothing is walked or hooked. That makes it useless as a no-op control.
→ The no-op preset must go through `quantize_modules` for the chosen scope and write each weight back unchanged. The run header's `quant_report` should then show the **same layer and parameter counts** as VE-W4 (or e2e-W4). That is how we know it exercised the same code path.

**12. Weights are fake-quantized in place.** Hooks are only used for activations. Our settings are weight-only, so "hook" in our docs means the in-place weight transform and its bookkeeping.

**13. Released π0 logs.** `results/libero/<suite>/pi0/rtn-<PRESET>-<scope>.jsonl`. Spatial has BASELINE, W2/W3/W4 for `ve`, `llm`, `ah`, `e2e`, plus W8-e2e. We re-checked the proposal's Table 1 from these files: BF16 78.0, VE-W4 76.5, W8 77.5, W4-e2e 69.0, AH-W4 68.5, VE-W2 76.5 (200 episodes each, 280 steps, 360 px). **Object and Goal have no `ve` runs**, only `e2e` (plus one AH-W4A4 on Object). So VE-W4 on Object and Goal is new, with no published reference.

**14. Record format.** Line 1 is a header (model, checkpoint, preset, scope, method, `quant_report`, seed, git commit, host, GPU). Each later line is one episode: `suite, task_id, task_name, episode, seed, success, steps, wall_time_s, …, extra{max_steps, camera_size, action_mode, init_state_id}`. Runs are resumable, and a header mismatch is refused. We keep this format and add our fields to `extra` (see `JSONL_SCHEMA.md`).

**15. Adding a benchmark.** The runner subclasses `BenchmarkRunner`, implements `tasks()` and `run_episode()`, and is registered in `vlaquantbench/registry.py::BENCHMARKS`. The CLI builds `--benchmark` choices from that dict when it starts.
→ To avoid forking: `quantvla/cli.py` adds `"libero_plus": "quantvla.libero_plus:LiberoPlusRunner"` to `BENCHMARKS` and then calls `vlaquantbench.cli.main()`. If that turns out awkward, fork VLAQuantBench and pin our fork as a submodule.

**16. Existing scripts we extend.** `scripts/clustered_ci.py` already does a task-paired cluster bootstrap for one cell vs. a reference. QRP needs four cells (BF16 and Q, clean and shifted) resampled with the **same task draw** in each replicate and the clean gap reused across shifts. `scripts/action_fidelity.py` and `replay_pi05_subsets.py` are the starting points for E6.
