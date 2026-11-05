# Per-episode record (JSONL schema)

Every run writes one `.jsonl` file: **line 1 is a run header, every later line is one episode.**
We keep VLAQuantBench's format unchanged (`vlaquantbench/results.py`) and put everything we
add inside the episode's `extra` object, so VLAQuantBench's own tools still read our files.

Status: the header and the base episode fields are confirmed from E0a
(`results/E0/E0a-BF16-spatial.jsonl`). The LIBERO-Plus fields in section 3 are **confirmed**
against `quantvla/libero_plus.py` (runner PR, reviewed Nov 4) and the Smoke-B records.

## 1. Header (line 1)

| Field | Example (E0a) | Meaning |
|---|---|---|
| `kind` | `"header"` | Marks the header line |
| `model` | `"pi0"` | Adapter name |
| `checkpoint` | `"lerobot/pi0_libero_finetuned_v044"` | Must be identical in every file we compare |
| `benchmark`, `suite` | `"libero"`, `"libero_spatial"` | Benchmark class and suite run. `libero` = environment A, `libero_plus` = environment B; **all core files are `libero_plus`** |
| `preset` | `"BASELINE"` | `BASELINE`, `NOOP`, `W4`, `W8`, … |
| `scope` | `"e2e"` | `e2e`, `ve`, `llm`, `ah`, … |
| `method` | `"rtn"` | Quantizer |
| `baseline_dtype` | `"bfloat16"` | Precision the checkpoint runs in |
| `episodes_per_task` | `50` | Clean runs: **20** (pass `--episodes 20`; the default is 50). Shifted runs: set by the sample file (e.g. 10); leave `--episodes` out |
| `seed` | `0` | Run seed |
| `quant_report` | `{"ve": {"layers": 0, "params": 0, "mean_rel_mse": null}, …}` | What was actually quantized, per component |
| `extra` | `{"adapter_options": {}, …}` | CLI options |
| `created`, `hostname`, `gpu`, `python`, `vqb_commit` | `"qvla-l4"`, `"NVIDIA L4"`, `"3.12.16"`, `"4a2cb7c"` | Provenance |

**Checks on `quant_report`:**

| Preset / scope | Expected |
|---|---|
| `BASELINE` | 0 layers everywhere |
| `NOOP` / `ve` and `W4` / `ve` | `ve`: **162 layers, 411.1M params** (from `vqb inspect` on our VM; confirmed in the E0b NOOP header), `mean_rel_mse` 0 for NOOP |
| `W4` / `e2e`, `W8` / `e2e` | all four components nonzero |

## 2. Episode lines (confirmed from E0a)

| Field | Type | Meaning |
|---|---|---|
| `kind` | `"episode"` | |
| `suite` | str | **Base** LIBERO suite: `libero_spatial`, `libero_object`, `libero_goal` |
| `task_id` | int 0–9 | **Base** LIBERO task index within the suite |
| `task_name` | str | Base task name |
| `episode` | int | Episode index; see the matching rules below |
| `seed` | int | Episode seed = `run_seed × 1,000,003 + task_id × 1,009 + episode`. With the seed patch this is also the flow-matching noise seed |
| `success` | bool | Task completed before the step limit |
| `steps` | int | Policy steps taken (= `max_steps` when it failed by timeout) |
| `wall_time_s` | float | Episode wall time (E0a: mean 16.5 s, median 13.2 s on the L4) |
| `subtasks_completed`, `progress_score` | null | Not used on LIBERO |
| `policy_latency_*`, `inference_latency_*`, `n_inferences` | float/int | Timing |
| `peak_vram_gb` | float | E0a: 8.4 GB |
| `extra.max_steps` | int | 280 Spatial / 280 Object / 300 Goal |
| `extra.camera_size` | int | 360 |
| `extra.action_mode` | str | `"delta"` |
| `extra.init_state_id` | int | Which stored initial state was loaded |

## 3. Fields the LIBERO-Plus runner adds in `extra` (environment B)

Every episode of a `libero_plus` run, clean ones included, carries these fields. Shifted
episodes copy them from the sample file written by `scripts/sample_variants.py`.

| Field | Type | Clean episode | Shifted episode |
|---|---|---|---|
| `condition` | str | `"clean"` | `lighting`, `background`, `layout` or `language` |
| `variant_id` | str or null | `null` | LIBERO-Plus variant **name** (stable; implementation note 5) |
| `plus_id` | int or null | `null` | The variant's `id` in `task_classification.json` (1-based) |
| `difficulty_level` | int or null | `null` | The variant's real level. `null` levels are never sampled (note 6) |
| `target_level` | int or null | `null` | The level chosen for this shift (per suite if levels differ by suite) |
| `fallback` | bool | `false` | `true` if the task had no variant at `target_level`, so the nearest level was used (ties go to the lower level) |
| `instruction` | str | Standard instruction | See below |
| `init_state_id` | int | = `episode` | = `episode`, except single-state layout variants (`_add_`, `_level`): `0` |

**Instruction rule** (what π0 is told, recorded so the prompt audit can run from the JSONL):
- clean and visual shifts: the **standard instruction**, the base task's file name with
  spaces (`"pick up the black bowl between the plate and the ramekin and place it on the
  plate"`), exactly what VLAQuantBench gives π0 on clean LIBERO and what E0 used. The BDDL's
  own `(:language ...)` line differs ("Pick the akita black bowl ...") and is **not** used.
- language shifts: the **rewritten** `(:language ...)` text from the variant's BDDL.

So the prompt audit checks: visual-shift `instruction` == the clean instruction of the same
task; language `instruction` != it.

## 4. Matching rules (needed by `scripts/qrp_bootstrap.py`)

Episodes are matched across precision settings by **(suite, task_id, variant_id, episode)**.

1. `suite` and `task_id` are always the **base** suite and task (0–9), also for shifted
   episodes. The QRP script refuses task IDs that are not in the clean grid.
2. The same rollout must get the same `episode` number in every precision setting.
3. `episode` runs 0..n−1 per task across that task's sampled variants (the sampler does
   this), so it is unique within (suite, task_id), and shifted episode *e* gets the same seed
   as clean episode *e*.
4. Every file in one analysis has the same `checkpoint`, `seed`, `hostname`, `gpu`,
   `vqb_commit` and `benchmark`; `episodes_per_task` must agree **within** each condition
   (W8, a subset, is exempt). `qrp_bootstrap.py` stops otherwise.
5. **Never pair environment A and environment B episodes** (decision Nov 4): across the
   environments 17/200 outcomes flip, within one 1–2. The `benchmark` check enforces it.
6. Pilot files (E0, E1) never go into the core-grid manifest.
