# Per-episode record (JSONL schema)

Every run writes one `.jsonl` file: **line 1 is a run header, every later line is one episode.**
We keep VLAQuantBench's format unchanged (`vlaquantbench/results.py`) and put everything we
add inside the episode's `extra` object, so VLAQuantBench's own tools still read our files.

Status: the header and the base episode fields are confirmed from E0a
(`results/E0/E0a-BF16-spatial.jsonl`). The LIBERO-Plus fields in section 3 are **proposed**
and must be confirmed against `quantvla/libero_plus.py` before E1.

## 1. Header (line 1)

| Field | Example (E0a) | Meaning |
|---|---|---|
| `kind` | `"header"` | Marks the header line |
| `model` | `"pi0"` | Adapter name |
| `checkpoint` | `"lerobot/pi0_libero_finetuned_v044"` | Must be identical in every file we compare |
| `benchmark`, `suite` | `"libero"`, `"libero_spatial"` | Benchmark class and suite run |
| `preset` | `"BASELINE"` | `BASELINE`, `NOOP`, `W4`, `W8`, … |
| `scope` | `"e2e"` | `e2e`, `ve`, `llm`, `ah`, … |
| `method` | `"rtn"` | Quantizer |
| `baseline_dtype` | `"bfloat16"` | Precision the checkpoint runs in |
| `episodes_per_task` | `50` | **Must be 20 for our runs** (pass `--episodes 20`; the default is 50) |
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

## 3. Fields we add in `extra` for LIBERO-Plus episodes (proposed)

| Field | Type | Meaning |
|---|---|---|
| `condition` | str | `clean`, `lighting`, `background`, `layout`, `language` |
| `variant_id` | str or null | LIBERO-Plus variant **name** (stable). `null` for clean episodes. Implementation note 5 |
| `plus_id` | int or null | The variant's `id` in `task_classification.json` (1-based) |
| `difficulty_level` | int or null | Level from `task_classification.json`; `null` levels are never used (note 6) |
| `instruction` | str | The exact text given to π0: base BDDL instruction for visual shifts, rewritten one for language (note 4) |

Layout variants have one stored initial state (note 3): record `init_state_id = 0` and vary the seed.

## 4. Matching rules (needed by `scripts/qrp_bootstrap.py`)

Episodes are matched across precision settings by **(suite, task_id, variant_id, episode)**.

1. `suite` and `task_id` are always the **base** suite and task (0–9), also for shifted
   episodes. The QRP script refuses task IDs that are not in the clean grid.
2. The same rollout must get the same `episode` number in every precision setting.
3. **Recommended:** make `episode` unique within (suite, task_id) across variants, e.g.
   episodes 0–19 of a task spread over its sampled variants. Otherwise two variants of one
   task share an `episode` number and therefore the same noise seed.
4. Every file in a comparison has the same `checkpoint`, `seed` and `episodes_per_task`.
5. Pilot files (E0, E1) never go into the core-grid manifest.
