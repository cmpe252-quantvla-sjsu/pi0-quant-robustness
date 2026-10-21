# Does Nearly Lossless Vision-Encoder Quantization Remain Robust Under Distribution Shift?

**A closed-loop study of π0** · CMPE 252 (AI and Data Engineering), San José State University, Fall 2026
Amina Akhtar · Smitha Soman

---

## The question

VLAQuantBench's released logs show that quantizing only π0's vision encoder to 4 bits costs about 1.5 points of task success on clean LIBERO-Spatial (78.0% → 76.5%). Clean benchmarks control lighting, backgrounds and object layout, though, and a real robot doesn't get that luxury. We ask whether that small clean loss stays small when the scene changes.

We run one released π0 checkpoint in full precision (BF16) and under a few pre-specified quantization settings, on clean LIBERO and on LIBERO-Plus scenes with changed lighting, backgrounds and object layouts. Language rewrites serve as a non-visual comparison. All settings within a condition run the same matched episodes. No training is involved.

## The main measure: Quantization Robustness Penalty (QRP)

```
QRP_p = (SR_BF16,p − SR_Q,p) − (SR_BF16,clean − SR_Q,clean) = Δ_p − Δ_clean
```

QRP > 0 means quantization hurts more under shift `p` than on clean scenes. The primary result is the pooled QRP for vision-encoder W4 across lighting, background and layout, with a 95% paired, task-clustered bootstrap interval (10,000 replicates).

**Decision rule for H1** (fixed before any core run):

| 95% interval for pooled QRP | Reading |
|---|---|
| Entirely above +5 points | Positive and practically meaningful |
| Entirely above 0, but not above +5 | Positive; size uncertain |
| Includes 0, upper end below +5 | No practically important penalty |
| Includes both 0 and values above +5 | Inconclusive (not evidence of robustness) |
| Upper end below 0 | Shift *reduces* the penalty (unexpected) |

## Hypotheses

- **H1 (primary):** VE-W4 has a positive pooled QRP across the three visual shifts.
- **H2:** Whole-model W4 has a positive pooled QRP.
- **H3:** VE-W4's QRP is larger for visual shifts than for language rewrites.

## Precision settings

| Setting | Role | Clean Spatial SR (VLAQuantBench logs) |
|---|---|---|
| BF16 | Reference | 78.0% |
| No-op hook (identity) | Harness check; must match BF16 | – |
| W8, whole model | Mild dose | 77.5% |
| **W4, vision encoder only** | **Primary treatment** | **76.5%** |
| W4, whole model | Secondary | 69.0% |
| W4, action head only *(optional)* | Contrast away from vision | 68.5% |
| W2, vision encoder only *(optional)* | Dose extension | 76.5% |

W4 is asymmetric per-group RTN (group size 128); W8 is symmetric per-channel. Only weights are quantized, and they are stored back in BF16, so we measure accuracy, not speed or memory.

## Experiments

| ID | What | Episodes | Owner |
|---|---|---|---|
| E0 | Harness validation in VLAQuantBench's original env: BF16 + no-op on clean Spatial | 400 | Both |
| E1 | Final-env checks, difficulty calibration, discordance, power update | ~1,150 | Both |
| E2 | Clean grid: 4 settings × Spatial/Object/Goal | 2,400 | Amina |
| E3 | Visual-shift grid: lighting, background, layout | 3,000 | Smitha |
| E4 | Language-rewrite contrast | 900 | Amina |
| E5 | Blinded failure-mode coding (required) | analysis | Both |
| E6 | Same-observation action replay (optional) | replay only | whoever has capacity |
| E7 | AH-W4 and/or VE-W2 (optional) | ≤3,000 | whoever has capacity |

E0 and E1 are pilots and are excluded from all hypothesis tests. The step-by-step runbook is in [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

## Repository layout

```
pi0-quant-robustness/
├── README.md                  ← you are here
├── CONTRIBUTING.md            ← branch, review-swap and same-machine rules
├── RUN_LOG.md                 ← shared log: every run, every machine, every surprise
├── LITERATURE_SURVEY.md
├── docs/
│   ├── SETUP.md               ← environment A (original) and B (final, LIBERO-Plus)
│   ├── EXPERIMENT_PLAN.md     ← E0–E7 step by step, with checks and owners
│   ├── JSONL_SCHEMA.md        ← the shared per-episode record
│   ├── IMPLEMENTATION_NOTES.md← what we found reading VLAQuantBench and LIBERO-Plus
│   └── FAILURE_CODEBOOK.md    ← E5 codebook, frozen before anyone watches a video
├── configs/
│   └── frozen_design.yaml     ← difficulty levels, sample sizes, seeds; filled at design freeze
├── quantvla/                  ← our code: LIBERO-Plus runner, no-op preset, CLI wrapper
├── scripts/                   ← sampling, validation, flattening, QRP bootstrap, mixed model
├── analysis/                  ← notebooks and figures for the report
└── results/                   ← per-episode JSONL (git), videos (NOT in git)
```

## What we build on

| Resource | License | Use |
|---|---|---|
| [VLAQuantBench](https://github.com/jiuyixu25/VLAQuantBench) (pinned `4a2cb7c`) | MIT | Quantizer, π0 adapter, LIBERO runner, replay and clustered-CI scripts, released logs |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (pinned `4976dc3`) + [assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | HF metadata lists MIT | Shifted task variants |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT / CC BY 4.0 | Clean tasks, initial states |
| [LeRobot](https://github.com/huggingface/lerobot) 0.5.2 | Apache 2.0 | π0 implementation |
| [`lerobot/pi0_libero_finetuned_v044`](https://huggingface.co/lerobot/pi0_libero_finetuned_v044) | Gemma terms | The policy under study |
| MuJoCo 3.3.2, robosuite 1.4.0 | Apache 2.0, MIT | Simulation |

Our three additions: a **LIBERO-Plus benchmark class**, a **no-op quantization preset**, and a **QRP analysis script**.

## Ground rules (short version; details in CONTRIBUTING.md)

1. **All core episodes run on one machine and one software environment.** We split who is responsible, not where it runs.
2. **Nobody merges their own shared code until the other person has run it.**
3. **Difficulty levels, sample sizes and decision rules are frozen after E1**, committed to `configs/frozen_design.yaml` and tagged `design-freeze` before any core run.
4. **Every run gets a line in `RUN_LOG.md`.**
5. **Experiment freeze Nov 20.** No new rollouts after that.

## Timeline

| Dates | Milestone | Owner |
|---|---|---|
| Oct 15–19 | Env setup; reproduce BF16 on LIBERO-Spatial (target 71.8–83.2%); agree JSONL schema | Both |
| Oct 20–26 | LIBERO-Plus class and final environment | Smitha |
| Oct 20–26 | No-op preset, QRP bootstrap script, clean parity check | Amina |
| Oct 27–31 | Pilot runs, difficulty calibration, power update, **design freeze** | Both |
| Nov 1–14 | Core grid: clean + language (3,300 episodes) | Amina |
| Nov 1–14 | Core grid: lighting, background, layout (3,000 episodes) | Smitha |
| Nov 15–20 | Blinded failure coding; optional replay/settings; **experiment freeze Nov 20** | Both |
| Nov 21–Dec 5 | Analysis and report writing | Both |
| Dec 9 / Dec 16 | Final presentation / final report | Both |

## Citation of the main tools

```bibtex
@misc{xu2026vlaquantbench,
  title  = {VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models},
  author = {Xu, Jiuyi and Jin, Qing and Chen, Meida and Wang, Song and Sui, Yang and Shi, Yangming},
  year   = {2026}, note = {arXiv:2609.25376}
}
@inproceedings{fei2026liberoplus,
  title     = {LIBERO-Plus: A Progressive Robustness Benchmark for Vision-Language-Action Models},
  author    = {Fei, Senyu and Wang, Siyin and Shi, Junhao and others},
  booktitle = {CVPR}, year = {2026}
}
```
