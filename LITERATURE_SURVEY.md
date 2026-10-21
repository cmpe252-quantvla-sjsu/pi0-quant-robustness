# Literature survey

How each piece of prior work feeds our project. Numbers in brackets match the proposal's reference list.

## Vision-language-action models

**π0** [5] — Black et al., RSS 2025. A PaliGemma backbone (SigLIP image encoder + Gemma language model) with a separate flow-matching action expert that produces 50-step action chunks. The modular structure is what lets us quantize the vision encoder alone. We use the LeRobot port, `lerobot/pi0_libero_finetuned_v044` [6], which is about 3.3B parameters. Its success rates differ from the original JAX release, so we compare only against our own BF16 runs.

**OpenVLA** [12] — Kim et al., CoRL 2024. A 7B open VLA. It reported real-robot success with 4-bit inference comparable to full precision, the kind of clean-condition evidence our study puts under pressure.

## Post-training quantization

**Nagel et al. white paper** [13] — The standard reference for uniform round-to-nearest (RTN) quantization with per-group or per-channel scales. We use RTN because it needs no calibration data. Any change in robustness then reflects reduced precision, not a calibration set drawn from clean scenes.

## Quantization of VLAs

**VLAQuantBench** [1] — Xu et al., 2026. The closest work and our code base: 409 analyzed closed-loop runs (94,574 episodes) on four VLAs, varying bit width, calibration and quantized component. It releases per-episode records, a same-observation replay tool and task-clustered intervals. Its precision comparisons are all under standard conditions. Its only visually varied runs (SIMPLER variant aggregation) were excluded from analysis because their coverage was unequal. Its released π0 logs give our motivating numbers: on clean Spatial, VE-W4 76.5% vs. BF16 78.0%, while AH-W4 drops to 68.5%. It also warns that MuJoCo ≥ 3.4 and open-loop chunk execution each silently move success by more than a typical method difference.

**QuantVLA** [7] — Scale-calibrated PTQ for VLAs (CVPR 2026). **HBVLA** [8] pushes PTQ to 1 bit. **DA-PTQ** [9] addresses how quantization error accumulates over a rollout, which is the closed-loop compounding that motivates measuring task success rather than action error alone. All three report results under standard benchmark conditions only.

## Robustness of quantized perception

**Xiao et al.** [11] — Lower-bit image classifiers are more vulnerable to natural corruptions and noise. **Recti-Q** [10] — 4-bit perception models with negligible clean loss can degrade substantially out of distribution. Both are open-loop perception results. Our question is whether the same thing happens when a robot repeatedly observes and acts.

## Benchmarks

**LIBERO** [4] — Liu et al., NeurIPS 2023 D&B. Manipulation suites built on MuJoCo [14] and robosuite [15]. We use Spatial, Object and Goal (10 tasks each). We exclude Long because in VLAQuantBench's logs even W8 flips 23% of matched Long episodes.

**LIBERO-Plus** [2] — Fei et al., CVPR 2026. 10,030 perturbed variants across seven dimensions, with per-variant difficulty levels in `task_classification.json`. We use lighting, background texture and object layout as visual shifts, and language rewrites as a non-visual comparison. We exclude robot initial state (not visual), sensor noise (a public reproduction found a much lower π0 baseline than published [3]), and camera viewpoint, which is optional only if a level falls in range. GitHub issue #65 [3] documents that prompts derived from variant filenames leak perturbation metadata. We confirmed this in the code and pass the base BDDL instruction instead.

## Statistics

**Field & Welsh** [16] — Bootstrapping clustered data; the basis for our paired, task-clustered bootstrap. **Bates et al., lme4** [17] — Logistic mixed-effects model for the sensitivity analysis. **Wilson** [18] — Per-cell descriptive intervals. **Holm** [19] — Step-down correction for the five secondary tests.

## The gap we fill

VLAQuantBench compares precision settings without controlled visual shift. LIBERO-Plus tests shifts without varying precision. We evaluate the same policy at several precision settings under clean and shifted conditions, with matched episodes within each condition. To our knowledge, this is the first closed-loop measurement of how a VLA's quantization penalty changes under controlled visual shift.
