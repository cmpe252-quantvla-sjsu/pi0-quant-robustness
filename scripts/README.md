# scripts/

Command-line tools for sampling, checking and analysing runs. Run them from the repo root.
They need the patched VLAQuantBench clone installed (`pip install -e`, see `docs/SETUP.md`).

| Script | Owner | Status | What |
|---|---|---|---|
| `parity_check.py` | Amina | ready | `weights`: does a preset change the model's weights? (NOOP must change nothing). `rollouts`: do two runs agree on matched episodes? `--mode exact` (same success and steps) or `--mode statistical` (Wilson + McNemar) |
| `qrp_bootstrap.py` | Amina | ready | Primary analysis: success rates, matched gaps, per-shift and pooled QRP with the paired task-clustered bootstrap (tasks-only primary, two-stage as sensitivity), H1 decision, Holm tests. Stops if cells differ in host, GPU, commit, seed or episodes per task. Reads a manifest from `configs/qrp/` |
| `sample_variants.py` | Smitha | todo | Variant counts per task and level; samples variants for E1 and the core runs (implementation note 7) |
| validation / flattening / mixed model | — | todo | Prompt audit, JSONL → table for R, `lme4` sensitivity model (proposal Eq. 7) |

## Examples

```bash
# Is the NOOP preset harmless? (needs the model, so on the VM)
python scripts/parity_check.py weights --model pi0 --preset NOOP --scope ve

# Do two runs match episode by episode?
python scripts/parity_check.py rollouts --ref results/E0/E0b-BF16-patched-spatial.jsonl --cell results/E0/E0b-NOOP-spatial.jsonl

# QRP check on VLAQuantBench's released logs: the QRP point estimate must be exactly 0
# (the interval is [0, 0] with the default tasks-only bootstrap)
python scripts/qrp_bootstrap.py configs/qrp/released_logs_check.yaml --base-dir ../VLAQuantBench

# The real analysis, after the core runs
python scripts/qrp_bootstrap.py configs/qrp/study_manifest.yaml --base-dir . --out-md analysis/qrp.md
```

`parity_check.py` exits with status 0 on PASS and 1 otherwise. Tests: `tests/test_parity_check.py`, `tests/test_qrp_bootstrap.py`.
