# results/

Per-episode JSONL files from our own runs (format: `docs/JSONL_SCHEMA.md`). Every file here
has a matching entry in `RUN_LOG.md` with the exact command.

## Layout

```
results/<experiment>/[<condition>/]<experiment>-<setting>-<suite>.jsonl
```

| Experiment | Example |
|---|---|
| E0 (pilot) | `results/E0/E0a-BF16-spatial.jsonl`, `results/E0/E0b-NOOP-spatial.jsonl` |
| E1 (pilot) | `results/E1/lighting/E1-BF16-spatial-L2.jsonl` (level in the name) |
| E2 | `results/E2/clean/E2-VE-W4-object.jsonl` |
| E3 | `results/E3/lighting/E3-W4-goal.jsonl` |
| E4 | `results/E4/language/E4-BF16-spatial.jsonl` |

**Setting names:** `BF16`, `NOOP`, `VE-W4`, `W4` (whole model), `W8` (whole model),
optional `AH-W4`, `VE-W2`. **Suites:** `spatial`, `object`, `goal`.

The setting comes before the suite on purpose: the analysis selects files with patterns like
`E2-W4-*`, which then can't accidentally also pick up `E2-VE-W4-*`.

## Rules

- Pass the full file name to `--out` (not a folder) and always `--episodes 20`.
- Never edit a results file by hand. If a run is wrong, rerun it to a new file and note it in `RUN_LOG.md`.
- Pilot files (E0, E1) are never used in a hypothesis test.
- **No videos in git.** Videos stay on the VM under `/opt/qvla` (path TBD); E5 coding uses random video IDs.
