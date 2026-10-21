# quantvla/

Our code that plugs into VLAQuantBench.

| File | Owner | Status | What |
|---|---|---|---|
| `cli.py` | Smitha | todo | Registers `libero_plus` in VLAQuantBench's `BENCHMARKS`, then calls `vlaquantbench.cli.main()` (exposed as `qvla`) |
| `libero_plus.py` | Smitha | todo | `LiberoPlusRunner`: clean base tasks, variant sampling, init-state mapping, base-BDDL instruction, extra record fields |
| `noop.py` | Amina | todo | `NOOP` preset: same layer walk as W4 for the scope, weights written back unchanged |

See `docs/IMPLEMENTATION_NOTES.md` before starting.
