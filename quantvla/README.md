# quantvla/

Our code that plugs into VLAQuantBench.

| File | Owner | Status | What |
|---|---|---|---|
| `cli.py` | Smitha | todo | Registers `libero_plus` in VLAQuantBench's `BENCHMARKS`, then calls `vlaquantbench.cli.main()` (exposed as `qvla`) |
| `libero_plus.py` | Smitha | todo | `LiberoPlusRunner`: clean base tasks, variant sampling, init-state mapping, base-BDDL instruction, extra record fields |
| `NOOP` preset | Amina | done, as a VLAQuantBench patch | `patches/vlaquantbench/0001-…`: same layer walk as W4 for the scope, weights written back unchanged |

The `NOOP` preset ended up as a small patch to the pinned VLAQuantBench clone instead of a
separate `noop.py`, together with two related fixes (`0002` episode seeds, `0003` default
output folder). Why, and how to apply them: `docs/SETUP.md`, section 4.

**For `libero_plus.py`:** the runner must hand each episode's seed to the adapter before
`adapter.reset()`. VLAQuantBench's own loop already does this after patch `0002`
(`BenchmarkRunner.run` calls `adapter.set_episode_seed(...)`), so a runner that uses that loop
gets it for free. Record fields and matching rules: `docs/JSONL_SCHEMA.md`.

See `docs/IMPLEMENTATION_NOTES.md` before starting.
