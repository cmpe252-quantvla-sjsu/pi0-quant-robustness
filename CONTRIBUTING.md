# Working together

Rules for the two of us. The short version is in the README's "Ground rules".

## 1. Same machine, same environment

- **All core episodes (E2–E4) run on `qvla-l4` in environment `vqb_b`.** We split who is
  responsible for a run, not where it runs.
- Don't change installed packages in a shared environment without writing it in
  `RUN_LOG.md` first. After the design freeze, don't change them at all.
- Check `mujoco` is 3.3.2 at the start of every session (`docs/SETUP.md`).
- Run long jobs inside `tmux`. Stop the VM when you're done (see `docs/SETUP.md` for when
  to leave it running during the core runs).

## 2. Branches and commits

- `main` holds reviewed work. Do your work on a branch named `<name>/<topic>`, e.g.
  `amina/qrp-script` or `smitha/libero-plus-runner`.
- Results files can go straight to `main` once the run is logged.
- Start commit messages with the experiment or area in brackets:
  `[E0] E0b NOOP parity: PASS`, `[docs] add JSONL schema`, `[qrp] fix W8 subset matching`.
- Never commit videos, model weights or anything large. Videos stay on the VM
  (`results/README.md`).

## 3. Review swap

**Nobody merges their own shared code until the other person has run it.**

Shared code: the LIBERO-Plus runner (`quantvla/`), the no-op preset and VLAQuantBench
patches, and the analysis scripts (`scripts/`).

To get a review:
1. Push your branch and open a pull request.
2. The reviewer pulls the branch and **runs** it: the tests, plus a few episodes or a
   script run on real files. Reading alone doesn't count.
3. The reviewer writes in the pull request what they ran and what they saw, then approves.
4. The author merges.

## 4. Logging

- Every run gets an entry in `RUN_LOG.md`, including failed, stopped and pilot runs.
- Every result file has a matching `RUN_LOG.md` entry with the exact command.

## 5. Freezes

- **Design freeze (~Nov 7):** after E1, fill in `configs/frozen_design.yaml`, commit it and
  tag it: `git tag design-freeze && git push origin design-freeze`. Difficulty levels,
  sample sizes, seeds and decision rules don't change after this tag.
- **Experiment freeze (Nov 20):** no new rollouts after this date.
- Pilot runs (E0, E1) are never used in a hypothesis test.

## 6. Tests

```bash
python -m pytest -q tests
```
Run them before opening a pull request that touches `scripts/` or `quantvla/`. They need
the patched VLAQuantBench clone installed with `pip install -e`.

## 7. Use of AI tools

**Course policy:** TODO: quote the syllabus / instructor wording here.

We use AI tools for code and for drafting repository docs (setup notes, run log,
plans). Whoever submits AI-assisted code or docs is responsible for checking and
understanding it, and AI-assisted code goes through the review swap above like any other.
If the course policy asks for it, the final report says how AI tools were used.
