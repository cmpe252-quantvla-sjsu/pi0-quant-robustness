#!/usr/bin/env python
"""E1 power update (proposal 7.4): how precise will the pooled visual QRP be?

Simulates the core design many times with success rates and discordance taken from the
E1 pilot, runs the real analysis (qrp_bootstrap.analyse, tasks-only bootstrap) on every
simulated data set, and reports the standard error (SE) of the pooled VE-W4 QRP for each
candidate shifted sample size (300 and 450 per shift).

Decision rule (proposal 7.4): if the projected SE is above ~2.5 points, raise the shifted
sample size (up to 450 per shift) before the design freeze.

Inputs
- Clean BF16 success per task: VLAQuantBench's released BF16 logs (same checkpoint, 20 per
  task, all three suites). Environment A, but used only as planning numbers.
- Shifted BF16 success per suite: E1b calibration files (BF16, chosen level per shift), plus
  the E1c BF16 episodes 2-3 for lighting and the top-up for background (120 episodes each).
  Per-task rates are the clean ones moved on the logit scale to match each suite's E1b rate.
- Discordance: E1c lighting VE-W4 vs BF16 matched pairs, as two rates:
  a = P(VE-W4 fails | BF16 succeeds), b = P(VE-W4 succeeds | BF16 fails).
  Clean VE-W4 has no environment-B data yet, so clean uses the same a and b.

Scenarios: "E1c" (rates as observed), "2x" (a and b doubled), and "uneven" (each task's a
drawn around the E1c value with a mean-1 exponential multiplier, i.e. quantization hurts some
tasks much more than others; this widens the task-level spread the bootstrap sees).

    python scripts/power_update.py --e1 path/to/E1-files --vqb ../VLAQuantBench
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qrp_bootstrap as q  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal")
SHIFTS = {"lighting": ["E1b/visual/BF16-lighting-L3-*.jsonl", "E1c/BF16-ep23-*.jsonl"],   # episodes 0-1 and 2-3
          "background": "E1b/visual/BF16-background-L1*.jsonl",     # includes the top-up
          "layout": "E1b/visual/BF16-layout-L3-*.jsonl"}
E1C_BF16 = ["E1b/visual/BF16-lighting-L3-*.jsonl", "E1c/BF16-ep23-*.jsonl"]   # BF16 episodes 0-1 and 2-3
E1C_Q = "E1c/VEW4-*.jsonl"


# --------------------------------------------------------------------------- #
# Inputs from the pilot
# --------------------------------------------------------------------------- #
def episodes(files) -> dict:
    out = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            if e.get("kind") == "episode":
                key = (e["suite"], int(e["task_id"]), (e.get("extra") or {}).get("variant_id"), int(e["episode"]))
                out[key] = int(bool(e["success"]))
    return out


def files(root: Path, pattern) -> list:
    pats = [pattern] if isinstance(pattern, str) else pattern
    found = sorted(f for p in pats for f in glob.glob(str(root / p)))
    if not found:
        raise SystemExit(f"no files for {pats} under {root}")
    return found


def clean_task_rates(vqb: Path) -> dict:
    """(suite, task) -> clean BF16 success rate from VLAQuantBench's released logs."""
    rates = {}
    for s in SUITES:
        eps = episodes([vqb / "results" / "libero" / s / "pi0" / "rtn-BASELINE-e2e.jsonl"])
        for t in range(10):
            ys = [y for k, y in eps.items() if k[1] == t]
            rates[(s, t)] = sum(ys) / len(ys)
    return rates


def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def shift_task_rates(clean: dict, suite_target: dict) -> dict:
    """Move each suite's clean task rates on the logit scale until their mean matches the target."""
    out = {}
    for s in SUITES:
        ts = [k for k in clean if k[0] == s]
        lo, hi = -10.0, 10.0
        for _ in range(60):                    # bisection on the logit offset
            mid = (lo + hi) / 2
            m = np.mean([1 / (1 + math.exp(-(logit(clean[k]) + mid))) for k in ts])
            lo, hi = (mid, hi) if m < suite_target[s] else (lo, mid)
        for k in ts:
            out[k] = 1 / (1 + math.exp(-(logit(clean[k]) + (lo + hi) / 2)))
    return out


def pilot_inputs(e1: Path, vqb: Path) -> dict:
    clean = clean_task_rates(vqb)
    shifts = {}
    for name, pat in SHIFTS.items():
        eps = episodes(files(e1, pat))
        target = {s: np.mean([y for k, y in eps.items() if k[0] == s]) for s in SUITES}
        shifts[name] = {"suite_rates": target, "n": len(eps), "task_rates": shift_task_rates(clean, target)}
    ref = episodes(files(e1, E1C_BF16))
    qq = episodes(files(e1, E1C_Q))
    keys = sorted(set(ref) & set(qq))
    n1 = sum(ref[k] for k in keys)
    n10 = sum(ref[k] == 1 and qq[k] == 0 for k in keys)
    n01 = sum(ref[k] == 0 and qq[k] == 1 for k in keys)
    return {"clean": clean, "shifts": shifts,
            "e1c": {"pairs": len(keys), "n10": n10, "n01": n01, "a": n10 / n1, "b": n01 / (len(keys) - n1)}}


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate_cell(rates: dict, per_task: int, a: dict, b: dict, rng):
    ref, qq = {}, {}
    for (s, t), p in rates.items():
        for e in range(per_task):
            y = int(rng.random() < p)
            yq = int(rng.random() >= a[(s, t)]) if y else int(rng.random() < b[(s, t)])
            ref[(s, t, e)], qq[(s, t, e)] = y, yq
    return ref, qq


def write_cell(path: Path, outcomes: dict):
    lines = [json.dumps({"kind": "header", "checkpoint": "sim", "preset": "x", "scope": "x", "seed": 0})]
    lines += [json.dumps({"kind": "episode", "suite": s, "task_id": t, "episode": e, "success": bool(y),
                          "steps": 100, "extra": {}}) for (s, t, e), y in outcomes.items()]
    path.write_text("\n".join(lines) + "\n")
    return path.name


def one_run(inp: dict, shift_n: int, scenario: str, B: int, rng, tmp: Path) -> tuple[float, float]:
    a0, b0 = inp["e1c"]["a"], inp["e1c"]["b"]
    if scenario == "2x":
        a0, b0 = 2 * a0, 2 * b0
    tasks = list(inp["clean"])
    if scenario == "uneven":
        a = {k: min(a0 * rng.exponential(1.0), 0.9) for k in tasks}
    else:
        a = {k: a0 for k in tasks}
    b = {k: b0 for k in tasks}

    conditions = {"clean": {"kind": "clean"}, **{s: {"kind": "visual"} for s in inp["shifts"]}}
    cells = {}
    ref, qq = simulate_cell(inp["clean"], 20, a, b, rng)
    cells["clean"] = {"BF16": write_cell(tmp / "c-r.jsonl", ref), "VE-W4": write_cell(tmp / "c-q.jsonl", qq)}
    for name, sh in inp["shifts"].items():
        ref, qq = simulate_cell(sh["task_rates"], shift_n // 30, a, b, rng)
        cells[name] = {"BF16": write_cell(tmp / f"{name}-r.jsonl", ref),
                       "VE-W4": write_cell(tmp / f"{name}-q.jsonl", qq)}
    manifest = {"baseline": "BF16", "primary": "VE-W4", "conditions": conditions, "cells": cells}
    r = q.analyse(manifest, tmp, B=B, seed=int(rng.integers(1 << 31)), threshold=5.0,
                  log=lambda *x: None, within="fixed")
    pv = r["qrp"]["VE-W4"]["pooled_visual"]
    return pv["estimate"], pv["se"]


def power_at(effect: float, se: float) -> float:
    z = effect / se - 1.959964
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e1", type=Path, required=True, help="folder holding E1b/ and E1c/")
    ap.add_argument("--vqb", type=Path, required=True, help="VLAQuantBench clone (released BF16 logs)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 450], help="shifted episodes per shift")
    ap.add_argument("--scenarios", nargs="+", default=["E1c", "2x", "uneven"])
    ap.add_argument("--sims", type=int, default=100, help="simulated studies per size and scenario")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap replicates per simulated study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", type=Path)
    a = ap.parse_args(argv)
    for n in a.sizes:
        if n % 30:
            raise SystemExit(f"--sizes: {n} is not divisible by 30 tasks")

    inp = pilot_inputs(a.e1, a.vqb)
    rng = np.random.default_rng(a.seed)
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for scen in a.scenarios:
            for n in a.sizes:
                est, se = zip(*(one_run(inp, n, scen, a.B, rng, Path(d)) for _ in range(a.sims)))
                se_boot, sd_mc = float(np.mean(se)), float(np.std(est, ddof=1))
                rows.append((scen, n, se_boot, sd_mc, 2.8 * se_boot, power_at(5.0, se_boot)))
                print(f"{scen:7s} {n}/shift: bootstrap SE {se_boot:.2f}, spread over simulations {sd_mc:.2f}",
                      file=sys.stderr)

    e = inp["e1c"]
    md = ["# E1 power update (proposal 7.4)", "",
          "## Inputs from the pilot", "",
          "| Shift | BF16 per suite (Spatial / Object / Goal) | Episodes |", "|---|---|---|"]
    for name, sh in inp["shifts"].items():
        r = sh["suite_rates"]
        md.append(f"| {name} | {' / '.join(f'{100 * r[s]:.0f}%' for s in SUITES)} | {sh['n']} |")
    md += ["", f"E1c lighting, VE-W4 vs BF16: {e['pairs']} matched pairs, n10 = {e['n10']}, n01 = {e['n01']} "
           f"(discordance {100 * (e['n10'] + e['n01']) / e['pairs']:.1f}%); a = {100 * e['a']:.1f}%, "
           f"b = {100 * e['b']:.1f}%. Clean uses the same rates (no environment-B clean VE-W4 data).", "",
           f"## Projected precision of the pooled VE-W4 QRP ({a.sims} simulated studies each, "
           f"tasks-only bootstrap, B = {a.B})", "",
           "| Scenario | Shifted per shift | Bootstrap SE (pp) | Spread over simulations (pp) | MDE at 80% power (pp) | Power at 5 pp |",
           "|---|---|---|---|---|---|"]
    for scen, n, se, sd, mde, pw in rows:
        md.append(f"| {scen} | {n} | {se:.2f} | {sd:.2f} | {mde:.1f} | {100 * pw:.0f}% |")
    md += ["", "Bootstrap SE: the SE our analysis would report. Spread over simulations: how much the estimate "
           "really varies between repeated studies; the two should be close. MDE = 2.8 x SE (proposal Eq. 8). "
           "Rule (proposal 7.4): raise the shifted sample size if the projected SE is above ~2.5 pp."]
    text = "\n".join(md) + "\n"
    print(text)
    if a.out_md:
        a.out_md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
