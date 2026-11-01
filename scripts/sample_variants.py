#!/usr/bin/env python
"""LIBERO-Plus variant counts and sampling (implementation note 7, EXPERIMENT_PLAN E1/E3/E4).

Part 1 (this file so far): ``count``. For each suite, shift and difficulty level it reports
how many LIBERO-Plus variants exist and how they spread over the 10 base tasks, so we can
see which levels are usable before the E1 pilot.

    python scripts/sample_variants.py count --liberoplus ../LIBERO-plus --vqb ../VLAQuantBench

Base task IDs (0-9) are read from VLAQuantBench's released BASELINE logs, so they are the
same IDs our E0 runs used. Every LIBERO-Plus variant is mapped back to its base task by name.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal")

# LIBERO-Plus category name -> our condition name (docs/JSONL_SCHEMA.md)
CONDITIONS = {
    "Light Conditions": "lighting",
    "Background Textures": "background",
    "Objects Layout": "layout",
    "Language Instructions": "language",
}


def base_tasks(vqb: Path, suite: str) -> dict[str, int]:
    """Base task name -> task_id (0-9), from VLAQuantBench's released BASELINE log."""
    log = vqb / "results" / "libero" / suite / "pi0" / "rtn-BASELINE-e2e.jsonl"
    names = {}
    for line in log.open():
        d = json.loads(line)
        if d.get("kind") == "episode":
            names[d["task_name"]] = int(d["task_id"])
    if sorted(names.values()) != list(range(10)):
        raise SystemExit(f"{log}: expected task_ids 0-9, got {sorted(names.values())}")
    return names


def base_of(variant: str, bases: dict[str, int]) -> str:
    """The base task a variant belongs to: the longest base name followed by '_'."""
    hits = [b for b in bases if variant.startswith(b + "_")]
    if not hits:
        raise SystemExit(f"variant {variant!r} matches no base task")
    return max(hits, key=len)


def load_variants(liberoplus: Path, vqb: Path) -> list[dict]:
    """All variants of our four shift types, with condition, level and base task attached."""
    path = liberoplus / "libero" / "libero" / "benchmark" / "task_classification.json"
    table = json.loads(path.read_text())
    out = []
    for suite in SUITES:
        bases = base_tasks(vqb, suite)
        for v in table[suite]:
            cond = CONDITIONS.get(v["category"])
            if cond is None:  # camera, robot initial state, sensor noise: not in our study
                continue
            base = base_of(v["name"], bases)
            out.append({
                "suite": suite,
                "task_id": bases[base],
                "condition": cond,
                "level": v["difficulty_level"],  # may be None (implementation note 6)
                "variant_id": v["name"],
                "plus_id": v["id"],
            })
    return out


def count(variants: list[dict]) -> str:
    """Markdown table: per suite x condition x level, variants in total and per task."""
    groups = defaultdict(list)
    for v in variants:
        groups[(v["suite"], v["condition"], v["level"])].append(v["task_id"])
    lines = ["| suite | condition | level | variants | tasks covered | min per task | max per task |",
             "|---|---|---|---|---|---|---|"]
    order = list(CONDITIONS.values())
    for (suite, cond, level), tids in sorted(groups.items(),
                                             key=lambda kv: (SUITES.index(kv[0][0]), order.index(kv[0][1]),
                                                             99 if kv[0][2] is None else kv[0][2])):
        per = Counter(tids)
        full = [per.get(t, 0) for t in range(10)]
        lines.append(f"| {suite.removeprefix('libero_')} | {cond} | {'null' if level is None else level} | "
                     f"{len(tids)} | {sum(1 for c in full if c)}/10 | {min(full)} | {max(full)} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count", help="variant counts per suite, condition and level")
    c.add_argument("--liberoplus", type=Path, required=True, help="LIBERO-Plus clone (commit 4976dc3)")
    c.add_argument("--vqb", type=Path, required=True, help="VLAQuantBench clone (for the base task IDs)")
    a = ap.parse_args(argv)
    if a.cmd == "count":
        print(count(load_variants(a.liberoplus, a.vqb)))


if __name__ == "__main__":
    main()
