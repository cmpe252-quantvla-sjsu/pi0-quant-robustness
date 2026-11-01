#!/usr/bin/env python
"""LIBERO-Plus variant counts and sampling (implementation note 7, EXPERIMENT_PLAN E1/E3/E4).

``count``  For each suite, shift and difficulty level: how many LIBERO-Plus variants exist
           and how they spread over the 10 base tasks.

    python scripts/sample_variants.py count --liberoplus ../LIBERO-plus --vqb ../VLAQuantBench

``sample`` Picks the episodes for one shift at one target level and writes a sample file
           that the LIBERO-Plus runner reads. Same arguments -> same file, every time.

    python scripts/sample_variants.py sample --liberoplus ../LIBERO-plus --vqb ../VLAQuantBench \\
        --condition lighting --level 3 --episodes-per-task 10 --out configs/samples/E3-lighting-L3.json

    --level takes one level for every suite, or one per suite: --level spatial=2,object=4,goal=2

Rules (team decision, Nov 1):
  * Every suite keeps the clean task mix: the same number of episodes for each of the 10 tasks.
  * A task with no variant at the target level takes the nearest level that has one; on a tie,
    the lower level (option 1). Each episode records ``target_level``, ``difficulty_level``
    and ``fallback``. Variants with no level (``null``) are never used.
  * If more than ``--max-fallback-tasks`` (default 2) of a suite's 10 tasks need a fallback,
    the command stops: choose another level, or pass ``--allow-over-cap`` to flag it.
  * Within a task, episodes use distinct variants while there are enough, then cycle through
    them again. Episodes are numbered 0..n-1 per task (unique within the task, as
    docs/JSONL_SCHEMA.md asks), so episode e gets the same noise seed and, where the variant
    has the base task's initial states, the same initial state as clean episode e.
  * Layout variants named ``_add_`` / ``_level`` have a single stored initial state
    (implementation note 3): ``init_state_id`` is 0 and only the seed varies.

Base task IDs (0-9) are read from VLAQuantBench's released BASELINE logs, so they are the
same IDs our E0 runs used. Every LIBERO-Plus variant is mapped back to its base task by name.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

LIBEROPLUS_COMMIT = "4976dc3"
N_BASE_INIT_STATES = 50   # stored initial states per base LIBERO task

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


def single_init_state(variant_id: str) -> bool:
    """True if LIBERO-Plus stores only one initial state for this variant.

    Mirrors ``Benchmark.get_task_init_states`` in LIBERO-Plus: language variants use the base
    task's states; otherwise ``_add_`` / ``_level`` variants load a single state.
    """
    return "_language_" not in variant_id and ("_add_" in variant_id or "_level" in variant_id)


def pick_level(levels: list[int], target: int) -> int:
    """Nearest available level to ``target``; on a tie, the lower one."""
    return min(levels, key=lambda lv: (abs(lv - target), lv))


def parse_levels(text: str, suites=SUITES) -> dict[str, int]:
    """'3' -> the same target level for every suite; 'spatial=2,object=4,goal=2' -> one per suite."""
    if "=" not in text:
        return {s: int(text) for s in suites}
    out = {}
    for part in text.split(","):
        name, lv = part.split("=")
        suite = name.strip() if name.strip().startswith("libero_") else f"libero_{name.strip()}"
        out[suite] = int(lv)
    missing = [s for s in suites if s not in out]
    if missing:
        raise SystemExit(f"--level gives no level for {missing}")
    return out


def sample(variants: list[dict], condition: str, target, per_task: int, seed: int,
           suites=SUITES, max_fallback_tasks: int = 2, allow_over_cap: bool = False) -> dict:
    """Choose ``per_task`` episodes for every (suite, task) of one condition.

    ``target`` is one level for all suites (int) or a {suite: level} dict.
    """
    targets = target if isinstance(target, dict) else {s: target for s in suites}
    pool = defaultdict(lambda: defaultdict(list))   # (suite, task) -> level -> variants
    for v in variants:
        if v["condition"] == condition and v["level"] is not None:
            pool[(v["suite"], v["task_id"])][v["level"]].append(v)

    episodes, summary, problems = [], {}, []
    for suite in suites:
        fallback_tasks, levels_used, distinct = [], Counter(), {}
        for task in range(10):
            by_level = pool.get((suite, task), {})
            if not by_level:
                raise SystemExit(f"{suite} task {task}: no {condition} variant at any level")
            level = pick_level(sorted(by_level), targets[suite])
            fallback = level != targets[suite]
            if fallback:
                fallback_tasks.append(task)
            # sort first so the result never depends on the JSON file's order
            chosen = sorted(by_level[level], key=lambda v: v["plus_id"])
            random.Random(f"{seed}|{condition}|{suite}|{task}").shuffle(chosen)
            distinct[task] = min(per_task, len(chosen))
            for e in range(per_task):
                v = chosen[e % len(chosen)]          # distinct while possible, then cycle
                single = single_init_state(v["variant_id"])
                episodes.append({
                    "suite": suite, "task_id": task, "episode": e,
                    "condition": condition, "variant_id": v["variant_id"], "plus_id": v["plus_id"],
                    "difficulty_level": level, "target_level": targets[suite], "fallback": fallback,
                    "init_state_id": 0 if single else e % N_BASE_INIT_STATES,
                })
                levels_used[level] += 1
        summary[suite] = {"target_level": targets[suite], "fallback_tasks": fallback_tasks, "levels_used": dict(sorted(levels_used.items())),
                          "min_distinct_variants_per_task": min(distinct.values())}
        if len(fallback_tasks) > max_fallback_tasks:
            problems.append(f"{suite}: {len(fallback_tasks)} of 10 tasks need a fallback level "
                            f"(tasks {fallback_tasks}; cap {max_fallback_tasks})")

    if problems and not allow_over_cap:
        raise SystemExit("too many fallback tasks at this level; choose another level, "
                         "or pass --allow-over-cap to flag it:\n  " + "\n  ".join(problems))
    return {
        "condition": condition, "target_levels": {s: targets[s] for s in suites}, "episodes_per_task": per_task, "seed": seed,
        "suites": list(suites), "liberoplus_commit": LIBEROPLUS_COMMIT,
        "max_fallback_tasks": max_fallback_tasks, "over_cap": problems,
        "summary": summary, "n_episodes": len(episodes), "episodes": episodes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("count", "variant counts per suite, condition and level"),
                        ("sample", "write the sample file for one condition and target level")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--liberoplus", type=Path, required=True, help=f"LIBERO-Plus clone (commit {LIBEROPLUS_COMMIT})")
        p.add_argument("--vqb", type=Path, required=True, help="VLAQuantBench clone (for the base task IDs)")
    s = sub.choices["sample"]
    s.add_argument("--condition", required=True, choices=list(CONDITIONS.values()))
    s.add_argument("--level", required=True,
                   help="target difficulty level for every suite (e.g. 3), or one per suite: spatial=2,object=4,goal=2")
    s.add_argument("--episodes-per-task", type=int, required=True)
    s.add_argument("--seed", type=int, default=0, help="sampler seed (recorded in frozen_design.yaml)")
    s.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    s.add_argument("--max-fallback-tasks", type=int, default=2)
    s.add_argument("--allow-over-cap", action="store_true")
    s.add_argument("--out", type=Path, required=True, help="sample file to write (.json)")
    a = ap.parse_args(argv)
    variants = load_variants(a.liberoplus, a.vqb)
    if a.cmd == "count":
        print(count(variants))
        return
    result = sample(variants, a.condition, parse_levels(a.level, a.suites), a.episodes_per_task, a.seed, a.suites,
                    a.max_fallback_tasks, a.allow_over_cap)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for suite, s_ in result["summary"].items():
        print(f"{suite}: target {s_['target_level']}, levels used {s_['levels_used']}, "
              f"fallback tasks {s_['fallback_tasks']}, min distinct variants per task {s_['min_distinct_variants_per_task']}")
    for p in result["over_cap"]:
        print(f"FLAG: {p}", file=sys.stderr)
    print(f"wrote {result['n_episodes']} episodes to {a.out}")


if __name__ == "__main__":
    main()
