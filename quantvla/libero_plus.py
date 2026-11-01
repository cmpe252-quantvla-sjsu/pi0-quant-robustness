"""LIBERO-Plus support for VLAQuantBench (our benchmark ``libero_plus``).

Part 1 (this file so far): the file lookups the runner needs, as small functions that can be
tested on a laptop without a simulator. Part 2 adds ``LiberoPlusRunner`` on top of them.

Why each one exists (docs/IMPLEMENTATION_NOTES.md):
  * LIBERO-Plus's suite lists contain only variants, not the 10 clean tasks (note 1), so
    clean tasks are built from the base task names, in VLAQuantBench's task_id order.
  * A variant's initial states are stored under its base task's file, except single-state
    layout variants (notes 2, 3): ``variant_init_states_file`` mirrors LIBERO-Plus's
    ``Benchmark.get_task_init_states``.
  * Language variants have no BDDL of their own: the ``_view_...`` suffix is dropped.
  * The instruction: LIBERO's own task language is built from the file name, so variant
    names leak the perturbation ("... basket light 1", note 4). Clean and visual-shift
    episodes therefore get the base task's standard instruction, exactly what VLAQuantBench
    gives π0 on clean LIBERO; language episodes get the rewritten one from the variant's BDDL.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal")
CONDITIONS = ("clean", "lighting", "background", "layout", "language")


# --------------------------------------------------------------------------- #
# Base tasks
# --------------------------------------------------------------------------- #
def default_vqb_root() -> Path:
    """The VLAQuantBench clone this Python environment uses (installed with pip install -e)."""
    import vlaquantbench

    return Path(vlaquantbench.__file__).resolve().parents[1]


def base_task_names(suite: str, vqb_root: Path | None = None) -> list[str]:
    """The 10 base task names of a suite, indexed by VLAQuantBench's task_id (0-9).

    Read from VLAQuantBench's released BASELINE log, so the IDs are the ones E0 used.
    """
    root = vqb_root or default_vqb_root()
    log = root / "results" / "libero" / suite / "pi0" / "rtn-BASELINE-e2e.jsonl"
    names: dict[int, str] = {}
    for line in log.open():
        d = json.loads(line)
        if d.get("kind") == "episode":
            names[int(d["task_id"])] = d["task_name"]
    if sorted(names) != list(range(10)):
        raise ValueError(f"{log}: expected task_ids 0-9, got {sorted(names)}")
    return [names[i] for i in range(10)]


def standard_instruction(base_name: str) -> str:
    """LIBERO's task language for a clean task: the file name with spaces (as VLAQuantBench uses)."""
    return " ".join(base_name.split("_"))


# --------------------------------------------------------------------------- #
# Variant files
# --------------------------------------------------------------------------- #
def variant_bddl_file(variant_id: str) -> str:
    """BDDL file name of a variant. Language variants drop their ``_view_...`` suffix."""
    return variant_id.split("_view_")[0] + ".bddl"


def single_init_state(variant_id: str) -> bool:
    """True if LIBERO-Plus stores one initial state for this variant (same rule as the sampler)."""
    return "_language_" not in variant_id and ("_add_" in variant_id or "_level" in variant_id)


def variant_init_states_file(suite: str, variant_id: str) -> str:
    """Init-states file of a variant, relative to LIBERO's ``init_states`` folder.

    Mirrors LIBERO-Plus ``Benchmark.get_task_init_states`` at commit 4976dc3.
    """
    if "_language_" in variant_id:
        return f"{suite}/{variant_id.split('_language_')[0]}.pruned_init"
    if single_init_state(variant_id):
        return f"libero_newobj/{suite}/{variant_id}.pruned_init"
    if "_view_" in variant_id:
        base = variant_id.split("_view_")[0]
    elif "_light_" in variant_id:
        base = variant_id.split("_light_")[0]
    else:
        base = re.sub(r"_(table|tb)_\d+", "", variant_id)
    return f"{suite}/{base}.pruned_init"


def bddl_language(path: Path) -> str:
    """The ``(:language ...)`` line of a BDDL file."""
    for line in Path(path).read_text().splitlines():
        if "(:language" in line:
            return line.split("(:language", 1)[1].strip().rstrip(")").strip()
    raise ValueError(f"{path}: no (:language ...) line")


def instruction_for(condition: str, base_name: str, bddl_path: Path | None = None) -> str:
    """What π0 is told: the standard base instruction, except for language rewrites."""
    if condition == "language":
        if bddl_path is None:
            raise ValueError("a language episode needs its variant's BDDL file")
        return bddl_language(bddl_path)
    return standard_instruction(base_name)


# --------------------------------------------------------------------------- #
# Sample files (written by scripts/sample_variants.py)
# --------------------------------------------------------------------------- #
def load_sample(path: Path, suite: str, condition: str) -> dict[tuple[int, int], dict]:
    """Episodes of one suite from a sample file, keyed by (task_id, episode)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("condition") != condition:
        raise ValueError(f"{path} is a {data.get('condition')!r} sample, not {condition!r}")
    if suite not in data.get("suites", []):
        raise ValueError(f"{path} has no episodes for {suite}")
    out = {}
    for e in data["episodes"]:
        if e["suite"] != suite:
            continue
        key = (int(e["task_id"]), int(e["episode"]))
        if key in out:
            raise ValueError(f"{path}: duplicate episode {key} in {suite}")
        out[key] = e
    return out
