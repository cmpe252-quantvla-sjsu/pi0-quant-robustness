"""LIBERO-Plus support for VLAQuantBench (our benchmark ``libero_plus``).

Part 1: the file lookups the runner needs, as small functions that can be tested on a laptop
without a simulator. Part 2: ``LiberoPlusRunner``, which VLAQuantBench runs like its own
LIBERO runner (registered by ``quantvla/cli.py``; run with ``python -m quantvla.cli run ...``).

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
from collections import Counter
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


# --------------------------------------------------------------------------- #
# Part 2: the runner
# --------------------------------------------------------------------------- #
import dataclasses  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from vlaquantbench.benchmarks.base import EpisodeOutcome  # noqa: E402
from vlaquantbench.benchmarks.libero import MAX_STEPS, LiberoRunner  # noqa: E402
from vlaquantbench.models.base import TaskSpec  # noqa: E402


def as_init_state_list(states: Any, single: bool):
    """LIBERO-Plus stores a single-state layout variant as one flat vector; make it a list of one."""
    if single:
        return np.asarray(states).reshape(1, -1)
    return states


class LiberoPlusRunner(LiberoRunner):
    """VLAQuantBench runner for clean LIBERO and LIBERO-Plus variants, in environment B.

    One run = one suite and one condition. ``tasks()`` always returns the 10 base tasks with
    VLAQuantBench's task_ids, so records match across conditions and settings by
    (suite, task_id, variant_id, episode). For a shifted condition each episode's variant,
    level and initial state come from a sample file written by ``scripts/sample_variants.py``.

        python -m quantvla.cli run --model pi0 --benchmark libero_plus --suite libero_spatial \\
            --preset BASELINE --bench-kwargs condition=lighting sample_file=configs/samples/E3-lighting-L3.json \\
            --out results/E3/lighting/E3-BF16-spatial.jsonl
    """

    name = "libero_plus"

    def __init__(self, suite: str, *, condition: str = "clean", sample_file: str | None = None,
                 vqb_root: str | None = None, **kwargs: Any):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        requested = kwargs.get("episodes_per_task")
        super().__init__(suite, **kwargs)
        self.condition = condition
        self._base = base_task_names(suite, Path(vqb_root) if vqb_root else None)
        self._sample: dict[tuple[int, int], dict] | None = None
        if condition == "clean":
            if sample_file:
                raise ValueError("clean runs take no sample_file")
        else:
            if not sample_file:
                raise ValueError(f"a {condition} run needs sample_file=<path from scripts/sample_variants.py>")
            self._sample = load_sample(Path(sample_file), suite, condition)
            per_task = Counter(t for t, _ in self._sample)
            sizes = set(per_task.values())
            if sorted(per_task) != list(range(10)) or len(sizes) != 1:
                raise ValueError(f"{sample_file}: every task 0-9 needs the same number of episodes in {suite}")
            n = sizes.pop()
            if requested not in (None, n):
                raise ValueError(f"--episodes {requested} disagrees with the sample file ({n} per task); leave it out")
            self.episodes_per_task = n
        self.sample_file = sample_file
        self._env_bddl: str | None = None
        self._env = None
        self._states_cache: dict[str, Any] = {}

    @classmethod
    def suites(cls) -> tuple[str, ...]:
        return SUITES

    # ------------------------------------------------------------------ #
    def _libero_path(self, key: str) -> Path:
        """LIBERO folders from the active LIBERO config (environment B: the LIBERO-Plus tree)."""
        from libero.libero import get_libero_path  # type: ignore

        return Path(get_libero_path(key))

    def tasks(self) -> list[TaskSpec]:
        steps = self.max_steps(self._adapter_max_steps.get(self.suite, MAX_STEPS[self.suite]))
        return [
            TaskSpec(benchmark=self.name, suite=self.suite, task_id=i, task_name=name,
                     instruction=standard_instruction(name), max_steps=steps,
                     extra={"bddl_file": f"{name}.bddl", "init_states_rel": f"{self.suite}/{name}.pruned_init",
                            "single_init_state": False})
            for i, name in enumerate(self._base)
        ]

    def episodes_for(self, task: TaskSpec) -> list[int]:
        if self._sample is None:
            return super().episodes_for(task)
        return sorted(e for t, e in self._sample if t == task.task_id)

    def episode_task(self, task: TaskSpec, episode: int) -> tuple[TaskSpec, dict]:
        """The TaskSpec for one episode (its own scene and instruction) and the fields to record."""
        if self._sample is None:
            return task, {"condition": "clean", "variant_id": None, "plus_id": None, "difficulty_level": None,
                          "target_level": None, "fallback": False, "instruction": task.instruction}
        rec = self._sample[(task.task_id, episode)]
        variant = rec["variant_id"]
        bddl_file = variant_bddl_file(variant)
        bddl_path = self._libero_path("bddl_files") / self.suite / bddl_file
        instruction = instruction_for(self.condition, task.task_name, bddl_path)
        ep_task = dataclasses.replace(task, instruction=instruction, extra={
            "bddl_file": bddl_file,
            "init_states_rel": variant_init_states_file(self.suite, variant),
            "single_init_state": single_init_state(variant),
            "expected_init_state_id": int(rec["init_state_id"]),
        })
        fields = {"condition": self.condition, "variant_id": variant, "plus_id": rec["plus_id"],
                  "difficulty_level": rec["difficulty_level"], "target_level": rec["target_level"],
                  "fallback": bool(rec["fallback"]), "instruction": instruction}
        return ep_task, fields

    def run_episode(self, adapter, task: TaskSpec, episode: int, meter) -> EpisodeOutcome:
        ep_task, fields = self.episode_task(task, episode)
        out = super().run_episode(adapter, ep_task, episode, meter)
        out.extra = {**(out.extra or {}), **fields}
        expected = ep_task.extra.get("expected_init_state_id")
        if expected is not None and out.extra.get("init_state_id") != expected:
            raise RuntimeError(f"task {task.task_id} episode {episode}: used initial state "
                               f"{out.extra.get('init_state_id')}, sample file says {expected}")
        return out

    # ------------------------------------------------------------------ #
    def _make_env(self, task: TaskSpec):
        bddl = str(self._libero_path("bddl_files") / self.suite / task.extra["bddl_file"])
        if self._env is not None and self._env_bddl == bddl:
            return self._env
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore

        self.close()
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self.camera_size, camera_widths=self.camera_size)
        env.seed(self.seed)
        self._env, self._env_bddl = env, bddl
        return env

    def _init_states(self, task: TaskSpec):
        path = str(self._libero_path("init_states") / task.extra["init_states_rel"])
        if path not in self._states_cache:
            import torch

            if len(self._states_cache) > 32:  # keep memory bounded across many variants
                self._states_cache.clear()
            self._states_cache[path] = as_init_state_list(torch.load(path, weights_only=False),
                                                          task.extra["single_init_state"])
        return self._states_cache[path]

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # pragma: no cover
                pass
        self._env, self._env_bddl = None, None
