"""Tests for quantvla/libero_plus.py: file lookups and the runner (no simulator needed).

The tests marked ``needs_liberoplus`` use a LIBERO-Plus clone at commit 4976dc3. They look
for it in $LIBERO_PLUS_DIR, then next to this repo, then in your home folder, and are
skipped if none is found.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from quantvla import libero_plus as lp  # noqa: E402


def _find_liberoplus():
    for c in (os.environ.get("LIBERO_PLUS_DIR"), REPO.parent / "LIBERO-plus", Path.home() / "LIBERO-plus"):
        if c and (Path(c) / "libero" / "libero" / "benchmark" / "task_classification.json").exists():
            return Path(c) / "libero" / "libero"
    return None


LP = _find_liberoplus()
needs_liberoplus = pytest.mark.skipif(LP is None, reason="LIBERO-Plus clone not found")

BOWL = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"


# --------------------------------------------------------------------------- #
def test_standard_instruction_is_the_file_name_with_spaces():
    assert lp.standard_instruction(BOWL) == "pick up the black bowl between the plate and the ramekin and place it on the plate"


@pytest.mark.parametrize("variant,expected", [
    (f"{BOWL}_light_3", f"libero_spatial/{BOWL}.pruned_init"),
    (f"{BOWL}_table_12", f"libero_spatial/{BOWL}.pruned_init"),
    (f"{BOWL}_tb_4", f"libero_spatial/{BOWL}.pruned_init"),
    (f"{BOWL}_add_10", f"libero_newobj/libero_spatial/{BOWL}_add_10.pruned_init"),
    (f"{BOWL}_level2_sample1", f"libero_newobj/libero_spatial/{BOWL}_level2_sample1.pruned_init"),
    (f"{BOWL}_language_2_view_0_0_100_0_0_initstate_0", f"libero_spatial/{BOWL}.pruned_init"),
])
def test_variant_init_states_file(variant, expected):
    assert lp.variant_init_states_file("libero_spatial", variant) == expected


def test_variant_bddl_file_drops_the_view_suffix_of_language_variants():
    assert lp.variant_bddl_file(f"{BOWL}_light_3") == f"{BOWL}_light_3.bddl"
    assert lp.variant_bddl_file(f"{BOWL}_language_2_view_0_0_100_0_0_initstate_0") == f"{BOWL}_language_2.bddl"


def test_visual_shifts_get_the_base_instruction_language_gets_the_rewrite(tmp_path):
    bddl = tmp_path / "x.bddl"
    bddl.write_text("(define (problem x)\n  (:language grasp the dark bowl and set it on the dish)\n)\n")
    for cond in ("clean", "lighting", "background", "layout"):
        assert lp.instruction_for(cond, BOWL, bddl) == lp.standard_instruction(BOWL)
    assert lp.instruction_for("language", BOWL, bddl) == "grasp the dark bowl and set it on the dish"
    with pytest.raises(ValueError):
        lp.instruction_for("language", BOWL)


def test_load_sample_checks_condition_suite_and_duplicates(tmp_path):
    eps = [{"suite": "libero_spatial", "task_id": 0, "episode": e, "variant_id": f"{BOWL}_light_1"} for e in range(3)]
    eps += [{"suite": "libero_goal", "task_id": 0, "episode": 0, "variant_id": "x"}]
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"condition": "lighting", "suites": ["libero_spatial", "libero_goal"], "episodes": eps}))
    got = lp.load_sample(p, "libero_spatial", "lighting")
    assert sorted(got) == [(0, 0), (0, 1), (0, 2)]
    with pytest.raises(ValueError, match="not 'layout'"):
        lp.load_sample(p, "libero_spatial", "layout")
    with pytest.raises(ValueError, match="no episodes"):
        lp.load_sample(p, "libero_object", "lighting")
    p.write_text(json.dumps({"condition": "lighting", "suites": ["libero_spatial"], "episodes": eps[:1] * 2}))
    with pytest.raises(ValueError, match="duplicate"):
        lp.load_sample(p, "libero_spatial", "lighting")


def test_base_task_names_match_vlaquantbench_ids():
    names = lp.base_task_names("libero_spatial")
    assert len(names) == 10 and names[0] == BOWL


# --------------------------------------------------------------------------- #
@needs_liberoplus
@pytest.mark.parametrize("suite", lp.SUITES)
def test_clean_base_tasks_exist_in_liberoplus(suite):
    for name in lp.base_task_names(suite):
        assert (LP / "bddl_files" / suite / f"{name}.bddl").exists(), name
        assert (LP / "init_files" / suite / f"{name}.pruned_init").exists(), name


@needs_liberoplus
def test_every_variant_of_our_four_shifts_resolves_to_existing_files():
    table = json.loads((LP / "benchmark" / "task_classification.json").read_text())
    ours = {"Light Conditions", "Background Textures", "Objects Layout", "Language Instructions"}
    n = 0
    for suite in lp.SUITES:
        for v in table[suite]:
            if v["category"] not in ours:
                continue
            n += 1
            assert (LP / "bddl_files" / suite / lp.variant_bddl_file(v["name"])).exists(), v["name"]
            assert (LP / "init_files" / lp.variant_init_states_file(suite, v["name"])).exists(), v["name"]
    assert n == 4022


@needs_liberoplus
def test_language_variants_change_only_the_instruction():
    rewrite = LP / "bddl_files" / "libero_spatial" / f"{BOWL}_language_1.bddl"
    base = LP / "bddl_files" / "libero_spatial" / f"{BOWL}.bddl"
    strip = lambda p: [l for l in p.read_text().splitlines() if "(:language" not in l]
    assert strip(rewrite) == strip(base)
    assert lp.bddl_language(rewrite) != lp.bddl_language(base)


# --------------------------------------------------------------------------- #
# Part 2: the runner (no simulator: env and rollout are stubbed or not touched)
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402

from vlaquantbench.benchmarks.base import EpisodeOutcome  # noqa: E402
from vlaquantbench.benchmarks.libero import LiberoRunner  # noqa: E402


class _Runner(lp.LiberoPlusRunner):
    """Points LIBERO's folders at the LIBERO-Plus clone instead of the installed config."""

    def _libero_path(self, key):
        return LP / {"bddl_files": "bddl_files", "init_states": "init_files"}[key]


def _sample_file(tmp_path, condition, category, per_task=2, suite="libero_spatial"):
    """A sample file using the first variants of `category` for each base task."""
    table = json.loads((LP / "benchmark" / "task_classification.json").read_text())
    bases = lp.base_task_names(suite)
    eps = []
    for t, base in enumerate(bases):
        vs = [v for v in table[suite] if v["category"] == category and v["difficulty_level"] is not None
              and max((b for b in bases if v["name"].startswith(b + "_")), key=len, default=None) == base]
        for e in range(per_task):
            v = vs[e % len(vs)]
            eps.append({"suite": suite, "task_id": t, "episode": e, "condition": condition,
                        "variant_id": v["name"], "plus_id": v["id"], "difficulty_level": v["difficulty_level"],
                        "target_level": v["difficulty_level"], "fallback": False,
                        "init_state_id": 0 if lp.single_init_state(v["name"]) else e % 50})
    p = tmp_path / f"{condition}.json"
    p.write_text(json.dumps({"condition": condition, "suites": [suite], "episodes": eps}))
    return p


def test_clean_runner_has_the_10_base_tasks_with_standard_instructions():
    r = lp.LiberoPlusRunner("libero_spatial", episodes_per_task=20)
    tasks = r.tasks()
    assert [t.task_id for t in tasks] == list(range(10))
    assert tasks[0].task_name == BOWL and tasks[0].instruction == lp.standard_instruction(BOWL)
    assert tasks[0].extra["init_states_rel"] == f"libero_spatial/{BOWL}.pruned_init"
    assert r.episodes_for(tasks[0]) == list(range(20))
    ep_task, fields = r.episode_task(tasks[0], 3)
    assert ep_task is tasks[0] and fields["condition"] == "clean" and fields["variant_id"] is None


def test_runner_argument_checks(tmp_path):
    with pytest.raises(ValueError, match="condition must be"):
        lp.LiberoPlusRunner("libero_spatial", condition="fog")
    with pytest.raises(ValueError, match="needs sample_file"):
        lp.LiberoPlusRunner("libero_spatial", condition="lighting")
    with pytest.raises(ValueError, match="no sample_file"):
        lp.LiberoPlusRunner("libero_spatial", sample_file="x.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"condition": "lighting", "suites": ["libero_spatial"], "episodes": [
        {"suite": "libero_spatial", "task_id": 0, "episode": 0, "variant_id": "x"}]}))
    with pytest.raises(ValueError, match="same number of episodes"):
        lp.LiberoPlusRunner("libero_spatial", condition="lighting", sample_file=str(bad))


@needs_liberoplus
def test_shifted_runner_takes_episodes_and_scenes_from_the_sample(tmp_path):
    p = _sample_file(tmp_path, "lighting", "Light Conditions", per_task=3)
    r = _Runner("libero_spatial", condition="lighting", sample_file=str(p))
    assert r.episodes_per_task == 3
    task = r.tasks()[0]
    assert r.episodes_for(task) == [0, 1, 2]
    ep_task, fields = r.episode_task(task, 1)
    assert ep_task.instruction == lp.standard_instruction(BOWL)          # never the variant's name
    assert ep_task.extra["bddl_file"] == lp.variant_bddl_file(fields["variant_id"])
    assert (LP / "bddl_files" / "libero_spatial" / ep_task.extra["bddl_file"]).exists()
    assert (LP / "init_files" / ep_task.extra["init_states_rel"]).exists()
    assert fields["condition"] == "lighting" and fields["fallback"] is False
    with pytest.raises(ValueError, match="disagrees with the sample"):
        _Runner("libero_spatial", condition="lighting", sample_file=str(p), episodes_per_task=20)


@needs_liberoplus
def test_language_runner_gives_the_rewritten_instruction(tmp_path):
    p = _sample_file(tmp_path, "language", "Language Instructions", per_task=1)
    r = _Runner("libero_spatial", condition="language", sample_file=str(p))
    ep_task, fields = r.episode_task(r.tasks()[0], 0)
    assert ep_task.instruction != lp.standard_instruction(BOWL)
    assert ep_task.instruction == fields["instruction"]
    assert ep_task.extra["init_states_rel"] == f"libero_spatial/{BOWL}.pruned_init"


@needs_liberoplus
def test_layout_runner_uses_the_single_stored_state(tmp_path):
    p = _sample_file(tmp_path, "layout", "Objects Layout", per_task=2)
    r = _Runner("libero_spatial", condition="layout", sample_file=str(p))
    singles = [r.episode_task(t, 0)[0].extra["single_init_state"] for t in r.tasks()]
    assert any(singles)


def test_single_state_files_become_a_list_of_one():
    flat = np.arange(12.0)
    assert lp.as_init_state_list(flat, single=True).shape == (1, 12)
    many = np.zeros((50, 12))
    assert lp.as_init_state_list(many, single=False) is many


@needs_liberoplus
def test_run_episode_adds_the_fields_and_checks_the_initial_state(tmp_path, monkeypatch):
    p = _sample_file(tmp_path, "lighting", "Light Conditions", per_task=3)
    r = _Runner("libero_spatial", condition="lighting", sample_file=str(p))
    seen = {}

    def fake_rollout(self, adapter, task, episode, meter):  # stands in for the simulator loop
        seen["instruction"] = task.instruction
        return EpisodeOutcome(success=True, steps=50, extra={"init_state_id": episode % 50})

    monkeypatch.setattr(LiberoRunner, "run_episode", fake_rollout)
    out = r.run_episode(None, r.tasks()[2], 1, None)
    assert out.extra["condition"] == "lighting" and out.extra["init_state_id"] == 1
    assert out.extra["variant_id"].startswith(r.tasks()[2].task_name + "_")
    assert seen["instruction"] == r.tasks()[2].instruction

    monkeypatch.setattr(LiberoRunner, "run_episode",
                        lambda self, a, t, e, m: EpisodeOutcome(success=True, steps=1, extra={"init_state_id": 7}))
    with pytest.raises(RuntimeError, match="sample file says"):
        r.run_episode(None, r.tasks()[2], 1, None)


def test_cli_registers_libero_plus():
    import quantvla.cli  # noqa: F401
    from vlaquantbench.registry import get_runner_cls

    assert get_runner_cls("libero_plus") is lp.LiberoPlusRunner
