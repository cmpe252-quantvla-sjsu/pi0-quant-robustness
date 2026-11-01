"""Tests for quantvla/libero_plus.py, part 1 (file lookups; no simulator needed).

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
