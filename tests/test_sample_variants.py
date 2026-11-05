"""Tests for scripts/sample_variants.py (variant counts and sampling).

Uses a tiny fake LIBERO-Plus and fake VLAQuantBench logs, so no download is needed.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sample_variants as sv  # noqa: E402

SUITES = sv.SUITES


def base_name(suite, t):
    return f"{suite}_task{t}"


def make_fake(tmp_path, variants):
    """variants: list of (suite, task, category, level, suffix). Writes both fake trees."""
    vqb = tmp_path / "vqb"
    for suite in SUITES:
        log = vqb / "results" / "libero" / suite / "pi0" / "rtn-BASELINE-e2e.jsonl"
        log.parent.mkdir(parents=True)
        lines = [json.dumps({"kind": "header"})]
        lines += [json.dumps({"kind": "episode", "task_id": t, "task_name": base_name(suite, t)}) for t in range(10)]
        log.write_text("\n".join(lines) + "\n")
    table = {s: [] for s in SUITES}
    # ids depend only on the variant's name, like LIBERO-Plus's fixed ids, not on list order
    ids = {n: i for i, n in enumerate(sorted(f"{base_name(s, t)}_{x}" for s, t, _, _, x in variants), start=1)}
    for suite, t, cat, level, suffix in variants:
        name = f"{base_name(suite, t)}_{suffix}"
        table[suite].append({"id": ids[name], "name": name, "category": cat, "difficulty_level": level})
    lp = tmp_path / "lp"
    path = lp / "libero" / "libero" / "benchmark" / "task_classification.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(table))
    return lp, vqb


def full_coverage(category="Light Conditions", level=2, per_task=3, prefix="light"):
    """per_task variants at `level` for every suite and task."""
    out = []
    for s in SUITES:
        for t in range(10):
            out += [(s, t, category, level, f"{prefix}_{level}{k}") for k in range(per_task)]
    return out


# --------------------------------------------------------------------------- #
def test_base_of_takes_the_longest_matching_name():
    bases = {"pick_up_the_bowl": 0, "pick_up_the_bowl_on_the_plate": 1}
    assert sv.base_of("pick_up_the_bowl_on_the_plate_light_3", bases) == "pick_up_the_bowl_on_the_plate"
    assert sv.base_of("pick_up_the_bowl_light_3", bases) == "pick_up_the_bowl"
    with pytest.raises(SystemExit):
        sv.base_of("something_else_light_1", bases)


def test_pick_level_is_nearest_and_lower_on_a_tie():
    assert sv.pick_level([1, 2, 5], 2) == 2
    assert sv.pick_level([1, 3, 5], 2) == 1       # 1 and 3 are equally near: take the lower
    assert sv.pick_level([4, 5], 2) == 4
    assert sv.pick_level([1], 5) == 1


def test_single_init_state_rule_matches_liberoplus():
    assert sv.single_init_state("task_add_3")
    assert sv.single_init_state("task_level2_sample1")
    assert not sv.single_init_state("task_light_3")
    assert not sv.single_init_state("task_table_12")
    assert not sv.single_init_state("task_language_2_view_0_0_100_0_0_initstate_0")


def test_parse_levels():
    assert sv.parse_levels("3") == {s: 3 for s in SUITES}
    assert sv.parse_levels("spatial=2,object=4,goal=1") == {
        "libero_spatial": 2, "libero_object": 4, "libero_goal": 1}
    with pytest.raises(SystemExit):
        sv.parse_levels("spatial=2,object=4")


def test_episodes_are_numbered_per_task_and_keep_the_task_mix(tmp_path):
    lp, vqb = make_fake(tmp_path, full_coverage())
    r = sv.sample(sv.load_variants(lp, vqb), "lighting", 2, per_task=5, seed=0)
    assert r["n_episodes"] == 3 * 10 * 5
    for s in SUITES:
        for t in range(10):
            eps = [e for e in r["episodes"] if (e["suite"], e["task_id"]) == (s, t)]
            assert sorted(e["episode"] for e in eps) == list(range(5))
            assert not any(e["fallback"] for e in eps)


def test_distinct_variants_first_then_cycle(tmp_path):
    lp, vqb = make_fake(tmp_path, full_coverage(per_task=3))
    r = sv.sample(sv.load_variants(lp, vqb), "lighting", 2, per_task=7, seed=0)
    eps = [e for e in r["episodes"] if (e["suite"], e["task_id"]) == ("libero_spatial", 0)]
    by_ep = [e["variant_id"] for e in sorted(eps, key=lambda e: e["episode"])]
    assert len(set(by_ep[:3])) == 3                # the first 3 episodes use 3 different variants
    assert by_ep[3:6] == by_ep[:3]                 # then the same order again
    assert r["summary"]["libero_spatial"]["min_distinct_variants_per_task"] == 3


def test_fallback_is_recorded_and_null_levels_are_never_used(tmp_path):
    v = full_coverage(level=2)
    v = [x for x in v if not (x[0] == "libero_goal" and x[1] == 3)]          # goal task 3: no level-2 variant
    v += [("libero_goal", 3, "Light Conditions", 1, "light_1a"),
          ("libero_goal", 3, "Light Conditions", 3, "light_3a"),                # 1 and 3 tie: take 1
          ("libero_goal", 3, "Light Conditions", None, "light_99")]           # null: never used
    lp, vqb = make_fake(tmp_path, v)
    r = sv.sample(sv.load_variants(lp, vqb), "lighting", 2, per_task=4, seed=0)
    eps = [e for e in r["episodes"] if (e["suite"], e["task_id"]) == ("libero_goal", 3)]
    assert all(e["fallback"] and e["target_level"] == 2 and e["difficulty_level"] == 1 for e in eps)
    assert all(e["variant_id"].endswith("light_1a") for e in eps)
    assert r["summary"]["libero_goal"]["fallback_tasks"] == [3]
    assert not any(e["variant_id"].endswith("light_99") for e in r["episodes"])


def test_more_than_two_fallback_tasks_stops_unless_allowed(tmp_path):
    v = [x for x in full_coverage(level=2) if not (x[0] == "libero_object" and x[1] < 3)]
    v += [("libero_object", t, "Light Conditions", 4, "light_4a") for t in range(3)]
    lp, vqb = make_fake(tmp_path, v)
    variants = sv.load_variants(lp, vqb)
    with pytest.raises(SystemExit, match="too many fallback tasks"):
        sv.sample(variants, "lighting", 2, per_task=2, seed=0)
    r = sv.sample(variants, "lighting", 2, per_task=2, seed=0, allow_over_cap=True)
    assert r["over_cap"] and "libero_object" in r["over_cap"][0]


def test_init_state_follows_the_episode_except_single_state_layouts(tmp_path):
    v = full_coverage("Objects Layout", level=3, per_task=2, prefix="add")
    v += [(s, t, "Light Conditions", 2, f"light_2{k}") for s in SUITES for t in range(10) for k in range(2)]
    lp, vqb = make_fake(tmp_path, v)
    variants = sv.load_variants(lp, vqb)
    lay = sv.sample(variants, "layout", 3, per_task=4, seed=0)
    assert all(e["init_state_id"] == 0 for e in lay["episodes"])
    light = sv.sample(variants, "lighting", 2, per_task=4, seed=0)
    assert all(e["init_state_id"] == e["episode"] for e in light["episodes"])


def test_same_arguments_give_the_same_file_in_any_json_order(tmp_path):
    v = full_coverage(per_task=4)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    ra = sv.sample(sv.load_variants(*make_fake(a, v)), "lighting", 2, per_task=6, seed=0)
    rb = sv.sample(sv.load_variants(*make_fake(b, list(reversed(v)))), "lighting", 2, per_task=6, seed=0)
    strip = lambda r: [(e["suite"], e["task_id"], e["episode"], e["variant_id"]) for e in r["episodes"]]
    assert strip(ra) == strip(rb)
    rc = sv.sample(sv.load_variants(*make_fake(tmp_path / "c", v)), "lighting", 2, per_task=6, seed=1)
    assert strip(rc) != strip(ra)                 # a different sampler seed gives a different draw


def test_count_reports_coverage(tmp_path):
    v = [x for x in full_coverage(level=2) if not (x[0] == "libero_spatial" and x[1] == 0)]
    lp, vqb = make_fake(tmp_path, v)
    table = sv.count(sv.load_variants(lp, vqb))
    assert "| spatial | lighting | 2 | 27 | 9/10 | 0 | 3 |" in table
    assert "| object | lighting | 2 | 30 | 10/10 | 3 | 3 |" in table
