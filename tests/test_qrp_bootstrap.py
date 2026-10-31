"""Tests for scripts/qrp_bootstrap.py (proposal Sections 4.4 and 7.1-7.3)."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qrp_bootstrap as q  # noqa: E402
import vlaquantbench  # noqa: E402

VQB = Path(vlaquantbench.__file__).resolve().parents[1]  # pinned VLAQuantBench clone (editable install)
needs_logs = pytest.mark.skipif(not (VQB / "results/libero/libero_spatial/pi0/rtn-BASELINE-e2e.jsonl").exists(),
                                reason="VLAQuantBench released logs not found")

CKPT = "lerobot/pi0_libero_finetuned_v044"


def write_cell(path: Path, outcomes: dict, preset="BASELINE", scope="e2e", ckpt=CKPT, shuffle=False):
    """outcomes: {(suite, task_id, episode[, variant]): success}"""
    lines = [json.dumps({"kind": "header", "checkpoint": ckpt, "preset": preset, "scope": scope, "seed": 0})]
    eps = []
    for key, ok in outcomes.items():
        suite, task, ep = key[:3]
        extra = {"variant_id": key[3]} if len(key) == 4 else {}
        eps.append(json.dumps({"kind": "episode", "suite": suite, "task_id": task, "episode": ep,
                               "success": bool(ok), "steps": 100 if ok else 280, "extra": extra}))
    if shuffle:
        random.Random(0).shuffle(eps)
    path.write_text("\n".join(lines + eps) + "\n")
    return path.name


def run(tmp_path, cells: dict, conditions: dict, B=2000, seed=0, **manifest_extra):
    manifest = {"baseline": "BF16", "primary": "VE-W4", "conditions": conditions, "cells": {}, **manifest_extra}
    for cond, by_setting in cells.items():
        manifest["cells"][cond] = {}
        for setting, outcomes in by_setting.items():
            manifest["cells"][cond][setting] = write_cell(tmp_path / f"{cond}-{setting}.jsonl", outcomes)
    return q.analyse(manifest, tmp_path, B=B, seed=seed, threshold=5.0, log=lambda *a: None)


def grid(suites=("s1", "s2"), tasks=2, eps=4, fn=lambda s, t, e: 1, variant=None):
    out = {}
    for s in suites:
        for t in range(tasks):
            for e in range(eps):
                key = (s, t, e) if variant is None else (s, t, e, variant(s, t, e))
                out[key] = fn(s, t, e)
    return out


COND = {"clean": {"kind": "clean"}, "light": {"kind": "visual"}}


# --------------------------------------------------------------------------- #
def test_point_estimates_match_hand_computation(tmp_path):
    # 16 clean episodes: BF16 all succeed, VE-W4 fails 2  -> gap 12.5 pp
    clean_ref = grid()
    clean_q = {**clean_ref, ("s1", 0, 0): 0, ("s2", 1, 3): 0}
    # 16 shifted: BF16 fails 4, VE-W4 fails those 4 + 4 more, but succeeds on one BF16 failure
    shift_ref = grid(fn=lambda s, t, e: 0 if e == 0 else 1)
    shift_q = grid(fn=lambda s, t, e: 0 if e in (0, 1) else 1)
    shift_q[("s1", 0, 0)] = 1
    r = run(tmp_path, {"clean": {"BF16": clean_ref, "VE-W4": clean_q},
                       "light": {"BF16": shift_ref, "VE-W4": shift_q}}, COND)
    g_clean, g_light = r["gaps"]["clean/VE-W4"], r["gaps"]["light/VE-W4"]
    assert g_clean["estimate"] == pytest.approx(12.5)
    assert (g_clean["n10"], g_clean["n01"]) == (2, 0)
    # light: SR_BF16 = 12/16, SR_Q = 8/16 + 1/16 = 9/16 -> gap 3/16 = 18.75 pp
    assert g_light["estimate"] == pytest.approx(18.75)
    assert (g_light["n10"], g_light["n01"]) == (4, 1)
    assert g_light["discordance"] == pytest.approx(5 / 16)
    assert r["qrp"]["VE-W4"]["light"]["estimate"] == pytest.approx(6.25)
    assert r["qrp"]["VE-W4"]["pooled_visual"]["estimate"] == pytest.approx(6.25)
    assert r["tests"]["H1"]["estimate"] == pytest.approx(6.25)


def test_identical_conditions_give_exactly_zero_qrp(tmp_path):
    ref = grid(fn=lambda s, t, e: (e + t) % 3 != 0)
    qq = grid(fn=lambda s, t, e: (e + t) % 3 == 1)
    cells = {"clean": {"BF16": ref, "VE-W4": qq}, "light": {"BF16": ref, "VE-W4": qq}}
    # resample: the point estimate is 0, the interval has width (within-task draws differ)
    s = run(tmp_path, cells, COND, bootstrap_within="resample")["qrp"]["VE-W4"]["pooled_visual"]
    assert s["estimate"] == 0.0
    assert s["ci_low"] < 0 < s["ci_high"]
    # fixed (default): identical data in both conditions -> every replicate is exactly 0
    s = run(tmp_path, cells, COND)["qrp"]["VE-W4"]["pooled_visual"]
    assert s["estimate"] == 0.0 and s["ci_low"] == s["ci_high"] == 0.0


def test_record_order_does_not_change_anything(tmp_path):
    ref = grid(eps=6, fn=lambda s, t, e: (e * 7 + t) % 4 != 0)
    qq = grid(eps=6, fn=lambda s, t, e: (e * 5 + t) % 3 != 0)
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    ra = run(a, {"clean": {"BF16": ref, "VE-W4": qq}, "light": {"BF16": qq, "VE-W4": ref}}, COND)
    # same data, reversed dict order and shuffled lines
    manifest = {"baseline": "BF16", "primary": "VE-W4", "conditions": COND, "cells": {
        "clean": {"BF16": write_cell(b / "c1.jsonl", dict(reversed(ref.items())), shuffle=True),
                  "VE-W4": write_cell(b / "c2.jsonl", qq, shuffle=True)},
        "light": {"BF16": write_cell(b / "l1.jsonl", qq, shuffle=True),
                  "VE-W4": write_cell(b / "l2.jsonl", ref, shuffle=True)}}}
    rb = q.analyse(manifest, b, B=2000, seed=0, threshold=5.0)
    assert ra["qrp"] == rb["qrp"] and ra["gaps"] == rb["gaps"]


def test_subset_setting_is_compared_on_its_matched_subset(tmp_path):
    """W8 runs only part of each shift; it must be compared with BF16 on those episodes only."""
    ref = grid(eps=10, fn=lambda s, t, e: 1)
    w8 = {k: (0 if k[2] == 0 else 1) for k in ref if k[2] < 2}  # 2 of 10 episodes per task
    r = run(tmp_path, {"clean": {"BF16": ref, "VE-W4": ref, "W8": ref},
                       "light": {"BF16": ref, "VE-W4": ref, "W8": w8}}, COND)
    g = r["gaps"]["light/W8"]
    assert g["n_matched"] == 8
    assert g["estimate"] == pytest.approx(50.0)   # 4 failures among the 8 matched, BF16 all succeed
    assert r["qrp"]["W8"]["light"]["estimate"] == pytest.approx(50.0)
    assert not any("W8" in w for w in r["warnings"])  # a subset of BF16 is expected, not a warning


def test_unpartnered_quantized_episodes_are_dropped_with_a_warning(tmp_path):
    ref = grid()
    qq = {**grid(), ("s1", 0, 99): 1}
    r = run(tmp_path, {"clean": {"BF16": ref, "VE-W4": qq}, "light": {"BF16": ref, "VE-W4": ref}}, COND)
    assert r["gaps"]["clean/VE-W4"]["n_matched"] == len(ref)
    assert any("no BF16 partner" in w for w in r["warnings"])


def test_stage_two_draws_exactly_k_units_per_task(tmp_path):
    """One suite, task 0 has 1 pair (d=+1), task 1 has 10 pairs (d=0). Whatever the draw,
    a replicate gap is a*1 / (a*1 + (2-a)*10) for a = number of times task 0 is drawn."""
    ref = {("s", 0, 0): 1, **{("s", 1, e): 1 for e in range(10)}}
    qq = {("s", 0, 0): 0, **{("s", 1, e): 1 for e in range(10)}}
    eps_ref, _ = q.load_cell(write_cell(tmp_path / "r.jsonl", ref), tmp_path)
    eps_q, _ = q.load_cell(write_cell(tmp_path / "q.jsonl", qq), tmp_path)
    pr = q.match("clean", "VE-W4", eps_ref, eps_q, eps_ref)
    draws = q.draw_tasks({"s": [("s", 0), ("s", 1)]}, 3000, np.random.default_rng(0))
    reps = q.boot_gap(pr, draws, np.random.default_rng(1))
    assert set(np.round(reps, 6)) <= {0.0, round(100 / 11, 6), 100.0}
    assert {0.0, 100.0} <= set(np.round(reps, 6))


def test_task_draws_are_shared_across_conditions(tmp_path):
    """A task-level effect present in both clean and shifted data must cancel in QRP:
    if each task's gap is identical in both conditions, every replicate QRP is ~0 apart
    from the independent within-task draws - so with one unit per task it is exactly 0."""
    ref = {("s", t, 0): 1 for t in range(10)}
    qq = {("s", t, 0): int(t % 3 != 0) for t in range(10)}  # task-level differences
    r = run(tmp_path, {"clean": {"BF16": ref, "VE-W4": qq}, "light": {"BF16": ref, "VE-W4": qq}}, COND, B=3000)
    s = r["qrp"]["VE-W4"]["light"]
    assert s["estimate"] == 0.0 and s["ci_low"] == 0.0 and s["ci_high"] == 0.0


def test_variant_clusters_are_resampled_whole(tmp_path):
    """Seeds of one variant share the scene; outcomes that agree within a variant carry
    less information than independent episodes, so clustering must widen the interval."""
    fn = lambda s, t, e: int((e // 5 + t) % 2 == 0)       # constant within blocks of 5 seeds
    ref = grid(tasks=5, eps=10, fn=lambda s, t, e: 1)
    q_var = grid(tasks=5, eps=10, fn=fn, variant=lambda s, t, e: f"v{e // 5}")
    ref_var = grid(tasks=5, eps=10, fn=lambda s, t, e: 1, variant=lambda s, t, e: f"v{e // 5}")
    q_plain = grid(tasks=5, eps=10, fn=fn)
    clean = grid(tasks=5, eps=10, fn=lambda s, t, e: 1)
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    r_var = run(a, {"clean": {"BF16": clean, "VE-W4": clean}, "light": {"BF16": ref_var, "VE-W4": q_var}}, COND,
                bootstrap_within="resample")
    r_plain = run(b, {"clean": {"BF16": clean, "VE-W4": clean}, "light": {"BF16": ref, "VE-W4": q_plain}}, COND,
                  bootstrap_within="resample")
    assert r_var["gaps"]["light/VE-W4"]["estimate"] == r_plain["gaps"]["light/VE-W4"]["estimate"]
    assert r_var["gaps"]["light/VE-W4"]["se"] > 1.5 * r_plain["gaps"]["light/VE-W4"]["se"]


def test_language_contrast_and_reduced_set_flag(tmp_path):
    cond = {"clean": {"kind": "clean"}, "light": {"kind": "visual"}, "lang": {"kind": "language"}}
    ref = grid(eps=10)
    q_light = grid(eps=10, fn=lambda s, t, e: int(e >= 2))   # gap 20
    q_lang = grid(eps=10, fn=lambda s, t, e: int(e >= 1))    # gap 10
    r = run(tmp_path, {"clean": {"BF16": ref, "VE-W4": ref}, "light": {"BF16": ref, "VE-W4": q_light},
                       "lang": {"BF16": ref, "VE-W4": q_lang}}, cond,
            planned_visual_shifts=["light", "background", "layout"])
    out = r["qrp"]["VE-W4"]
    assert out["lang"]["estimate"] == pytest.approx(10.0)
    assert out["visual_minus_language"]["estimate"] == pytest.approx(10.0)
    assert out["pooled_visual"]["reduced_set"] is True
    assert r["tests"]["holm"]["m"] == 2  # H3 + one per-shift test (no W4 cell -> no H2)


@pytest.mark.parametrize("lo,hi,expected", [
    (-4.0, -0.5, "NEGATIVE"),
    (5.5, 12.0, "practically meaningful"),
    (0.5, 12.0, "size uncertain"),
    (5.0, 9.0, "size uncertain"),       # entirely above 0 but not ENTIRELY above +5
    (-3.0, 4.9, "NO PRACTICALLY IMPORTANT"),
    (-3.0, 5.0, "INCONCLUSIVE"),        # reaches +5: cannot rule out a 5-point penalty
    (-3.0, 11.0, "INCONCLUSIVE"),
    (0.0, 3.0, "NO PRACTICALLY IMPORTANT"),  # touches 0 -> "includes zero"
])
def test_h1_decision_rule(lo, hi, expected):
    assert expected in q.h1_decision(lo, hi, 5.0)


def test_holm_step_down_stops_at_first_failure():
    out = q.holm([("a", 0.001), ("b", 0.04), ("c", 0.012), ("d", 0.30)], alpha=0.05)
    # sorted: a .001 <= .05/4 yes; c .012 <= .05/3 yes; b .04 <= .05/2 no -> stop; d no
    assert out["a"]["reject"] and out["c"]["reject"]
    assert not out["b"]["reject"] and not out["d"]["reject"]
    assert out["b"]["threshold"] == pytest.approx(0.025)


def test_guards(tmp_path):
    ref = grid()
    bad_task = {**grid(), ("s1", 7, 0): 1}  # task 7 does not exist in clean
    with pytest.raises(SystemExit, match="not in the clean baseline"):
        run(tmp_path, {"clean": {"BF16": ref, "VE-W4": ref}, "light": {"BF16": bad_task, "VE-W4": bad_task}}, COND)
    p = tmp_path / "dup.jsonl"
    write_cell(p, ref)
    p.write_text(p.read_text() + p.read_text().splitlines()[1] + "\n")
    with pytest.raises(SystemExit, match="duplicate episode"):
        q.load_cell("dup.jsonl", tmp_path)
    manifest = {"conditions": COND, "cells": {
        "clean": {"BF16": write_cell(tmp_path / "x1.jsonl", ref), "VE-W4": write_cell(tmp_path / "x2.jsonl", ref, ckpt="other")},
        "light": {"BF16": "x1.jsonl", "VE-W4": "x1.jsonl"}}}
    with pytest.raises(SystemExit, match="different checkpoints"):
        q.analyse(manifest, tmp_path, B=100, seed=0, threshold=5.0)


@needs_logs
def test_released_logs_reproduce_proposal_numbers():
    """Table 1 / Section 7.4 numbers from VLAQuantBench's shipped pi0 Spatial logs."""
    root = VQB
    r = q.analyse(
        {"conditions": {"clean": {"kind": "clean"}, "same": {"kind": "visual"}},
         "cells": {c: {"BF16": "results/libero/libero_spatial/pi0/rtn-BASELINE-e2e.jsonl",
                       "VE-W4": "results/libero/libero_spatial/pi0/rtn-W4-ve.jsonl"} for c in ("clean", "same")},
         "bootstrap_within": "resample"},
        root, B=10_000, seed=0, threshold=5.0)
    g = r["gaps"]["clean/VE-W4"]
    assert g["estimate"] == pytest.approx(1.5)            # 78.0 - 76.5
    assert 2.0 < g["se"] < 2.6                             # proposal 7.4 ("about 2.3") used this scheme
    assert r["qrp"]["VE-W4"]["same"]["estimate"] == 0.0


@needs_logs
def test_tasks_only_scheme_is_narrower_and_recorded():
    root = VQB
    m = {"conditions": {"clean": {"kind": "clean"}, "same": {"kind": "visual"}},
         "cells": {c: {"BF16": "results/libero/libero_spatial/pi0/rtn-BASELINE-e2e.jsonl",
                       "VE-W4": "results/libero/libero_spatial/pi0/rtn-W4-ve.jsonl"} for c in ("clean", "same")}}
    two = q.analyse({**m, "bootstrap_within": "resample"}, root, B=5000, seed=0, threshold=5.0)
    one = q.analyse(m, root, B=5000, seed=0, threshold=5.0)   # default is fixed (team decision Oct 30)
    assert two["config"]["bootstrap_within"] == "resample" and one["config"]["bootstrap_within"] == "fixed"
    assert one["gaps"]["clean/VE-W4"]["estimate"] == two["gaps"]["clean/VE-W4"]["estimate"]
    assert one["gaps"]["clean/VE-W4"]["se"] < 0.8 * two["gaps"]["clean/VE-W4"]["se"]
    # tasks-only with identical data in both conditions: every replicate QRP is exactly 0
    s = one["qrp"]["VE-W4"]["same"]
    assert s["ci_low"] == s["ci_high"] == 0.0
    with pytest.raises(SystemExit, match="bootstrap_within"):
        q.analyse({**m, "bootstrap_within": "both"}, root, B=100, seed=0, threshold=5.0)


def test_provenance_mismatch_stops_the_analysis_unless_allowed(tmp_path):
    """One-machine rule: cells from another host / GPU / commit / seed / episodes_per_task are refused."""
    ref = grid()
    manifest = {"conditions": COND, "cells": {
        "clean": {"BF16": write_cell(tmp_path / "c1.jsonl", ref), "VE-W4": write_cell(tmp_path / "c2.jsonl", ref)},
        "light": {"BF16": write_cell(tmp_path / "l1.jsonl", ref), "VE-W4": write_cell(tmp_path / "l2.jsonl", ref)}}}
    assert q.analyse(manifest, tmp_path, B=100, seed=0, threshold=5.0)["warnings"] == []
    for key, value in (("hostname", "other-host"), ("gpu", "RTX 5080"), ("vqb_commit", "4a2cb7c"),
                       ("seed", 1), ("episodes_per_task", 50)):
        path = tmp_path / "l2.jsonl"
        lines = path.read_text().splitlines()
        original = lines[0]
        h = json.loads(original); h[key] = value
        path.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n")
        with pytest.raises(SystemExit, match=key):
            q.analyse(manifest, tmp_path, B=100, seed=0, threshold=5.0)
        r = q.analyse({**manifest, "allow_mixed_provenance": True}, tmp_path, B=100, seed=0, threshold=5.0)
        assert any(key in w and "light/VE-W4" in w for w in r["warnings"])
        path.write_text("\n".join([original] + lines[1:]) + "\n")


def test_sensitivity_section_reports_the_other_scheme(tmp_path, capsys):
    ref = grid(eps=6, fn=lambda s, t, e: (e + t) % 4 != 0)
    qq = grid(eps=6, fn=lambda s, t, e: (e * 3 + t) % 5 != 0)
    cells = {}
    for cond, (a, b) in {"clean": (ref, ref), "light": (ref, qq)}.items():
        cells[cond] = {"BF16": write_cell(tmp_path / f"{cond}-a.jsonl", a), "VE-W4": write_cell(tmp_path / f"{cond}-b.jsonl", b)}
    m = tmp_path / "m.yaml"
    m.write_text(json.dumps({"conditions": COND, "cells": cells, "sensitivity_within": "resample"}))
    out = tmp_path / "r.json"
    q.main([str(m), "--B", "500", "--out-json", str(out)])
    md = capsys.readouterr().out
    assert "resampling: tasks only" in md and "## Sensitivity analysis (resampling: tasks, then units within task)" in md
    r = json.loads(out.read_text())
    assert r["config"]["bootstrap_within"] == "fixed" and r["sensitivity"]["bootstrap_within"] == "resample"
    assert r["sensitivity"]["qrp"]["VE-W4"]["pooled_visual"]["estimate"] == r["qrp"]["VE-W4"]["pooled_visual"]["estimate"]
