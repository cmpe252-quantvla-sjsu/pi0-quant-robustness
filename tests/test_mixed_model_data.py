"""Tests for scripts/mixed_model_data.py (long table for the proposal Eq. 7 mixed-effects model)."""
import csv
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("vlaquantbench")  # mixed_model_data reuses qrp_bootstrap's loader
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mixed_model_data as g  # noqa: E402


def cell(path: Path, outcomes: dict, fallback=()):
    """outcomes: {(suite, task_id, episode[, variant]): success}"""
    lines = [json.dumps({"kind": "header", "checkpoint": "c", "seed": 0})]
    for key, ok in outcomes.items():
        extra = {"variant_id": key[3]} if len(key) == 4 else {}
        if key in fallback:
            extra["fallback"] = True
        lines.append(json.dumps({"kind": "episode", "suite": key[0], "task_id": key[1], "episode": key[2],
                                 "success": bool(ok), "steps": 1, "extra": extra}))
    path.write_text("\n".join(lines) + "\n")
    return path.name


def test_rows_are_matched_pairs_per_setting_and_condition(tmp_path):
    clean = {("s", t, e): 1 for t in range(2) for e in range(3)}
    light = {("s", t, e, f"v{e}"): (e != 0) for t in range(2) for e in range(3)}
    w8_light = {k: 1 for k in light if k[2] == 0}               # subset: episode 0 only
    manifest = {
        "baseline": "BF16",
        "conditions": {"clean": {"kind": "clean"}, "light": {"kind": "visual"}, "lang": {"kind": "language"}},
        "cells": {
            "clean": {"BF16": cell(tmp_path / "c-b.jsonl", clean),
                      "VE-W4": cell(tmp_path / "c-v.jsonl", {**clean, ("s", 0, 0): 0}),
                      "W8": cell(tmp_path / "c-w8.jsonl", clean)},
            "light": {"BF16": cell(tmp_path / "l-b.jsonl", light),
                      "VE-W4": cell(tmp_path / "l-v.jsonl", light, fallback={("s", 1, 2, "v2")}),
                      "W8": cell(tmp_path / "l-w8.jsonl", w8_light)},
            "lang": {"BF16": cell(tmp_path / "g-b.jsonl", clean)},   # language rows are never used
        },
    }
    rows = g.build_rows(manifest, tmp_path)
    n = {}
    for r in rows:
        n[(r["comparison"], r["condition"], r["Q"])] = n.get((r["comparison"], r["condition"], r["Q"]), 0) + 1
    assert n == {("VE-W4", "clean", 0): 6, ("VE-W4", "clean", 1): 6, ("VE-W4", "light", 0): 6,
                 ("VE-W4", "light", 1): 6, ("W8", "clean", 0): 6, ("W8", "clean", 1): 6,
                 ("W8", "light", 0): 2, ("W8", "light", 1): 2}   # BF16 restricted to W8's subset
    assert not any(r["condition"] == "lang" for r in rows)
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["comparison"], r["pair"]), []).append(r)
    assert all(len(v) == 2 and {x["Q"] for x in v} == {0, 1} for v in by_pair.values())
    v = {(r["setting"], r["pair"]): r for r in rows if r["comparison"] == "VE-W4"}
    assert v[("VE-W4", "clean|s|0|-|0")]["success"] == 0 and v[("BF16", "clean|s|0|-|0")]["success"] == 1
    assert v[("VE-W4", "light|s|1|v2|2")]["fallback"] == 1 and v[("BF16", "light|s|1|v2|2")]["fallback"] == 0
    assert {r["shifted"] for r in rows if r["condition"] == "light"} == {1}
    assert {r["task"] for r in rows} == {"s:0", "s:1"}


def test_cli_writes_csv(tmp_path, capsys):
    clean = {("s", 0, e): 1 for e in range(2)}
    m = {"baseline": "BF16", "conditions": {"clean": {"kind": "clean"}},
         "cells": {"clean": {"BF16": cell(tmp_path / "b.jsonl", clean), "W4": cell(tmp_path / "w.jsonl", clean)}}}
    (tmp_path / "m.yaml").write_text(json.dumps(m))     # JSON is valid YAML
    out = tmp_path / "long.csv"
    g.main([str(tmp_path / "m.yaml"), "--base-dir", str(tmp_path), "--out", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 4 and list(rows[0]) == g.FIELDS
    assert "2 matched episodes" in capsys.readouterr().out
