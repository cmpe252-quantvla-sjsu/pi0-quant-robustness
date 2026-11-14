"""Tests for scripts/e5_videos.py (E5 blinded failure coding), on a tiny fake grid."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e5_videos as e5  # noqa: E402

SUITE = "libero_spatial"
CODEBOOK = "| Code | Label |\n|---|---|\n| `TNR` | a |\n| `DROP` | b |\n| `TIMEOUT` | c |\n"


def codebook(tmp_path):
    p = tmp_path / "codebook.md"
    p.write_text(CODEBOOK)
    return str(p)


def write(path, outcomes):
    """outcomes: {(task, episode): success}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "header"})]
    for (t, e), ok in outcomes.items():
        lines.append(json.dumps({"kind": "episode", "suite": SUITE, "task_id": t, "episode": e, "success": bool(ok),
                                 "extra": {"variant_id": None, "instruction": f"task {t}"}}))
    path.write_text("\n".join(lines) + "\n")


def fake(tmp_path, n_w4_each=5):
    """clean grid: BF16 all ok except (0,0); VE-W4 fails (1,0),(1,1), ok on (0,0); W4 fails many."""
    base, videos = tmp_path / "repo", tmp_path / "videos"
    keys = [(t, e) for t in range(4) for e in range(10)]
    bf16 = {k: int(k != (0, 0)) for k in keys}
    vew4 = {k: int(k not in [(1, 0), (1, 1)]) for k in keys}
    w4 = {k: int(not (k[0] == 2 or k == (3, 0))) for k in keys}
    write(base / "results/E2/clean/E2-BF16-spatial.jsonl", bf16)
    write(base / "results/E2/clean/E2-VE-W4-spatial.jsonl", vew4)
    write(base / "results/E2/clean/E2-W4-spatial.jsonl", w4)
    for setting, oc in (("BF16", bf16), ("VE-W4", vew4), ("W4", w4)):
        d = videos / "E2/clean" / setting
        d.mkdir(parents=True, exist_ok=True)
        for (t, e), ok in oc.items():
            (d / f"{SUITE}_t{t}_e{e}_{'ok' if ok else 'fail'}.mp4").write_bytes(f"{setting}{t}{e}".encode())
    man = base / "manifest.yaml"
    man.write_text(json.dumps({"cells": {"clean": {
        "BF16": "results/E2/clean/E2-BF16-*.jsonl", "VE-W4": "results/E2/clean/E2-VE-W4-*.jsonl",
        "W4": "results/E2/clean/E2-W4-*.jsonl"}}}))
    return base, videos, man


def run_sample(base, videos, man, out, **kw):
    args = ["sample", "--manifest", str(man), "--base-dir", str(base), "--videos", str(videos), "--out", str(out),
            "--shared", "4"] + [x for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    e5.main(args)
    return list(csv.DictReader(open(out / "coding_sheet.csv"))), list(csv.DictReader(open(out / "key/key.csv")))


def test_all_ve_w4_pairs_and_capped_w4_pairs_one_video_each(tmp_path):
    base, videos, man = fake(tmp_path)
    sheet, key = run_sample(base, videos, man, tmp_path / "e5", w4_per_direction=3)
    ve = [k for k in key if k["pair_setting"] == "VE-W4"]
    assert len(ve) == 3                                        # (1,0),(1,1) Q fail + (0,0) BF16 fail
    w4 = [k for k in key if k["pair_setting"] == "W4"]
    assert sum(k["direction"] == "BF16 ok, Q fail" for k in w4) == 3   # capped from 11
    assert sum(k["direction"] == "Q ok, BF16 fail" for k in w4) == 1   # only (0,0)
    # the BF16 failure (0,0) is discordant with both settings but is one video
    assert len(sheet) == len({k["video_id"] for k in key}) == len(key) - 1
    copied = sorted(p.read_bytes() for p in (tmp_path / "e5/videos").iterdir())
    assert b"BF1600" in copied and len(copied) == len(sheet)
    # the coding sheet never names the setting, condition or episode
    assert set(sheet[0]) == set(e5.SHEET_FIELDS)
    assert {r["assigned"] for r in sheet} <= {"both", "A", "B"} and sum(r["assigned"] == "both" for r in sheet) == 4


def test_same_seed_same_sample_and_no_resampling(tmp_path):
    base, videos, man = fake(tmp_path)
    s1, k1 = run_sample(base, videos, man, tmp_path / "a", w4_per_direction=3)
    s2, k2 = run_sample(base, videos, man, tmp_path / "b", w4_per_direction=3)
    assert s1 == s2 and k1 == k2
    with pytest.raises(SystemExit, match="never resample"):
        run_sample(base, videos, man, tmp_path / "a", w4_per_direction=3)


def test_kappa_values():
    assert e5.kappa(["TNR", "DROP", "TNR", "DROP"], ["TNR", "DROP", "TNR", "DROP"]) == 1.0
    assert e5.kappa(["TNR", "TNR", "DROP", "DROP"], ["TNR", "DROP", "TNR", "DROP"]) == pytest.approx(0.0)


def test_report_needs_final_codes_and_counts_per_pair(tmp_path, capsys):
    base, videos, man = fake(tmp_path)
    out = tmp_path / "e5"
    sheet, key = run_sample(base, videos, man, out, w4_per_direction=3)
    for r in sheet:
        c = "TIMEOUT"
        if r["assigned"] == "both":
            r["code_A"] = r["code_B"] = c
        else:
            r["code_" + r["assigned"]] = c
    sheet[0]["code_B"] = "drop"                  # lower case is accepted
    if sheet[0]["assigned"] == "both":
        sheet[0]["final_code"] = ""
    with open(out / "coding_sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader(); w.writerows(sheet)
    if sheet[0]["assigned"] == "both":
        with pytest.raises(SystemExit, match="no final code"):
            e5.main(["report", "--out", str(out), "--codebook", codebook(tmp_path)])
        sheet[0]["final_code"] = "TIMEOUT"
        with open(out / "coding_sheet.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader(); w.writerows(sheet)
    e5.main(["report", "--out", str(out), "--md", str(tmp_path / "r.md"), "--codebook", codebook(tmp_path)])
    md = (tmp_path / "r.md").read_text()
    assert f"| clean | W4 | W4 | 3 |" in md and "| clean | VE-W4 | VE-W4 | 2 |" in md


def test_unknown_code_is_refused(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader()
        w.writerow({"video_id": "v1", "assigned": "A", "code_A": "SLIP"})
    with pytest.raises(SystemExit, match="not a codebook code"):
        e5.main(["kappa", "--sheet", str(p), "--codebook", codebook(tmp_path)])


def test_codes_come_from_the_codebook_table(tmp_path):
    assert e5.codebook_codes(codebook(tmp_path)) == ("TNR", "DROP", "TIMEOUT")
    empty = tmp_path / "empty.md"
    empty.write_text("# nothing yet\n")
    with pytest.raises(SystemExit, match="no codes found"):
        e5.codebook_codes(str(empty))
