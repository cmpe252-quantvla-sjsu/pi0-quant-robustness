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
                                 "extra": {"variant_id": None, "instruction": f"put the bowl on the plate — task {t}"}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    return (list(csv.DictReader(open(out / "coding_sheet.csv", newline="", encoding="utf-8"))),
            list(csv.DictReader(open(out / "key/key.csv", newline="", encoding="utf-8"))))


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
    with open(out / "coding_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader(); w.writerows(sheet)
    if sheet[0]["assigned"] == "both":
        with pytest.raises(SystemExit, match="no final code"):
            e5.main(["report", "--out", str(out), "--codebook", codebook(tmp_path)])
        sheet[0]["final_code"] = "TIMEOUT"
        with open(out / "coding_sheet.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader(); w.writerows(sheet)
    e5.main(["report", "--out", str(out), "--md", str(tmp_path / "r.md"), "--codebook", codebook(tmp_path)])
    md = (tmp_path / "r.md").read_text()
    assert f"| clean | W4 | W4 | 3 |" in md and "| clean | VE-W4 | VE-W4 | 2 |" in md


def test_unknown_code_is_refused(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
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


def test_copies_do_not_keep_the_original_timestamps(tmp_path):
    """Settings ran one after another, so original timestamps would reveal the setting."""
    import os
    base, videos, man = fake(tmp_path)
    for f in videos.rglob("*.mp4"):
        os.utime(f, (1_000_000_000, 1_000_000_000))      # 2001
    run_sample(base, videos, man, tmp_path / "e5", w4_per_direction=3)
    assert all(f.stat().st_mtime > 1_500_000_000 for f in (tmp_path / "e5/videos").iterdir())


def test_dry_run_and_missing_videos_write_nothing(tmp_path):
    base, videos, man = fake(tmp_path)
    out = tmp_path / "e5"
    e5.main(["sample", "--manifest", str(man), "--base-dir", str(base), "--videos", str(videos),
             "--out", str(out), "--dry-run"])
    assert not out.exists()
    next(videos.rglob("*_fail.mp4")).unlink()
    with pytest.raises(SystemExit):
        e5.main(["sample", "--manifest", str(man), "--base-dir", str(base), "--videos", str(videos),
                 "--out", str(out)])
    assert not out.exists()
    # after a dry run, the real run still works without --force
    fake(tmp_path / "again")
    b2, v2, m2 = tmp_path / "again/repo", tmp_path / "again/videos", tmp_path / "again/repo/manifest.yaml"
    e5.main(["sample", "--manifest", str(m2), "--base-dir", str(b2), "--videos", str(v2), "--out", str(out), "--dry-run"])
    sheet, _ = run_sample(b2, v2, m2, out, w4_per_direction=3)
    assert sheet and "—" in sheet[0]["instruction"]        # non-ASCII text survives (utf-8)


def test_report_counts_shared_bf16_failure_in_both_pairs_and_first_wrong_object_notes(tmp_path):
    base, videos, man = fake(tmp_path)
    out = tmp_path / "e5"
    sheet, key = run_sample(base, videos, man, out, w4_per_direction=3)
    shared_bf16 = [k["video_id"] for k in key if k["failed_setting"] == "BF16"]
    assert len(shared_bf16) == 2 and shared_bf16[0] == shared_bf16[1]   # one video, two pairs
    for r in sheet:
        for c in ("A", "B"):
            if r["assigned"] in (c, "both"):
                r["code_" + c] = "TIMEOUT"
        if r["video_id"] == shared_bf16[0]:
            r["note_A"] = "First: wrong_object (went for the plate), then grasp"
    with open(out / "coding_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader(); w.writerows(sheet)
    e5.main(["report", "--out", str(out), "--md", str(tmp_path / "r.md"), "--codebook", codebook(tmp_path)])
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "| clean | VE-W4 | BF16 | 1 | 1 |" in md and "| clean | W4 | BF16 | 1 | 1 |" in md
    assert "| clean | VE-W4 | VE-W4 | 2 | 0 |" in md


def test_merge_combines_both_copies_and_reads_excel_bom(tmp_path, capsys):
    rows = [{"video_id": "v1", "instruction": "x", "assigned": "both"},
            {"video_id": "v2", "instruction": "y", "assigned": "A"}]
    def write(name, enc, extra):
        p = tmp_path / name
        with open(p, "w", newline="", encoding=enc) as fh:
            w = csv.DictWriter(fh, e5.SHEET_FIELDS); w.writeheader()
            for r in rows:
                w.writerow({**r, **extra.get(r["video_id"], {})})
        return p
    a = write("A.csv", "utf-8", {"v1": {"code_A": "TNR", "note_A": "first: WRONG_OBJECT"}, "v2": {"code_A": "DROP"}})
    b = write("B.csv", "utf-8-sig", {"v1": {"code_B": "DROP"}})   # Excel "CSV UTF-8" adds a BOM
    out = tmp_path / "merged.csv"
    e5.main(["merge", "--a", str(a), "--b", str(b), "--to", str(out)])
    m = {r["video_id"]: r for r in e5.load_csv(out)}
    assert (m["v1"]["code_A"], m["v1"]["note_A"], m["v1"]["code_B"]) == ("TNR", "first: WRONG_OBJECT", "DROP")
    assert m["v2"]["code_A"] == "DROP" and m["v2"]["code_B"] == ""
    e5.main(["kappa", "--sheet", str(b), "--codebook", codebook(tmp_path)])   # BOM sheet reads fine
    with pytest.raises(SystemExit, match="exists"):
        e5.main(["merge", "--a", str(a), "--b", str(b), "--to", str(out)])
    rows.append({"video_id": "v3", "instruction": "z", "assigned": "B"})
    c = write("C.csv", "utf-8", {})
    with pytest.raises(SystemExit, match="same videos"):
        e5.main(["merge", "--a", str(a), "--b", str(c), "--to", str(tmp_path / "m2.csv")])
