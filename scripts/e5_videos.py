#!/usr/bin/env python
"""E5: blinded failure-mode coding (docs/FAILURE_CODEBOOK.md).

``sample``  Finds the discordant BF16 / quantized pairs in the core results, copies the video
            of the episode that failed to a random ID, and writes a coding sheet. The key
            (which setting failed, condition, task, episode) goes to a separate file that
            neither coder opens until coding is finished.

    python scripts/e5_videos.py sample --videos /opt/qvla/videos --out /opt/qvla/e5

``kappa``   Cohen's kappa on the shared videos, and the list of disagreements.

    python scripts/e5_videos.py kappa --sheet /opt/qvla/e5/coding_sheet.csv

``report``  Unblinds: failure codes per condition and failing setting.

    python scripts/e5_videos.py report --out /opt/qvla/e5 --md analysis/e5_failure_modes.md

Allowed codes are read from the frozen codebook's table (``--codebook``), so this script never
needs editing when the codebook changes before the freeze.

Sampling (pre-registered in the codebook): every VE-W4 discordant pair in every condition,
plus up to ``--w4-per-direction`` W4 pairs per condition and direction, drawn with ``--seed``.
Each pair contributes one video: the episode that failed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

import re

DEFAULT_CODEBOOK = "docs/FAILURE_CODEBOOK.md"


def codebook_codes(path: str) -> tuple[str, ...]:
    """The codes of the frozen codebook: the first column of its table, written as `CODE`."""
    codes = tuple(re.findall(r"^\|\s*`([A-Z_]+)`\s*\|", Path(path).read_text(encoding="utf-8"), flags=re.M))
    if not codes:
        raise SystemExit(f"{path}: no codes found (expected table rows like | `TNR` | ...)")
    return codes
CODERS = ("A", "B")
SHEET_FIELDS = ["video_id", "instruction", "assigned", "code_A", "note_A", "code_B", "note_B", "final_code"]
KEY_FIELDS = ["video_id", "condition", "failed_setting", "pair_setting", "direction", "suite", "task_id",
              "episode", "variant_id", "source"]


def load(pattern: str, base: Path) -> dict:
    """(suite, task_id, variant_id, episode) -> (success, instruction) for every episode in the glob."""
    out = {}
    files = sorted(glob.glob(str(base / pattern)))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    for f in files:
        for line in open(f):
            d = json.loads(line)
            if d.get("kind") != "episode":
                continue
            x = d.get("extra") or {}
            out[(d["suite"], int(d["task_id"]), x.get("variant_id"), int(d["episode"]))] = (
                int(bool(d["success"])), x.get("instruction", ""))
    return out


def video_dir(pattern: str, videos: Path, setting: str) -> Path:
    """results/E3/lighting/E3-BF16-*.jsonl -> <videos>/E3/lighting/<setting>/"""
    parts = Path(pattern).parts
    return videos.joinpath(*parts[1:-1], setting)


def video_name(suite: str, task: int, episode: int) -> str:
    return f"{suite}_t{task}_e{episode}_fail.mp4"   # VLAQuantBench's naming (benchmarks/libero.py)


def find_pairs(manifest: dict, base: Path, settings: list[str]) -> list[dict]:
    pairs = []
    cells = manifest["cells"]
    for cond, by_setting in cells.items():
        ref = load(by_setting["BF16"], base)
        for setting in settings:
            if setting not in by_setting:
                continue
            q = load(by_setting[setting], base)
            for key in sorted(ref.keys() & q.keys(), key=str):
                (r, instr), (s, _) = ref[key], q[key]
                if r == s:
                    continue
                suite, task, variant, ep = key
                failed = setting if r else "BF16"
                pairs.append({
                    "condition": cond, "pair_setting": setting, "failed_setting": failed,
                    "direction": "BF16 ok, Q fail" if r else "Q ok, BF16 fail",
                    "suite": suite, "task_id": task, "episode": ep, "variant_id": variant or "",
                    "instruction": instr,
                    "pattern": by_setting[failed],
                })
    return pairs


def select(pairs: list[dict], w4_per_direction: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen = [p for p in pairs if p["pair_setting"] == "VE-W4"]
    groups = defaultdict(list)
    for p in pairs:
        if p["pair_setting"] == "W4":
            groups[(p["condition"], p["direction"])].append(p)
    for k in sorted(groups):
        g = groups[k]
        chosen += rng.sample(g, min(w4_per_direction, len(g)))
    return chosen


def cmd_sample(a):
    manifest = yaml.safe_load(open(a.manifest))
    pairs = find_pairs(manifest, Path(a.base_dir), ["VE-W4", "W4"] if a.w4_per_direction else ["VE-W4"])
    chosen = select(pairs, a.w4_per_direction, a.seed)
    for p in chosen:
        p["source"] = str(video_dir(p["pattern"], Path(a.videos), p["failed_setting"])
                          / video_name(p["suite"], p["task_id"], p["episode"]))
    # one video per failed episode: a BF16 failure can be discordant with both VE-W4 and W4
    sources = sorted({p["source"] for p in chosen})
    rng = random.Random(f"{a.seed}|ids")
    rng.shuffle(sources)
    ids = {}
    while len(ids) < len(sources):
        vid = f"v{rng.randrange(16**5):05x}"
        if vid not in ids.values():
            ids[sources[len(ids)]] = vid

    out = Path(a.out)
    if (out / "coding_sheet.csv").exists() and not a.force:
        raise SystemExit(f"{out}/coding_sheet.csv exists; never resample after coding starts (use --force only before)")
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "key").mkdir(parents=True, exist_ok=True)
    order = list(range(len(sources)))
    shared = set(rng.sample(order, min(a.shared, len(order))))
    rest = [i for i in order if i not in shared]

    missing, sheet, key = [], [], []
    instr = {p["source"]: p["instruction"] for p in chosen}
    for i, src in enumerate(sources):
        vid = ids[src]
        if not Path(src).exists():
            missing.append(src)
        elif not a.dry_run:
            shutil.copy2(src, out / "videos" / f"{vid}.mp4")
        assigned = "both" if i in shared else CODERS[rest.index(i) % 2]
        sheet.append({"video_id": vid, "instruction": instr[src], "assigned": assigned})
    for p in chosen:
        key.append({"video_id": ids[p["source"]], **{k: p[k] for k in KEY_FIELDS if k in p}})

    with open(out / "coding_sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, SHEET_FIELDS); w.writeheader()
        for r in sorted(sheet, key=lambda r: r["video_id"]):
            w.writerow(r)
    with open(out / "key" / "key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, KEY_FIELDS); w.writeheader(); w.writerows(key)

    per = Counter((p["condition"], p["pair_setting"], p["direction"]) for p in chosen)
    print(f"{len(sources)} videos ({len(shared)} shared) for {len(chosen)} selected pairs, from {len(pairs)} discordant pairs")
    for k in sorted(per):
        print(f"  {k[0]:11s} {k[1]:6s} {k[2]:16s} {per[k]}")
    print(f"coding sheet: {out/'coding_sheet.csv'}   key (do not open): {out/'key'/'key.csv'}")
    if missing:
        print(f"MISSING {len(missing)} videos, e.g. {missing[0]}", file=sys.stderr)
        sys.exit(1)


def read_sheet(path, codes):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for c in ("code_A", "code_B", "final_code"):
            v = (r.get(c) or "").strip().upper()
            if v and v not in codes:
                raise SystemExit(f"{r['video_id']}: {c} = {v!r} is not a codebook code {codes}")
            r[c] = v
    return rows


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_kappa(a):
    rows = [r for r in read_sheet(a.sheet, codebook_codes(a.codebook)) if r["assigned"] == "both"]
    done = [r for r in rows if r["code_A"] and r["code_B"]]
    if len(done) < len(rows):
        print(f"{len(rows) - len(done)} shared videos not coded by both yet")
    if not done:
        return
    k = kappa([r["code_A"] for r in done], [r["code_B"] for r in done])
    agree = sum(r["code_A"] == r["code_B"] for r in done)
    print(f"Cohen's kappa = {k:.2f} on {len(done)} shared videos ({agree} agree)")
    for r in done:
        if r["code_A"] != r["code_B"]:
            print(f"  {r['video_id']}: A {r['code_A']} vs B {r['code_B']}")


def final(r):
    if r["final_code"]:
        return r["final_code"]
    if r["assigned"] == "both":
        return r["code_A"] if r["code_A"] == r["code_B"] else ""
    return r["code_" + r["assigned"]]


def cmd_report(a):
    out = Path(a.out)
    codes = codebook_codes(a.codebook)
    rows = {r["video_id"]: r for r in read_sheet(out / "coding_sheet.csv", codes)}
    key = {r["video_id"]: r for r in csv.DictReader(open(out / "key" / "key.csv"))}
    open_rows = [v for v, r in rows.items() if not final(r)]
    if open_rows:
        raise SystemExit(f"{len(open_rows)} videos have no final code yet (e.g. {open_rows[0]}); "
                         "resolve shared disagreements in final_code first")
    tab = defaultdict(Counter)
    for v, r in rows.items():
        k = key[v]
        tab[(k["condition"], k["pair_setting"], k["failed_setting"])][final(r)] += 1
    L = ["# E5 failure modes (unblinded)", "",
         "One video per discordant pair: the episode that failed. Counts of the primary code.", "",
         "| condition | pair | failed | n | " + " | ".join(codes) + " |",
         "|---|---|---|---|" + "---|" * len(codes)]
    for (c, s, f) in sorted(tab):
        cnt = tab[(c, s, f)]
        L.append(f"| {c} | {s} | {f} | {sum(cnt.values())} | " + " | ".join(str(cnt.get(x, 0)) for x in codes) + " |")
    shared = [r for r in rows.values() if r["assigned"] == "both" and r["code_A"] and r["code_B"]]
    if shared:
        L += ["", f"Agreement on {len(shared)} shared videos: Cohen's kappa = "
              f"{kappa([r['code_A'] for r in shared], [r['code_B'] for r in shared]):.2f}"]
    text = "\n".join(L) + "\n"
    print(text)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--manifest", default="configs/qrp/study_manifest.yaml")
    s.add_argument("--base-dir", default=".")
    s.add_argument("--videos", required=True, help="video root, e.g. /opt/qvla/videos")
    s.add_argument("--out", required=True, help="E5 folder outside the repo, e.g. /opt/qvla/e5")
    s.add_argument("--w4-per-direction", type=int, default=15)
    s.add_argument("--shared", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dry-run", action="store_true", help="check videos exist, copy nothing")
    s.add_argument("--force", action="store_true")
    k = sub.add_parser("kappa"); k.add_argument("--sheet", required=True)
    r = sub.add_parser("report"); r.add_argument("--out", required=True); r.add_argument("--md")
    for p in (k, r):
        p.add_argument("--codebook", default=DEFAULT_CODEBOOK, help="the frozen codebook; codes are read from its table")
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "kappa": cmd_kappa, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
