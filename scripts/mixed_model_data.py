"""Build the long table for the mixed-effects sensitivity model (proposal Eq. 7).

One row per (comparison, condition, precision setting, matched episode). For each quantized
setting Q (VE-W4, W4, W8) we keep the episodes that BF16 and Q share in a condition, the same
matching as scripts/qrp_bootstrap.py (suite, task_id, variant_id, episode), so the model and
the QRP analysis see exactly the same episodes. W8 therefore uses its 4-per-task subset in the
shifted conditions, with BF16 restricted to those episodes.

    python scripts/mixed_model_data.py configs/qrp/study_manifest.yaml --base-dir . --out analysis/mixed_model_long.csv
    Rscript analysis/mixed_model.R analysis/mixed_model_long.csv analysis/mixed_model.md analysis/mixed_model.json

Columns: comparison (Q setting), condition, shifted (0/1), setting, Q (0 = BF16, 1 = Q),
suite, task ("suite:task_id"), pair (one id per matched episode within a condition),
fallback (0/1), success (0/1).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qrp_bootstrap import load_cell  # noqa: E402

FIELDS = ["comparison", "condition", "shifted", "setting", "Q", "suite", "task", "pair", "fallback", "success"]


def build_rows(manifest: dict, base_dir: Path, settings=None) -> list[dict]:
    ref_name = manifest["baseline"]
    kinds = {c: v["kind"] for c, v in manifest["conditions"].items()}
    cells = manifest["cells"]
    if settings is None:
        settings = [s for s in cells["clean"] if s != ref_name]
    rows = []
    for cond, kind in kinds.items():
        if kind not in ("clean", "visual"):
            continue  # the model is fitted to clean versus visual shifts only
        ref, _ = load_cell(cells[cond][ref_name], base_dir)
        for q_name in settings:
            if q_name not in cells[cond]:
                continue
            q, _ = load_cell(cells[cond][q_name], base_dir)
            common = sorted(set(ref) & set(q), key=lambda k: (k[0], k[1], "" if k[2] is None else k[2], k[3]))
            for k in common:
                suite, task_id, variant, episode = k
                pair = f"{cond}|{suite}|{task_id}|{variant or '-'}|{episode}"
                for setting, eps, qflag in ((ref_name, ref, 0), (q_name, q, 1)):
                    ep = eps[k]
                    rows.append({"comparison": q_name, "condition": cond, "shifted": int(kind == "visual"),
                                 "setting": setting, "Q": qflag, "suite": suite, "task": f"{suite}:{task_id}",
                                 "pair": pair, "fallback": int(ep.fallback), "success": ep.success})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    manifest = yaml.safe_load(open(a.manifest, encoding="utf-8"))
    rows = build_rows(manifest, Path(a.base_dir))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[(r["comparison"], r["condition"])] = by.get((r["comparison"], r["condition"]), 0) + 1
    for (c, cond), n in sorted(by.items()):
        print(f"{c:6s} {cond:11s} {n // 2:4d} matched episodes")
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
