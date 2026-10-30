#!/usr/bin/env python
"""Parity checks for the CMPE 252 harness (proposal E0/E1, Section 6.4).

Two questions, two subcommands:

``weights``  - Does a preset change the model's weights? Loads the policy, fingerprints
               every parameter (SHA-256 of the raw bytes), applies the preset to the
               scope through the normal quantization path, and fingerprints again.
               For NOOP every parameter must be bit-identical. Also prints the dtype of
               the in-scope weights (useful: the pi0 checkpoint keeps some vision /
               norm weights in float32). Needs the model environment; seconds to run.

    python scripts/parity_check.py weights --model pi0 --preset NOOP --scope ve --device cuda

``rollouts`` - Do two runs of matched episodes agree? Episodes are matched by
               (suite, task_id, variant_id, episode).
               --mode exact        same environment, NOOP vs BASELINE: every matched episode
                                   must have identical success AND identical step count
                                   (step count is a cheap trajectory fingerprint).
               --mode statistical  different environment or a nondeterministic GPU: the
                                   cell's success rate must fall inside the reference's
                                   Wilson 95% interval and McNemar's test on the
                                   discordant pairs must not reject (p >= 0.05).

    python scripts/parity_check.py rollouts --ref results/E0/E0b-BF16-patched-spatial.jsonl --cell results/E0/E0b-NOOP-spatial.jsonl
    python scripts/parity_check.py rollouts --mode statistical \\
        --ref results/E0/E0a-BF16-spatial.jsonl --cell results/E0/E0b-BF16-patched-spatial.jsonl

Exit status is 0 on PASS and 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for qrp_bootstrap
from qrp_bootstrap import boot_gap, draw_tasks, load_cell, match  # noqa: E402
from vlaquantbench.stats import wilson_ci  # noqa: E402

HOST_FIELDS = ("hostname", "gpu", "python", "vqb_commit")


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #
def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact McNemar p-value (binomial test of n10 vs n01 at p = 1/2)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def full_headers(patterns: str) -> list[dict]:
    import glob

    out = []
    for p in sorted(glob.glob(patterns)):
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if d.get("kind") == "header":
                    out.append({"path": p, **d})
                    break
    return out


def compare_rollouts(ref_path: str, cell_path: str, mode: str = "exact", B: int = 10_000, seed: int = 0,
                     show: int = 10) -> tuple[bool, list[str]]:
    ref, _ = load_cell(ref_path, Path("."))
    cell, _ = load_cell(cell_path, Path("."))
    L: list[str] = []
    ok = True

    # ---- headers
    h_ref, h_cell = full_headers(ref_path), full_headers(cell_path)
    if not h_ref or not h_cell:
        return False, ["FAIL: a run file has no header line"]
    L.append("Headers")
    for name, hs in (("ref ", h_ref), ("cell", h_cell)):
        for h in hs:
            rep = h.get("quant_report") or {}
            layers = {c: v.get("layers", 0) for c, v in rep.items() if isinstance(v, dict) and "layers" in v}
            L.append(f"  {name} {Path(h['path']).name}: preset={h.get('preset')} scope={h.get('scope')} "
                     f"seed={h.get('seed')} layers={layers}")
    for key in ("checkpoint", "seed", "episodes_per_task"):
        a = {str(h.get(key)) for h in h_ref}
        b = {str(h.get(key)) for h in h_cell}
        if a != b:
            ok = False
            L.append(f"  FAIL: {key} differs: ref {sorted(a)} vs cell {sorted(b)}")
    for key in HOST_FIELDS:
        a = {str(h.get(key)) for h in h_ref}
        b = {str(h.get(key)) for h in h_cell}
        if a != b:
            L.append(f"  note: {key} differs: ref {sorted(a)} vs cell {sorted(b)}"
                     + ("  <- expected in statistical mode" if mode == "statistical" else "  <- breaks the one-machine rule"))
    for h in h_cell:
        if str(h.get("preset", "")).upper() == "NOOP":
            rep = h.get("quant_report") or {}
            n_layers = sum(v.get("layers", 0) for v in rep.values() if isinstance(v, dict))
            mses = [v.get("mean_rel_mse") for v in rep.values() if isinstance(v, dict) and v.get("mean_rel_mse") is not None]
            if n_layers == 0:
                ok = False
                L.append("  FAIL: the NOOP run quantized 0 layers, so it did not exercise the quantization path")
            if any(m != 0 for m in mses):
                ok = False
                L.append(f"  FAIL: the NOOP run reports nonzero weight error {mses}")

    # ---- matching
    pr = match("parity", "cell", ref, cell, ref)
    L.append("")
    L.append(f"Episodes: ref {len(ref)}, cell {len(cell)}, matched {pr.n}")
    if pr.n == 0:
        return False, L + ["FAIL: no matched episodes"]
    if pr.unmatched_ref or pr.unmatched_q:
        L.append(f"  note: {pr.unmatched_ref} ref-only and {pr.unmatched_q} cell-only episodes are ignored")
        if mode == "exact":
            ok = False
            L.append("  FAIL: exact parity needs the same episode set in both runs")

    keys = pr.keys
    same_success = [ref[k].success == cell[k].success for k in keys]
    same_traj = [s and ref[k].steps == cell[k].steps for s, k in zip(same_success, keys)]
    n10, n01 = pr.discordance()
    k_ref = int(pr.y_ref.sum())
    k_cell = int(pr.y_q.sum())
    w = wilson_ci(k_ref, pr.n)
    sr_ref, sr_cell = 100 * k_ref / pr.n, 100 * k_cell / pr.n
    p_mc = mcnemar_exact(n10, n01)
    tasks: dict = {}
    for k in keys:
        tasks.setdefault(ref[k].suite, set()).add(ref[k].task)
    tasks = {s: sorted(v) for s, v in tasks.items()}
    reps = boot_gap(pr, draw_tasks(tasks, B, np.random.default_rng(seed)), np.random.default_rng(seed + 1), within="fixed")
    lo, hi = np.percentile(reps, [2.5, 97.5])

    L.append(f"  success rate: ref {sr_ref:.1f}% (Wilson 95% [{100 * w.low:.1f}, {100 * w.high:.1f}]), cell {sr_cell:.1f}%")
    L.append(f"  same success: {sum(same_success)}/{pr.n}   same success AND same steps: {sum(same_traj)}/{pr.n}")
    L.append(f"  discordant pairs: n10 (only ref ok) = {n10}, n01 (only cell ok) = {n01}, McNemar exact p = {p_mc:.3f}")
    L.append(f"  paired gap ref - cell: {sr_ref - sr_cell:+.1f} pp, task-clustered 95% CI [{lo:+.1f}, {hi:+.1f}]")
    diffs = [k for k, s in zip(keys, same_traj) if not s]
    if diffs:
        L.append(f"  first {min(show, len(diffs))} differing episodes (suite, task, variant, episode): ref -> cell")
        for k in diffs[:show]:
            L.append(f"    {k}: success {ref[k].success}->{cell[k].success}, steps {ref[k].steps}->{cell[k].steps}")

    L.append("")
    if mode == "exact":
        if all(same_traj) and ok:
            L.append("PASS: every matched episode has identical success and step count")
        else:
            ok = False
            L.append("FAIL: the runs are not identical episode-by-episode")
            if not all(same_traj):
                L.append("  Next: run the same BF16 cell once more to a new file and compare BF16 vs BF16. That")
                L.append("  measures how many mismatches GPU non-determinism alone produces. Only if NOOP differs")
                L.append("  clearly more than that is the quantization path changing the model.")
    else:
        inside = w.low <= k_cell / pr.n <= w.high
        if not inside:
            ok = False
            L.append(f"CHECK: cell success rate {sr_cell:.1f}% is outside the reference Wilson interval")
        if p_mc < 0.05:
            ok = False
            L.append(f"CHECK: discordant pairs are unbalanced (McNemar p = {p_mc:.3f})")
        if ok:
            L.append("PASS: success rates agree (cell inside the reference Wilson interval; McNemar p >= 0.05)")
    return ok, L


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def fingerprint(module) -> dict[str, str]:
    """SHA-256 of every parameter's raw bytes (exact; no float tolerance)."""
    import torch

    out = {}
    for name, p in module.named_parameters():
        t = p.detach().reshape(-1).contiguous().cpu()
        out[name] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
    return out


def weight_parity(adapter, preset: str, scope: str) -> tuple[bool, list[str]]:
    """``adapter`` must be loaded (adapter.model set). Applies ``preset`` to ``scope`` in place."""
    from vlaquantbench import parse_spec, quantize_scope
    from vlaquantbench.components import resolve_scope
    from vlaquantbench.quant.apply import DEFAULT_EXCLUDE_PATTERNS, iter_quantizable

    cmap = adapter.component_map()
    comps = resolve_scope(scope, cmap)
    L = [f"model {type(adapter).__name__}, preset {preset}, scope {scope} -> components {list(comps)}"]
    for c in comps:
        excl = list(DEFAULT_EXCLUDE_PATTERNS) + list(cmap.exclude.get(c, []))
        layers = list(iter_quantizable(cmap.roots(c), exclude_patterns=excl, allow_patterns=cmap.allow.get(c, [])))
        dtypes = Counter(str(m.weight.dtype).replace("torch.", "") for _, m in layers)
        L.append(f"  {c}: {len(layers)} quantizable layers, weight dtypes {dict(dtypes)}")
    before = fingerprint(adapter.model)
    reports = quantize_scope(cmap, parse_spec(preset), scope=scope, compute_error=True)
    after = fingerprint(adapter.model)
    changed = [n for n in before if before[n] != after[n]]
    n_layers = sum(r.n_layers for r in reports.values())
    L.append(f"  quantization report: {n_layers} layers touched; "
             + ", ".join(f"{c}: {r.n_layers}" for c, r in reports.items()))
    L.append(f"  parameters changed: {len(changed)} of {len(before)}")
    for n in changed[:10]:
        L.append(f"    {n}")
    is_noop = preset.strip().upper() == "NOOP"
    if is_noop:
        ok = n_layers > 0 and not changed
        L.append("PASS: NOOP walked the quantization path and every parameter is bit-identical" if ok else
                 "FAIL: " + ("NOOP touched no layers" if n_layers == 0 else "NOOP changed parameters"))
    else:
        ok = n_layers > 0 and bool(changed)
        L.append(("PASS" if ok else "FAIL") + f": {preset} changed {len(changed)} parameter tensors")
    return ok, L


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollouts", help="compare two runs of matched episodes")
    r.add_argument("--ref", required=True, help="reference JSONL (path or glob), e.g. the BASELINE cell")
    r.add_argument("--cell", required=True, help="JSONL to check (path or glob), e.g. the NOOP cell")
    r.add_argument("--mode", choices=("exact", "statistical"), default="exact")
    r.add_argument("--B", type=int, default=10_000)
    r.add_argument("--show", type=int, default=10, help="differing episodes to list")
    w = sub.add_parser("weights", help="check what a preset does to the model's weights")
    w.add_argument("--model", default="pi0")
    w.add_argument("--checkpoint", default="lerobot/pi0_libero_finetuned_v044")
    w.add_argument("--preset", default="NOOP")
    w.add_argument("--scope", default="ve")
    w.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "rollouts":
        ok, lines = compare_rollouts(a.ref, a.cell, a.mode, a.B, show=a.show)
    else:
        from vlaquantbench.registry import get_adapter_cls

        adapter = get_adapter_cls(a.model)(a.checkpoint, device=a.device)
        adapter.load()
        ok, lines = weight_parity(adapter, a.preset, a.scope)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
