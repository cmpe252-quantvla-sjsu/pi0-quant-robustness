#!/usr/bin/env python
"""Quantization Robustness Penalty (QRP) with a paired, task-clustered bootstrap.

Implements the primary analysis of the CMPE 252 proposal (Sections 4.4, 7.1-7.3):

    gap_c       = SR_BF16,c - SR_Q,c                  (matched episodes of condition c)
    QRP_p       = gap_p - gap_clean                   (Eq. 1)
    QRP_pooled  = mean over the visual shifts p of QRP_p   (Eq. 5)

Uncertainty: 95% percentile interval from a two-stage bootstrap (default 10,000
replicates). Each replicate

  1. resamples the tasks of every suite with replacement (stratified by suite); the
     SAME task draws are used for every condition and setting in that replicate, and
  2. inside each drawn task, resamples that condition's units with replacement, where
     a unit is one matched (BF16, Q) episode pair - or, when episodes carry a
     ``variant_id``, all episodes of one variant - so BF16 and Q outcomes always move
     together. The clean gap is computed once per replicate and reused for every shift.

Step 2 is what Section 7.1 specifies (``bootstrap_within: resample``). A coverage
simulation at the proposal's design size found that resampling at both stages counts
within-task noise twice (99% coverage for a nominal 95% interval, SE ~35% too large).
``bootstrap_within: fixed`` resamples tasks only and keeps each drawn task's episodes as
observed (92-94% coverage with 30 tasks), as in Field & Welsh (2007). **Team decision
(Oct 30, before any core data): ``fixed`` is primary** and the default; ``resample`` is
reported as a sensitivity analysis (manifest ``sensitivity_within: resample``). The
choice is recorded in the output.

Provenance: all cells must share checkpoint, hostname, GPU, VLAQuantBench commit, run
seed and benchmark (the one-machine, one-environment rule; ``benchmark`` separates
environment A, ``libero``, from environment B, ``libero_plus``, whose headers otherwise
look the same). Episodes per task must agree *within* each condition (clean runs have 20
per task, shifted runs take theirs from the sample file); settings listed under
``subset_settings`` (W8, which runs a subset of each shift) are exempt in the shifted
conditions only; in the clean condition they run the full grid and are checked. A mismatch stops the analysis
unless the manifest sets ``allow_mixed_provenance: true`` (or the CLI flag), in which case
it is reported as a warning.

Fallback sensitivity (manifest ``sensitivity_covered_only: true``): shifted episodes record
``extra.fallback`` when their task had no variant at the target level. The sensitivity
analysis drops, per shifted condition, every task with a fallback episode and computes
that condition's clean gap on the same remaining tasks (pre-registered Nov 4).
The bootstrap keeps one set of task draws for all conditions (so the pooled QRP stays
paired across shifts); a drawn task that was dropped adds neither a difference nor an
episode to that replicate, so it is skipped, not counted as a zero gap.

This extends ``clustered_ci.py``, whose within-task resampling draws the two cells'
episodes independently; here the within-task draw is paired.

Inputs are per-episode JSONL files (VLAQuantBench format), listed in a YAML manifest
(see ``configs/qrp/``). Episodes are matched across settings by
(suite, task_id, variant_id, episode).

    python scripts/qrp_bootstrap.py configs/qrp/study_manifest.yaml --out-json qrp.json --out-md qrp.md
    python scripts/qrp_bootstrap.py configs/qrp/released_logs_check.yaml --base-dir <VLAQuantBench clone>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from vlaquantbench.stats import wilson_ci  # noqa: E402

KINDS = ("clean", "visual", "language")
#: Header fields that must agree across every cell of one analysis (one machine, one environment)
PROVENANCE = ("hostname", "gpu", "vqb_commit", "seed", "benchmark")
#: Header fields that must agree across the cells of one condition only
PER_CONDITION = ("episodes_per_task",)


# --------------------------------------------------------------------------- #
# Loading and matching
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Episode:
    suite: str
    task_id: int
    variant: str | None
    episode: int
    success: int
    steps: int
    fallback: bool = False  # extra.fallback: the task had no variant at the target level

    @property
    def key(self):
        return (self.suite, self.task_id, self.variant, self.episode)

    @property
    def task(self):
        return (self.suite, self.task_id)

    @property
    def unit(self):
        """Resampling unit inside a task: the variant if there is one, else the episode."""
        return (self.suite, self.task_id, "v", self.variant) if self.variant is not None else self.key


def load_cell(patterns, base_dir: Path):
    """Read every episode from the files matched by ``patterns`` (a path or glob, or a list)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(str(base_dir / pat)))
        if not hits:
            raise SystemExit(f"no file matches {pat!r} (relative to {base_dir})")
        paths += hits
    episodes, headers = {}, []
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            if d.get("kind") == "header":
                headers.append({"path": p, **{k: d.get(k) for k in ("checkpoint", "suite", "preset", "scope")
                                                       + PROVENANCE + PER_CONDITION}})
                continue
            if d.get("kind") != "episode":
                continue
            extra = d.get("extra") or {}
            variant = extra.get("variant_id")
            ep = Episode(d["suite"], int(d["task_id"]), None if variant is None else str(variant),
                         int(d["episode"]), int(bool(d["success"])), int(d.get("steps") or 0),
                         bool(extra.get("fallback", False)))
            if ep.key in episodes:
                raise SystemExit(f"duplicate episode {ep.key} in {p}")
            episodes[ep.key] = ep
    return episodes, headers


@dataclass
class Pair:
    """Matched (BF16, Q) episodes of one condition, grouped for the bootstrap."""

    condition: str
    setting: str
    keys: list
    y_ref: np.ndarray
    y_q: np.ndarray
    unmatched_ref: int
    unmatched_q: int
    # filled by index(): per task, the units with their summed difference and size
    units: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.keys)

    def gap(self):  # in percentage points
        return 100.0 * float(np.mean(self.y_ref - self.y_q))

    def discordance(self):
        n10 = int(np.sum((self.y_ref == 1) & (self.y_q == 0)))
        n01 = int(np.sum((self.y_ref == 0) & (self.y_q == 1)))
        return n10, n01


def match(condition, setting, ref: dict, q: dict, ref_eps: dict) -> Pair:
    common = sorted(set(ref) & set(q), key=lambda k: (k[0], k[1], "" if k[2] is None else str(k[2]), k[3]))
    pair = Pair(condition, setting, common,
                np.array([ref[k].success for k in common], dtype=np.int8),
                np.array([q[k].success for k in common], dtype=np.int8),
                unmatched_ref=len(set(ref) - set(q)), unmatched_q=len(set(q) - set(ref)))
    by_task = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # task -> unit -> [sum d, n]
    for k, a, b in zip(common, pair.y_ref, pair.y_q):
        ep = ref_eps[k]
        u = by_task[ep.task][ep.unit]
        u[0] += int(a) - int(b)
        u[1] += 1
    pair.units = {t: np.array(list(us.values()), dtype=np.int64) for t, us in by_task.items()}
    return pair


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def draw_tasks(tasks_by_suite: dict, B: int, rng) -> list:
    """Stage 1: per suite, draw that suite's tasks with replacement. Returns a list of
    (B, T) arrays of task labels' positions plus the label lists."""
    draws = []
    for suite in sorted(tasks_by_suite):
        labels = tasks_by_suite[suite]
        draws.append((labels, rng.integers(0, len(labels), size=(B, len(labels)))))
    return draws


def boot_gap(pair: Pair, task_draws: list, rng, within: str = "resample") -> np.ndarray:
    """Stage 2 for one (condition, setting): B replicate gaps in percentage points.

    ``within="resample"`` (proposal 7.1 as written) also resamples units inside each drawn
    task; ``within="fixed"`` keeps each drawn task's units as observed (tasks-only bootstrap).
    """
    B = task_draws[0][1].shape[0]
    tot_d = np.zeros(B)
    tot_n = np.zeros(B)
    for labels, pick in task_draws:
        K = np.array([len(pair.units.get(t, ())) for t in labels])          # units per task
        kmax = int(K.max()) if K.size else 0
        if kmax == 0:
            continue
        if within == "fixed":
            d_t = np.array([pair.units[t][:, 0].sum() if t in pair.units else 0 for t in labels], dtype=float)
            n_t = np.array([pair.units[t][:, 1].sum() if t in pair.units else 0 for t in labels], dtype=float)
            tot_d += d_t[pick].sum(axis=1)
            tot_n += n_t[pick].sum(axis=1)
            continue
        if within != "resample":
            raise ValueError(f"within must be 'resample' or 'fixed', got {within!r}")
        d = np.zeros((len(labels), kmax))
        n = np.zeros((len(labels), kmax))
        for i, t in enumerate(labels):
            u = pair.units.get(t)
            if u is not None:
                d[i, : len(u)], n[i, : len(u)] = u[:, 0], u[:, 1]
        Kp = K[pick]                                                        # (B, T)
        idx = np.floor(rng.random((B, pick.shape[1], kmax)) * np.maximum(Kp, 1)[..., None]).astype(np.int64)
        live = np.arange(kmax)[None, None, :] < Kp[..., None]               # draw K_t units, not kmax
        rows = pick[..., None]
        tot_d += np.where(live, d[rows, idx], 0.0).sum(axis=(1, 2))
        tot_n += np.where(live, n[rows, idx], 0.0).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * tot_d / tot_n


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def summarize(point: float, reps: np.ndarray, alpha: float = 0.05) -> dict:
    reps = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = float(np.std(reps, ddof=1))
    if se > 0:
        p = math.erfc(abs(point / se) / math.sqrt(2))  # two-sided normal-approximation (Wald) p
    else:  # no resampling variability at all (e.g. identical cells)
        p = 1.0 if point == 0 else 0.0
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "se": se, "wald_p": p}


def h1_decision(lo: float, hi: float, threshold: float = 5.0) -> str:
    """Section 4.4 decision rule, applied in order to the 95% interval of pooled QRP."""
    if hi < 0:
        return "NEGATIVE: the shift reduces the quantization penalty (unexpected)"
    if lo > 0:
        if lo > threshold:
            return f"H1 SUPPORTED, practically meaningful (whole interval above +{threshold:g})"
        return "H1 SUPPORTED, size uncertain (interval above 0 but not entirely above " f"+{threshold:g})"
    if hi < threshold:
        return f"NO PRACTICALLY IMPORTANT PENALTY (interval includes 0, stays below +{threshold:g})"
    return f"INCONCLUSIVE (interval includes 0 and values of +{threshold:g} or more); not evidence of robustness"


def holm(tests: list[tuple[str, float]], alpha: float = 0.05) -> dict:
    """Holm step-down: reject H(k) while p(k) <= alpha/(m-k+1); stop at the first failure."""
    m = len(tests)
    out, still = {}, True
    for k, (name, p) in enumerate(sorted(tests, key=lambda x: x[1]), start=1):
        thr = alpha / (m - k + 1)
        still = still and p <= thr
        out[name] = {"p": p, "threshold": thr, "reject": still}
    return out


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def analyse(manifest: dict, base_dir: Path, B: int, seed: int, threshold: float, log=print,
            within: str | None = None, allow_mixed_provenance: bool | None = None,
            covered_only: bool = False) -> dict:
    baseline = manifest.get("baseline", "BF16")
    primary = manifest.get("primary", "VE-W4")
    conditions = manifest["conditions"]
    for c, spec in conditions.items():
        if spec.get("kind") not in KINDS:
            raise SystemExit(f"condition {c!r}: kind must be one of {KINDS}")
    clean = [c for c, s in conditions.items() if s["kind"] == "clean"]
    if len(clean) != 1:
        raise SystemExit("exactly one condition must have kind: clean")
    clean = clean[0]
    visual = [c for c, s in conditions.items() if s["kind"] == "visual"]
    language = [c for c, s in conditions.items() if s["kind"] == "language"]
    planned_visual = manifest.get("planned_visual_shifts", visual)
    within = within or manifest.get("bootstrap_within", "fixed")
    if within not in ("resample", "fixed"):
        raise SystemExit(f"bootstrap_within must be 'resample' or 'fixed', got {within!r}")

    cells, headers = {}, []
    for cond, by_setting in manifest["cells"].items():
        if cond not in conditions:
            raise SystemExit(f"cells lists unknown condition {cond!r}")
        if baseline not in by_setting:
            raise SystemExit(f"condition {cond!r} has no {baseline} cell")
        for setting, pats in by_setting.items():
            eps, hdr = load_cell(pats, base_dir)
            cells[(cond, setting)] = eps
            headers += [{"condition": cond, "setting": setting, **h} for h in hdr]

    missing_cells = [c for c in conditions if c not in manifest["cells"]]
    if missing_cells:
        print(f"note: declared conditions without cells are skipped: {missing_cells}", file=sys.stderr)
    ckpts = {h["checkpoint"] for h in headers}
    if len(ckpts) > 1:
        raise SystemExit(f"cells come from different checkpoints: {sorted(map(str, ckpts))}")
    if allow_mixed_provenance is None:
        allow_mixed_provenance = bool(manifest.get("allow_mixed_provenance", False))
    provenance_issues = []

    def _differs(key, hs, where):
        seen = defaultdict(list)
        for h in hs:
            seen[str(h.get(key))].append(f"{h['condition']}/{h['setting']}")
        if len(seen) > 1:
            detail = "; ".join(f"{v}: {', '.join(sorted(set(c)))}" for v, c in sorted(seen.items()))
            provenance_issues.append(f"{key} differs {where} ({detail})")

    for key in PROVENANCE:
        _differs(key, headers, "across cells")
    # subset settings (W8) run a subset of the shifted episodes only; in the clean condition they
    # run the full grid, so they are checked there like any other setting
    subset = set(manifest.get("subset_settings", []))
    for key in PER_CONDITION:
        for cond in manifest["cells"]:
            exempt = subset if cond != clean else set()
            _differs(key, [h for h in headers if h["condition"] == cond and h["setting"] not in exempt],
                     f"within condition {cond!r}")
    if provenance_issues and not allow_mixed_provenance:
        raise SystemExit("cells were not produced under identical conditions (one-machine rule):\n  "
                         + "\n  ".join(provenance_issues)
                         + "\nSet allow_mixed_provenance: true in the manifest only for non-core checks.")

    # tasks: the clean baseline defines the task set; shifted conditions must use the same IDs
    clean_ref = cells[(clean, baseline)]
    tasks_by_suite = defaultdict(list)
    for t in sorted({e.task for e in clean_ref.values()}):
        tasks_by_suite[t[0]].append(t)
    all_tasks = {t for ts in tasks_by_suite.values() for t in ts}
    warnings = []
    for (cond, setting), eps in cells.items():
        extra = {e.task for e in eps.values()} - all_tasks
        if extra:
            raise SystemExit(f"{cond}/{setting} has tasks not in the clean baseline: {sorted(extra)[:5]} "
                             "(shifted episodes must log the base LIBERO suite and task_id)")
        missing = all_tasks - {e.task for e in eps.values()}
        if missing and setting == baseline:
            warnings.append(f"{cond}: {len(missing)} clean tasks have no episodes")

    # matched pairs per (condition, quantized setting)
    pairs = {}
    for (cond, setting), eps in cells.items():
        if setting == baseline:
            continue
        ref = cells[(cond, baseline)]
        pr = match(cond, setting, ref, eps, ref)
        if pr.n == 0:
            raise SystemExit(f"{cond}/{setting}: no episode matches the {baseline} cell")
        if pr.unmatched_q:
            warnings.append(f"{cond}/{setting}: {pr.unmatched_q} episodes have no {baseline} partner (dropped)")
        pairs[(cond, setting)] = pr

    # fallback sensitivity: per shifted condition, keep only tasks without fallback episodes, and
    # compare against the clean gap on those same tasks
    clean_for = {}       # (shifted cond, setting) -> clean pair to subtract
    covered = {}
    if covered_only:
        for cond in visual + language:
            fb = {e.task for (c, _), eps in cells.items() if c == cond for e in eps.values() if e.fallback}
            keep = all_tasks - fb
            covered[cond] = {"excluded_tasks": sorted(fb), "n_tasks": len(keep)}
            if not keep:
                raise SystemExit(f"{cond}: every task has a fallback episode; nothing left for the sensitivity analysis")
            for (c, setting) in list(pairs):
                if c != cond:
                    continue
                ref = {k: e for k, e in cells[(cond, baseline)].items() if e.task in keep}
                qq = {k: e for k, e in cells[(cond, setting)].items() if e.task in keep}
                pairs[(cond, setting)] = match(cond, setting, ref, qq, ref)
                if (clean, setting) in pairs:
                    cref = {k: e for k, e in cells[(clean, baseline)].items() if e.task in keep}
                    cq = {k: e for k, e in cells[(clean, setting)].items() if e.task in keep}
                    clean_for[(cond, setting)] = match(clean, setting, cref, cq, cref)

    rng = np.random.default_rng(seed)
    task_draws = draw_tasks(tasks_by_suite, B, rng)
    reps = {k: boot_gap(p, task_draws, rng, within=within) for k, p in sorted(pairs.items())}
    clean_reps = {k: boot_gap(p, task_draws, rng, within=within) for k, p in sorted(clean_for.items())}

    settings = sorted({s for (_, s) in pairs})
    result = {"config": {"B": B, "seed": seed, "bootstrap_within": within, "threshold_pp": threshold,
                         "baseline": baseline, "primary": primary,
                         "clean": clean, "visual": visual, "language": language,
                         "planned_visual_shifts": planned_visual, "covered_only": covered_only},
              "covered_tasks": covered,
              "headers": headers, "warnings": warnings + provenance_issues, "descriptive": [], "gaps": {}, "qrp": {}, "tests": {}}

    # descriptive: SR + Wilson, steps of successful episodes
    for (cond, setting), eps in sorted(cells.items()):
        k = sum(e.success for e in eps.values())
        n = len(eps)
        ci = wilson_ci(k, n)
        ok_steps = [e.steps for e in eps.values() if e.success]
        result["descriptive"].append({"condition": cond, "setting": setting, "n": n, "sr": 100 * k / n,
                                      "wilson_low": 100 * ci.low, "wilson_high": 100 * ci.high,
                                      "mean_steps_successful": float(np.mean(ok_steps)) if ok_steps else None})

    for (cond, setting), pr in sorted(pairs.items()):
        n10, n01 = pr.discordance()
        result["gaps"][f"{cond}/{setting}"] = {**summarize(pr.gap(), reps[(cond, setting)]), "n_matched": pr.n,
                                              "n10": n10, "n01": n01, "discordance": (n10 + n01) / pr.n}

    for setting in settings:
        if (clean, setting) not in pairs:
            continue
        g0, r0 = pairs[(clean, setting)].gap(), reps[(clean, setting)]
        q = {}
        for cond in visual + language:
            if (cond, setting) in pairs:
                if (cond, setting) in clean_for:   # covered-only: clean gap on the same tasks
                    c0, cr0 = clean_for[(cond, setting)].gap(), clean_reps[(cond, setting)]
                else:
                    c0, cr0 = g0, r0
                q[cond] = (pairs[(cond, setting)].gap() - c0, reps[(cond, setting)] - cr0)
        out = {c: summarize(*v) for c, v in q.items()}
        vis = [c for c in visual if c in q]
        if vis:
            pooled = (float(np.mean([q[c][0] for c in vis])), np.mean([q[c][1] for c in vis], axis=0))
            out["pooled_visual"] = summarize(*pooled)
            out["pooled_visual"]["shifts"] = vis
            out["pooled_visual"]["reduced_set"] = set(vis) != set(planned_visual)
            lang = [c for c in language if c in q]
            if lang:
                c = lang[0]
                out["visual_minus_language"] = summarize(pooled[0] - q[c][0], pooled[1] - q[c][1])
        result["qrp"][setting] = out

    # H1 (primary, single test) and the Holm family of secondary tests
    prim = result["qrp"].get(primary, {}).get("pooled_visual")
    if prim:
        result["tests"]["H1"] = {"setting": primary, **prim,
                                 "decision": h1_decision(prim["ci_low"], prim["ci_high"], threshold)}
    family = []
    w4 = manifest.get("secondary_setting", "W4")
    if result["qrp"].get(w4, {}).get("pooled_visual"):
        family.append((f"H2: {w4} pooled QRP", result["qrp"][w4]["pooled_visual"]["wald_p"]))
    if result["qrp"].get(primary, {}).get("visual_minus_language"):
        family.append((f"H3: {primary} visual - language", result["qrp"][primary]["visual_minus_language"]["wald_p"]))
    for c in visual:
        if c in result["qrp"].get(primary, {}):
            family.append((f"{primary} QRP {c}", result["qrp"][primary][c]["wald_p"]))
    if family:
        result["tests"]["holm"] = {"m": len(family), "results": holm(family)}
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(s: dict) -> str:
    return f"{s['estimate']:+.1f} [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]"


def to_markdown(r: dict) -> str:
    cfg = r["config"]
    scheme = "tasks, then units within task" if cfg["bootstrap_within"] == "resample" else "tasks only"
    L = [f"# QRP analysis ({cfg['B']:,} bootstrap replicates, seed {cfg['seed']}, resampling: {scheme})", ""]
    if r["warnings"]:
        L += ["**Warnings**", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    L += ["## Success rates (descriptive)", "", "| condition | setting | n | SR % | Wilson 95% | mean steps (successes) |",
          "|---|---|---|---|---|---|"]
    for d in r["descriptive"]:
        st = f"{d['mean_steps_successful']:.1f}" if d["mean_steps_successful"] is not None else "-"
        L.append(f"| {d['condition']} | {d['setting']} | {d['n']} | {d['sr']:.1f} | "
                 f"[{d['wilson_low']:.1f}, {d['wilson_high']:.1f}] | {st} |")
    L += ["", f"## Quantization gaps ({cfg['baseline']} - setting, pp; paired task-clustered 95% CI)", "",
          "| condition/setting | matched n | gap | n10 (only BF16 ok) | n01 (only Q ok) | discordance |",
          "|---|---|---|---|---|---|"]
    for k, g in r["gaps"].items():
        L.append(f"| {k} | {g['n_matched']} | {fmt(g)} | {g['n10']} | {g['n01']} | {100 * g['discordance']:.1f}% |")
    L += ["", "## QRP (gap under shift - clean gap, pp)", "", "| setting | contrast | QRP [95% CI] | SE | Wald p |",
          "|---|---|---|---|---|"]
    for setting, out in r["qrp"].items():
        for name, s in out.items():
            label = name + (" (reduced set)" if isinstance(s, dict) and s.get("reduced_set") else "")
            L.append(f"| {setting} | {label} | {fmt(s)} | {s['se']:.2f} | {s['wald_p']:.3f} |")
    if "H1" in r["tests"]:
        h = r["tests"]["H1"]
        L += ["", "## Primary hypothesis (H1)", "",
              f"Pooled visual QRP for {h['setting']}: **{fmt(h)}** points", "", f"Decision: **{h['decision']}**"]
    if r.get("sensitivity"):
        sv = r["sensitivity"]
        sscheme = "tasks, then units within task" if sv["bootstrap_within"] == "resample" else "tasks only"
        L += ["", f"## Sensitivity analysis (resampling: {sscheme})", "",
              "Same point estimates; only the bootstrap interval changes.", "",
              "| setting | pooled visual QRP [95% CI] |", "|---|---|"]
        for setting, out in sv["qrp"].items():
            if "pooled_visual" in out:
                L.append(f"| {setting} | {fmt(out['pooled_visual'])} |")
        if sv.get("H1"):
            L += ["", f"H1 under this scheme: **{sv['H1']['decision']}**"]
    if r.get("sensitivity_covered"):
        sc = r["sensitivity_covered"]
        L += ["", "## Sensitivity analysis: tasks without fallback episodes only", "",
              "Per shift, tasks with any fallback episode are dropped and the clean gap is computed on the "
              "same remaining tasks.", "", "| condition | excluded tasks | tasks kept |", "|---|---|---|"]
        for cond, cv in sc["covered_tasks"].items():
            ex = ", ".join(f"{s.replace('libero_', '')} {t}" for s, t in cv["excluded_tasks"]) or "none"
            L.append(f"| {cond} | {ex} | {cv['n_tasks']} |")
        L += ["", "| setting | pooled visual QRP [95% CI] |", "|---|---|"]
        for setting, out in sc["qrp"].items():
            if "pooled_visual" in out:
                L.append(f"| {setting} | {fmt(out['pooled_visual'])} |")
        if sc.get("H1"):
            L += ["", f"H1 on covered tasks only: **{sc['H1']['decision']}**"]
    if "holm" in r["tests"]:
        L += ["", f"## Secondary tests (Holm, m = {r['tests']['holm']['m']}, exploratory)", "",
              "| test | Wald p | Holm threshold | reject |", "|---|---|---|---|"]
        for name, t in r["tests"]["holm"]["results"].items():
            L.append(f"| {name} | {t['p']:.4f} | {t['threshold']:.4f} | {'yes' if t['reject'] else 'no'} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", help="YAML manifest listing the cells (see configs/qrp/)")
    ap.add_argument("--B", type=int, default=10_000, help="bootstrap replicates (proposal: 10,000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--threshold", type=float, default=5.0, help="practical threshold in pp (proposal: 5)")
    ap.add_argument("--within", choices=("resample", "fixed"), default=None,
                    help="override the manifest's bootstrap_within (default: fixed)")
    ap.add_argument("--allow-mixed-provenance", action="store_true",
                    help="warn instead of stopping when cells differ in host, GPU, commit, seed or episodes per task")
    ap.add_argument("--base-dir", default=None, help="paths in the manifest are relative to this (default: manifest dir)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args(argv)
    mpath = Path(a.manifest)
    manifest = yaml.safe_load(mpath.read_text())
    base = Path(a.base_dir) if a.base_dir else mpath.resolve().parent
    allow = True if a.allow_mixed_provenance else None
    r = analyse(manifest, base, a.B, a.seed, a.threshold, within=a.within, allow_mixed_provenance=allow)
    sens = manifest.get("sensitivity_within")
    if sens and sens != r["config"]["bootstrap_within"]:
        r2 = analyse(manifest, base, a.B, a.seed, a.threshold, within=sens, allow_mixed_provenance=allow)
        r["sensitivity"] = {"bootstrap_within": sens, "qrp": r2["qrp"], "H1": r2["tests"].get("H1")}
    if manifest.get("sensitivity_covered_only"):
        r3 = analyse(manifest, base, a.B, a.seed, a.threshold, within=a.within, allow_mixed_provenance=allow,
                     covered_only=True)
        r["sensitivity_covered"] = {"covered_tasks": r3["covered_tasks"], "qrp": r3["qrp"], "H1": r3["tests"].get("H1")}
    md = to_markdown(r)
    print(md)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(r, indent=2, default=float))
    if a.out_md:
        Path(a.out_md).write_text(md)


if __name__ == "__main__":
    main()
