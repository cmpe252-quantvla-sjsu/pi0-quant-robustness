"""Tests for scripts/power_update.py."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import power_update as p  # noqa: E402

CLEAN = {(s, t): 0.5 + 0.04 * t for s in p.SUITES for t in range(10)}   # 50% .. 86% by task


def test_shift_task_rates_match_each_suites_target_and_keep_task_order():
    target = {"libero_spatial": 0.6, "libero_object": 0.75, "libero_goal": 0.45}
    out = p.shift_task_rates(CLEAN, target)
    for s in p.SUITES:
        rates = [out[(s, t)] for t in range(10)]
        assert np.mean(rates) == pytest.approx(target[s], abs=1e-6)
        assert rates == sorted(rates)          # easier clean tasks stay easier under the shift


def inputs(a, b):
    shifts = {name: {"suite_rates": {}, "n": 0, "task_rates": CLEAN} for name in ("lighting", "background", "layout")}
    return {"clean": CLEAN, "shifts": shifts, "e1c": {"pairs": 0, "n10": 0, "n01": 0, "a": a, "b": b}}


def test_no_discordance_gives_zero_qrp_and_zero_se():
    with tempfile.TemporaryDirectory() as d:
        est, se = p.one_run(inputs(0.0, 0.0), 300, "E1c", 200, np.random.default_rng(0), Path(d))
    assert est == 0 and se == 0


def test_more_discordance_and_fewer_episodes_mean_a_larger_se():
    def mean_se(a, n):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            return np.mean([p.one_run(inputs(a, a), n, "E1c", 300, rng, Path(d))[1] for _ in range(20)])
    assert mean_se(0.10, 300) > mean_se(0.03, 300)
    assert mean_se(0.10, 150) > mean_se(0.10, 450)


def test_sizes_must_divide_over_30_tasks():
    with pytest.raises(SystemExit, match="not divisible by 30"):
        p.main(["--e1", ".", "--vqb", ".", "--sizes", "100"])
