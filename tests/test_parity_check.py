"""Tests for scripts/parity_check.py (E0/E1 harness checks)."""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import vlaquantbench  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parity_check as pc  # noqa: E402

from vlaquantbench import ComponentMap  # noqa: E402

# VLAQuantBench's released logs, found through the editable install of the pinned clone
VQB = Path(vlaquantbench.__file__).resolve().parents[1]
R = VQB / "results/libero/libero_spatial/pi0"
needs_logs = pytest.mark.skipif(not (R / "rtn-BASELINE-e2e.jsonl").exists(),
                                reason="VLAQuantBench released logs not found (install the clone with pip install -e)")


class _StubAdapter:
    """Loaded-adapter stand-in: a tiny bf16 model with VE weights partly in float32,
    like the released pi0 checkpoint keeps some vision / norm weights in float32."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = nn.ModuleDict({
            "vision": nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, 8)),
            "llm": nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)),
            "head": nn.Linear(8, 7),
        }).to(torch.bfloat16)
        self.model["vision"][2].float()

    def component_map(self):
        cm = ComponentMap()
        cm.add("ve", "vision", self.model["vision"])
        cm.add("llm", "llm", self.model["llm"])
        cm.add("ah", "head", self.model["head"])
        return cm


def test_weights_noop_is_bit_identical_and_reports_dtypes():
    ok, lines = pc.weight_parity(_StubAdapter(), "NOOP", "ve")
    text = "\n".join(lines)
    assert ok, text
    assert "parameters changed: 0 of" in text
    assert "'bfloat16': 1" in text and "'float32': 1" in text


def test_weights_w4_is_detected_as_a_change():
    ok, lines = pc.weight_parity(_StubAdapter(), "W4", "ve")
    assert ok
    assert "changed 2 parameter tensors" in lines[-1]   # the two VE weight matrices; biases untouched


def test_weights_noop_fails_if_the_identity_is_broken(monkeypatch):
    import vlaquantbench.quant.apply as apply

    real = apply.fake_quant_weight
    monkeypatch.setattr(apply, "fake_quant_weight", lambda w, spec: real(w, spec) * 1.5)
    ok, lines = pc.weight_parity(_StubAdapter(), "NOOP", "ve")
    assert not ok and "NOOP changed parameters" in lines[-1]


@needs_logs
def test_rollouts_identical_files_pass_exact():
    ok, lines = pc.compare_rollouts(str(R / "rtn-BASELINE-e2e.jsonl"), str(R / "rtn-BASELINE-e2e.jsonl"), "exact", B=500)
    assert ok and lines[-1].startswith("PASS")


@needs_logs
def test_rollouts_w8_fails_exact_but_passes_statistical():
    ok, _ = pc.compare_rollouts(str(R / "rtn-BASELINE-e2e.jsonl"), str(R / "rtn-W8-e2e.jsonl"), "exact", B=500)
    assert not ok
    ok, lines = pc.compare_rollouts(str(R / "rtn-BASELINE-e2e.jsonl"), str(R / "rtn-W8-e2e.jsonl"), "statistical", B=500)
    assert ok and "n10 (only ref ok) = 6, n01 (only cell ok) = 5" in "\n".join(lines)


@needs_logs
def test_rollouts_whole_model_w4_is_flagged_statistically():
    ok, lines = pc.compare_rollouts(str(R / "rtn-BASELINE-e2e.jsonl"), str(R / "rtn-W4-e2e.jsonl"), "statistical", B=500)
    assert not ok
    assert any("outside the reference Wilson interval" in l for l in lines)


@needs_logs
def test_noop_run_that_quantized_nothing_fails(tmp_path):
    src = (R / "rtn-BASELINE-e2e.jsonl").read_text().splitlines()
    header = json.loads(src[0])
    header["preset"] = "NOOP"  # quant_report still says 0 layers everywhere
    bad = tmp_path / "rtn-NOOP-ve.jsonl"
    bad.write_text("\n".join([json.dumps(header)] + src[1:]) + "\n")
    ok, lines = pc.compare_rollouts(str(R / "rtn-BASELINE-e2e.jsonl"), str(bad), "exact", B=200)
    assert not ok
    assert any("quantized 0 layers" in l for l in lines)


@needs_logs
def test_exact_mode_requires_the_same_episode_set(tmp_path):
    src = (R / "rtn-BASELINE-e2e.jsonl").read_text().splitlines()
    part = tmp_path / "part.jsonl"
    part.write_text("\n".join(src[:150]) + "\n")
    ok, lines = pc.compare_rollouts(str(R / "rtn-BASELINE-e2e.jsonl"), str(part), "exact", B=200)
    assert not ok and any("same episode set" in l for l in lines)


@pytest.mark.parametrize("n10,n01", [(0, 0), (6, 5), (36, 18), (10, 0), (3, 9)])
def test_mcnemar_matches_scipy(n10, n01):
    stats = pytest.importorskip("scipy.stats")
    want = 1.0 if n10 + n01 == 0 else stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue
    assert pc.mcnemar_exact(n10, n01) == pytest.approx(want, rel=1e-9)
