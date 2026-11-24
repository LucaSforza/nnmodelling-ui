"""The rehearsal profiles' PyTorch code gives the outcome each profile claims (docs/knowledge/testing/expert-study.md)."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "expert_study"
sys.path.insert(0, str(TOOLS))

import simulate_lab as sim  # noqa: E402
import study_checks  # noqa: E402


def define(*sources):
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    for source in sources:
        exec(source, namespace)  # noqa: S102 - code typed by the simulated experts
    return namespace


def test_t1_code_passes():
    ns = define(sim.T1_MODEL, sim.T1_LOSS)
    assert study_checks.evaluate("T1", ns["MLP"], ns["compute_loss"]).passed


@pytest.mark.parametrize("variant", sorted(sim.T2_CODE))
def test_t2_variants_match_their_profile(variant):
    ns = define(sim.T2_CODE[variant], sim.T2_LOSS)
    report = study_checks.evaluate("T2", ns["MultiHeadRowAttentionClassifier"], ns["compute_loss"])
    assert report.passed is sim.EXPECTED_T2[variant]
    if variant == "wrong_scale":
        assert any("attention scale" in c.name and not c.passed for c in report.checks)


def test_profiles_are_consistent():
    groups = {pid: p["group"] for pid, p in sim.PROFILES.items()}
    assert all(900 < pid < 1000 for pid in groups)  # never a real participant number
    assert sorted(groups.values()).count("expert") == 4 and len(groups) == 10
    for pid, p in sim.PROFILES.items():
        if p["group"] == "expert":
            assert p["order"] in ("nnm-first", "pytorch-first") and p["t2_code"] in sim.T2_CODE
    assert sorted(p["order"] for p in sim.PROFILES.values() if p["group"] == "expert") == \
        ["nnm-first", "nnm-first", "pytorch-first", "pytorch-first"]
