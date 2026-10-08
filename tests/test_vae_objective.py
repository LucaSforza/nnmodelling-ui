from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LOSS_PATH = ROOT / "examples/models/mnist-vae/packages/total-loss/pytorch.py"
_SPEC = importlib.util.spec_from_file_location("mnist_vae_total_loss", LOSS_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_total_loss_scales_mean_pixel_mse_and_preserves_kl_gradient() -> None:
    reconstruction_mse = torch.tensor(2.5, requires_grad=True)
    per_sample_kl = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)

    loss = _MODULE.TotalLoss()(reconstruction_mse, per_sample_kl)
    loss.backward()

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(784 * 2.5 + 3.0))
    torch.testing.assert_close(reconstruction_mse.grad, torch.tensor(784.0))
    torch.testing.assert_close(per_sample_kl.grad, torch.full((3,), 1 / 3))
    assert torch.isfinite(reconstruction_mse.grad).all()
    assert torch.isfinite(per_sample_kl.grad).all()
