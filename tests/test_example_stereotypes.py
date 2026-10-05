import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "examples" / "mnist-vae" / "packages"


def load_resource(name: str) -> ModuleType:
    path = PACKAGES / name / "pytorch.py"
    spec = importlib.util.spec_from_file_location(f"vae_{name.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(name: str, latent_features: int = 3) -> torch.nn.Module:
    return load_resource(name).build({"latent_features": latent_features}, {}, object())


def test_project_packages_register_numerical_entrypoints_and_uv_dependency():
    for name in ("diagonal-gaussian", "reparameterize", "kl-divergence", "sigmoid", "total-loss"):
        directory = PACKAGES / name
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["id"].startswith("vae.")
        assert manifest["version"] == "0.1.0"
        assert manifest["entrypoints"]["pytorch"] == {"language": "python", "file": "pytorch.py"}
        assert (directory / "pyproject.toml").is_file()
        assert "nnmodelling-runtime>=0.1.0" in (directory / "pyproject.toml").read_text()


def test_diagonal_gaussian_preserves_order_shape_and_gradients():
    module = build("diagonal-gaussian")
    mean = torch.randn(4, 3, requires_grad=True)
    log_variance = torch.randn(4, 3, requires_grad=True)
    packed = module(mean, log_variance)
    assert packed.shape == (4, 2, 3)
    torch.testing.assert_close(packed[:, 0], mean)
    torch.testing.assert_close(packed[:, 1], log_variance)
    packed.sum().backward()
    torch.testing.assert_close(mean.grad, torch.ones_like(mean))
    torch.testing.assert_close(log_variance.grad, torch.ones_like(log_variance))
    with pytest.raises(ValueError, match="ordered"):
        module(mean)
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(4, 2), torch.randn(4, 3))
    with pytest.raises(ValueError, match="latent_features"):
        module(torch.randn(4, 4), torch.randn(4, 4))
    with pytest.raises(ValueError, match="dtype"):
        module(torch.randn(4, 3), torch.randn(4, 3, dtype=torch.float64))


def test_reparameterization_samples_in_train_and_uses_mean_in_eval():
    module = build("reparameterize")
    packed = torch.zeros(8, 2, 3, requires_grad=True)
    module.train()
    first = module(packed)
    second = module(packed)
    assert first.shape == (8, 3)
    assert not torch.equal(first, second)
    first.sum().backward()
    assert packed.grad is not None and torch.isfinite(packed.grad).all()
    module.eval()
    torch.testing.assert_close(module(packed), packed[:, 0])
    torch.testing.assert_close(module(packed), packed[:, 0])
    with pytest.raises(ValueError, match="one packed"):
        module(packed, packed)
    with pytest.raises(ValueError, match="latent_features"):
        module(torch.zeros(2, 2, 4))


def test_kl_is_per_sample_and_matches_formula_with_gradients():
    module = build("kl-divergence")
    mean = torch.tensor([[0.0, 1.0, -1.0], [0.5, 0.0, 0.25]], requires_grad=True)
    log_variance = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]], requires_grad=True)
    packed = torch.stack((mean, log_variance), dim=1)
    actual = module(packed)
    expected = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(dim=1)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert log_variance.grad is not None and torch.isfinite(log_variance.grad).all()
    with pytest.raises(ValueError, match="one packed"):
        module(packed, packed)
    with pytest.raises(ValueError, match="latent_features"):
        module(torch.zeros(2, 2, 4))


def test_sigmoid_preserves_values_shape_and_gradient():
    module = load_resource("sigmoid").build({}, {}, object())
    logits = torch.tensor([[-2.0, 0.0, 2.0]], requires_grad=True)
    output = module(logits)
    assert output.shape == logits.shape
    torch.testing.assert_close(output, torch.sigmoid(logits))
    output.sum().backward()
    assert logits.grad is not None and torch.all(logits.grad > 0)
    with pytest.raises(ValueError, match="exactly one"):
        module(logits, logits)
    with pytest.raises(ValueError, match="floating"):
        module(torch.ones(2, dtype=torch.int64))


def test_total_loss_adds_mean_kl_and_backpropagates_scalar_objective():
    module = load_resource("total-loss").build({}, {}, object())
    reconstruction = torch.tensor(0.25, requires_grad=True)
    kl = torch.tensor([0.1, 0.3, 0.5], requires_grad=True)
    loss = module(reconstruction, kl)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, torch.tensor(0.55))
    loss.backward()
    torch.testing.assert_close(reconstruction.grad, torch.tensor(1.0))
    torch.testing.assert_close(kl.grad, torch.full_like(kl, 1.0 / 3.0))
    with pytest.raises(ValueError, match="reconstruction MSE then"):
        module(reconstruction)
    with pytest.raises(ValueError, match="scalar reconstruction"):
        module(reconstruction.reshape(1), kl)
    with pytest.raises(ValueError, match="same floating"):
        module(reconstruction, kl.to(torch.float64))
