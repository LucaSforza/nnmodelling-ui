from __future__ import annotations

import copy
import math

import torch

from backend.optimizers import CompositeOptimizer, DeepSeekMuon, _orthogonalize, build_optimizer


def _reference_orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    value = matrix.float() / matrix.float().norm().clamp_min(1e-7)
    for count, coefficients in ((8, (3.4445, -4.7750, 2.0315)), (2, (2.0, -1.5, 0.5))):
        a, b, c = coefficients
        for _ in range(count):
            gram = value @ value.T
            value = a * value + (b * gram + c * (gram @ gram)) @ value
    return value


def test_deepseek_muon_matches_paper_matrix_update():
    parameter = torch.nn.Parameter(torch.tensor([[0.2, -0.3], [0.4, 0.1]]))
    parameter.grad = torch.tensor([[0.3, 0.1], [-0.2, 0.5]])
    initial = parameter.detach().clone()
    gradient = parameter.grad.clone()
    lr, decay = 0.01, 0.1

    DeepSeekMuon([parameter], lr=lr, weight_decay=decay).step()

    direction = _reference_orthogonalize(gradient) * (math.sqrt(2) * 0.18)
    expected = initial * (1.0 - lr * decay) - lr * direction
    torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-6)


def test_grouped_parameter_slices_are_orthogonalized_independently():
    values = torch.tensor([
        [[1.0, 2.0], [0.0, 1.0]],
        [[0.1, 0.0], [3.0, 1.0]],
    ])
    grouped = torch.nn.Parameter(values.clone())
    grouped.grad = values.clone()
    DeepSeekMuon([grouped], lr=0.03, weight_decay=0).step()

    separate = []
    for matrix in values:
        parameter = torch.nn.Parameter(matrix.clone())
        parameter.grad = matrix.clone()
        DeepSeekMuon([parameter], lr=0.03, weight_decay=0).step()
        separate.append(parameter.detach())
    torch.testing.assert_close(grouped, torch.stack(separate), rtol=1e-6, atol=1e-6)


def test_muon_optimizer_dispatch_and_state_resume():
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.tensor([[0.2, 0.4], [0.1, -0.2]]))
            self.matrix._nnm_optimizer = "adamw"
            self.experts = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) / 8)
            self.vector = torch.nn.Parameter(torch.tensor([0.5, -0.25]))

    model = Mixed()
    optimizer = build_optimizer(model, "deepseek-muon", 0.01, 0.1)
    assert isinstance(optimizer, CompositeOptimizer)
    assert optimizer.muon is not None and optimizer.adamw is not None
    assert optimizer.muon.param_groups[0]["params"] == [model.experts]
    assert optimizer.adamw.param_groups[0]["params"] == [model.matrix, model.vector]

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()

    resumed_model = Mixed()
    resumed_model.load_state_dict(model.state_dict())
    resumed_optimizer = build_optimizer(resumed_model, "deepseek-muon", 0.01, 0.1)
    resumed_optimizer.load_state_dict(copy.deepcopy(state))
    for parameter in resumed_model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    resumed_optimizer.step()
    for actual, expected in zip(resumed_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_orthogonalize_returns_finite_fp32_direction():
    result = _orthogonalize(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16))
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
