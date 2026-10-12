"""Training optimizers used by the isolated backend worker."""

from __future__ import annotations

import math
from typing import Any, Iterable


_MUON_STAGES = ((8, (3.4445, -4.7750, 2.0315)), (2, (2.0, -1.5, 0.5)))


def _orthogonalize(matrix: Any) -> Any:
    import torch

    # Newton-Schulz is most efficient when its Gram matrix uses the smaller side.
    transposed = matrix.shape[-2] > matrix.shape[-1]
    value = matrix.transpose(-2, -1) if transposed else matrix
    value = value.float()
    value = value / value.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    for count, (a, b, c) in _MUON_STAGES:
        for _ in range(count):
            gram = value @ value.transpose(-2, -1)
            value = a * value + (b * gram + c * (gram @ gram)) @ value
    return value.transpose(-2, -1) if transposed else value


class DeepSeekMuon:
    """Muon with paper Algorithm 1 and one state buffer per parameter."""

    def __init__(self, params: Iterable[Any], lr: float, weight_decay: float):
        import torch

        self._optimizer = torch.optim.Optimizer(
            list(params), {"lr": lr, "weight_decay": weight_decay, "momentum": 0.95}
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        import torch

        with torch.no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                decay = float(group["weight_decay"])
                momentum = float(group["momentum"])
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    state = self._optimizer.state[parameter]
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = state["momentum_buffer"] = torch.zeros_like(parameter)
                    buffer.mul_(momentum).add_(gradient)
                    nesterov = momentum * buffer + gradient

                    # Leading dimensions identify independent experts/groups.
                    matrices = nesterov.reshape(-1, nesterov.shape[-2], nesterov.shape[-1])
                    update = _orthogonalize(matrices)
                    update = update * (math.sqrt(max(nesterov.shape[-2:])) * 0.18)
                    update = update.reshape_as(parameter).to(dtype=parameter.dtype)
                    if decay:
                        parameter.mul_(1.0 - lr * decay)
                    parameter.add_(update, alpha=-lr)

    def state_dict(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state)


class CompositeOptimizer:
    """Dispatches AdamW exclusions and logically independent matrices to Muon."""

    def __init__(self, adamw: Any | None, muon: DeepSeekMuon | None):
        self.adamw = adamw
        self.muon = muon

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            *(self.adamw.param_groups if self.adamw is not None else []),
            *(self.muon.param_groups if self.muon is not None else []),
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.adamw is not None:
            self.adamw.step()
        if self.muon is not None:
            self.muon.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
            "muon": self.muon.state_dict() if self.muon is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"adamw", "muon"}:
            raise ValueError("Muon checkpoint has invalid optimizer state.")
        if (self.adamw is None) != (state["adamw"] is None) or (self.muon is None) != (state["muon"] is None):
            raise ValueError("Muon checkpoint optimizer groups do not match this model.")
        if self.adamw is not None:
            self.adamw.load_state_dict(state["adamw"])
        if self.muon is not None:
            self.muon.load_state_dict(state["muon"])


def build_optimizer(
    model: Any,
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> Any:
    import torch

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name != "deepseek-muon":
        raise ValueError(f"Unsupported training optimizer '{name}'.")

    adamw_params = []
    muon_params = []
    for parameter in model.parameters():
        if parameter.ndim >= 2 and getattr(parameter, "_nnm_optimizer", None) != "adamw":
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)
    adamw = torch.optim.AdamW(
        adamw_params, lr=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.95), eps=1e-20,
    ) if adamw_params else None
    muon = DeepSeekMuon(muon_params, lr=learning_rate, weight_decay=weight_decay) if muon_params else None
    if adamw is None and muon is None:
        raise ValueError("The model has no trainable parameters.")
    return CompositeOptimizer(adamw, muon)
