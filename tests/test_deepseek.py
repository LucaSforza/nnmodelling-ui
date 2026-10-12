import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "examples" / "models" / "miniDeepSeekv4" / "packages"
DATASET = ROOT / "examples" / "models" / "miniDeepSeekv4" / "datasets" / "dsv4.fineweb-edu-1.0.0"


def load_resource(name: str) -> ModuleType:
    path = PACKAGES / f"dsv4.{name}-1.0.0" / "pytorch.py"
    spec = importlib.util.spec_from_file_location(f"dsv4_{name.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(name: str, parameters: dict, services=None):
    return load_resource(name).build(parameters, {}, services or object())


def assert_finite_gradients(module: torch.nn.Module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mhc_sinkhorn_is_doubly_stochastic_nonexpansive_and_differentiable():
    module = build("mhc", {"dim": 32, "streams": 4, "iterations": 20})
    state = torch.randn(2, 7, 4, 32, requires_grad=True)
    hidden, post, matrix = module.pre(state)
    torch.testing.assert_close(matrix.sum(-1), torch.ones_like(matrix.sum(-1)), atol=1e-5, rtol=0)
    torch.testing.assert_close(matrix.sum(-2), torch.ones_like(matrix.sum(-2)), atol=2e-4, rtol=0)
    assert torch.linalg.matrix_norm(matrix, ord=2).max() <= 1.00001
    output = module.post(state, hidden, post, matrix)
    assert output.shape == state.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert_finite_gradients(module)
    assert state.grad is not None and torch.isfinite(state.grad).all()


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_hybrid_attention_forward_backward_and_finite_outputs(ratio: int):
    module = build("attention", {
        "dim": 32, "heads": 2, "head_dim": 64, "q_rank": 16,
        "o_groups": 1, "o_rank": 16, "ratio": ratio, "index_heads": 2,
        "index_dim": 8, "topk": 5, "window": 128,
    })
    x = torch.randn(1, 140, 32, requires_grad=True)
    result = module(x)
    assert result["out"].shape == x.shape and result["loss"].ndim == 0
    assert torch.isfinite(result["out"]).all() and torch.isfinite(result["loss"])
    (result["out"].square().mean() + result["loss"]).backward()
    assert_finite_gradients(module)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    if ratio == 4:
        # The differentiable distillation term trains the sparse lightning indexer.
        assert module.index_queries.weight.grad is not None
        assert module.index_weights.weight.grad is not None
        assert module.index_compressor.values.weight.grad is not None


@pytest.mark.parametrize("ratio,changed_at,checked_before", [(4, 20, 20), (128, 139, 139)])
def test_compressed_attention_does_not_leak_future_tokens(ratio: int, changed_at: int, checked_before: int):
    torch.manual_seed(4)
    module = build("attention", {
        "dim": 32, "heads": 2, "head_dim": 64, "q_rank": 16,
        "o_groups": 1, "o_rank": 16, "ratio": ratio, "index_heads": 2,
        "index_dim": 8, "topk": 5, "window": 128,
    }).eval()
    tokens = torch.randn(1, 140, 32)
    changed = tokens.clone()
    changed[:, changed_at:] += torch.randn_like(changed[:, changed_at:]) * 5
    with torch.no_grad():
        before = module(tokens)["out"]
        after = module(changed)["out"]
    torch.testing.assert_close(before[:, :checked_before], after[:, :checked_before], atol=0, rtol=0)


@pytest.mark.parametrize("hash_routing", [False, True])
def test_moe_routes_distinct_experts_and_backpropagates(hash_routing: bool):
    module = build("moe", {
        "dim": 32, "hidden": 32, "experts": 4, "topk": 2,
        "hash_routing": hash_routing, "route_scale": 1.0,
    })
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    token_ids = torch.arange(16).reshape(2, 8)
    result = module(hidden, token_ids)
    assert result["out"].shape == hidden.shape
    assert torch.isfinite(result["out"]).all() and torch.isfinite(result["loss"])
    result["out"].square().mean().add(result["loss"]).backward()
    assert module.router.weight.grad is not None
    assert torch.isfinite(module.router.weight.grad).all()
    assert any(expert.gate.weight.grad is not None for expert in module.experts)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    if hash_routing:
        base = (token_ids.flatten() * 2654435761 + 1013904223) % 4
        chosen = (base[:, None] + torch.arange(2)) % 4
        assert torch.all(chosen[:, 0] != chosen[:, 1])


class ComponentServices:
    """Small service shim matching the resource builder API."""

    def build_stereotype(self, reference):
        name = reference["id"].removeprefix("dsv4.")
        return build(name, reference["parameters"], self)

    def build_subflow(self):
        return ComponentBlock(self.profile, self)

    def __init__(self, profile=None):
        self.profile = profile or tiny_profile()


def tiny_profile():
    return {
        "dim": 32, "vocab_size": 257, "streams": 4,
        "heads": 2, "head_dim": 64, "q_rank": 16, "o_groups": 1,
        "o_rank": 16, "hidden": 32, "experts": 4, "active_experts": 2,
        "index_heads": 2, "index_dim": 8, "topk": 2, "window": 128,
        "iterations": 20, "ratio": 4,
    }


class ComponentBlock(torch.nn.Module):
    """A real resource-level residual block assembled in graph order."""

    def __init__(self, p, services):
        super().__init__()
        self.p = p
        self.pre_attention = build("mhc-pre", {**p, "iterations": p["iterations"]}, services)
        self.select_attention = build("mhc-input", p, services)
        self.norm_attention = build("rmsnorm", {"dim": p["dim"]}, services)
        self.attention = build("attention", p, services)
        self.merge_attention = build("mhc-merge", p, services)
        self.pre_moe = build("mhc-pre", {**p, "iterations": p["iterations"]}, services)
        self.select_moe = build("mhc-input", p, services)
        self.norm_moe = build("rmsnorm", {"dim": p["dim"]}, services)
        self.moe = build("hash-moe", {
            "dim": p["dim"], "hidden": p["hidden"], "experts": p["experts"],
            "topk": p["topk"], "route_scale": 1.0,
        }, services)
        self.merge_moe = build("mhc-merge", p, services)

    def forward(self, state, token_ids=None):
        if token_ids is None:
            token_ids = torch.zeros(state.shape[:2], device=state.device, dtype=torch.long)
        packed = self.pre_attention(state)
        normalized = self.norm_attention(self.select_attention(packed))
        attention = self.attention(normalized)
        state = self.merge_attention(packed, attention["out"])
        packed = self.pre_moe(state)
        normalized = self.norm_moe(self.select_moe(packed))
        moe = self.moe(normalized, token_ids)
        state = self.merge_moe(packed, moe["out"])
        return {"out": state, "loss": attention["loss"] + moe["loss"]}


class TinyEditableV4(torch.nn.Module):
    """Test-only assembly of visible resources; no model package builds a decoder."""

    def __init__(self, p):
        super().__init__()
        services = ComponentServices(p)
        self.embedding = build("embedding", p, services)
        self.expand = build("stream-expand", p, services)
        self.main = ComponentBlock(p, services)
        self.mtp_shift = build("mtp-shift", {"mode": "embedding"}, services)
        self.mtp_fusion = build("mtp-fusion", {}, services)
        self.mtp = ComponentBlock(p, services)
        self.main_collapse = build("mhc-head", p, services)
        self.mtp_collapse = build("mhc-head", p, services)
        self.main_norm = build("rmsnorm", {"dim": p["dim"]}, services)
        self.mtp_norm = build("rmsnorm", {"dim": p["dim"]}, services)
        self.feature_stack = build("feature-stack", {}, services)
        self.head = build("vocab-head", p, services)
        self.loss_sum = build("loss-sum", {}, services)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        state = self.expand(embedded)
        main = self.main(state, tokens)
        main_features = self.main_norm(self.main_collapse(main["out"]))

        shifted_embedding = self.mtp_shift(embedded)
        previous_state = main["out"][:, :-1]
        mtp_state = self.mtp_fusion(shifted_embedding, previous_state)
        mtp = self.mtp(mtp_state, tokens[:, 1:])
        mtp_features = self.mtp_norm(self.mtp_collapse(mtp["out"]))
        features = self.feature_stack(main_features, mtp_features)
        logits = self.head(features)
        auxiliary = self.loss_sum(main["loss"], mtp["loss"])
        return {"out": logits, "loss": auxiliary}


def test_repeat_builds_independent_composed_resource_bodies():
    profile = tiny_profile()
    services = ComponentServices(profile)
    repeated = build("repeat", {"times": 2}, services)
    assert len(repeated.bodies) == 2
    first, second = repeated.bodies
    first_parameters = list(first.parameters())
    second_parameters = list(second.parameters())
    assert first_parameters and second_parameters
    assert {id(parameter) for parameter in first_parameters}.isdisjoint(
        id(parameter) for parameter in second_parameters
    )
    state = torch.randn(1, 5, profile["streams"], profile["dim"])
    tokens = torch.arange(5).reshape(1, 5)
    result = repeated(state)
    assert result["out"].shape == state.shape
    assert torch.isfinite(result["out"]).all() and torch.isfinite(result["loss"])


def test_packed_mhc_components_match_original_pre_and_post_equations():
    p = tiny_profile()
    services = ComponentServices(p)
    original = build("mhc", p, services)
    pre = build("mhc-pre", p, services)
    select = build("mhc-input", p, services)
    merge = build("mhc-merge", p, services)
    state = torch.randn(2, 6, p["streams"], p["dim"], requires_grad=True)
    pre.hc.load_state_dict(original.state_dict())
    hidden, post, matrix = original.pre(state)
    packed = pre(state)
    torch.testing.assert_close(select(packed), hidden)
    replacement = torch.randn_like(hidden, requires_grad=True)
    expected = original.post(state, replacement, post, matrix)
    actual = merge(packed, replacement)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert_finite_gradients(pre)
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert replacement.grad is not None and torch.isfinite(replacement.grad).all()


def test_tiny_editable_composition_parameter_formula_matches_registered_parameters():
    p = tiny_profile()
    model = TinyEditableV4(p)
    n, d, e, h = p["streams"], p["dim"], p["experts"], p["hidden"]
    mhc = n * (n + 2) * n * d + n * (n + 2) + 3
    mhc_head = n * n * d + n + 1
    attention = (
        d * p["q_rank"] + p["q_rank"]
        + p["q_rank"] * p["heads"] * p["head_dim"]
        + d * p["head_dim"] + p["head_dim"]
        + p["heads"] * p["head_dim"] * p["o_rank"]
        + p["o_groups"] * p["o_rank"] * d + p["heads"]
    )
    # Ratio-four compressor and sparse indexer each own their visible projections.
    attention += 2 * d * (2 * p["head_dim"]) + 4 * (2 * p["head_dim"]) + p["head_dim"]
    attention += p["q_rank"] * p["index_heads"] * p["index_dim"] + d * p["index_heads"]
    attention += 2 * d * (2 * p["index_dim"]) + 4 * (2 * p["index_dim"]) + p["index_dim"]
    moe = d * e + 3 * d * h * (e + 1)
    one_block = 2 * mhc + 2 * d + attention + moe
    expected = (
        p["vocab_size"] * d + p["vocab_size"] * d
        + 2 * one_block + 2 * mhc_head + 2 * d
    )
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert expected == actual


def test_mtp_shapes_shared_embedding_and_head_receive_objective_gradients():
    torch.manual_seed(12)
    model = TinyEditableV4(tiny_profile())
    tokens = torch.randint(0, 257, (2, 7))
    result = model(tokens)
    assert result["out"].shape == (2, 7, 2, 257)
    assert torch.isfinite(result["out"]).all() and torch.isfinite(result["loss"])
    objective = build("objective", {})
    loss = objective(result["out"], result["loss"], tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding.weight.grad is not None
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()
    assert torch.isfinite(model.head.weight.grad).all()
    assert sum(parameter is model.embedding.weight for parameter in model.parameters()) == 1
    assert sum(parameter is model.head.weight for parameter in model.parameters()) == 1


@pytest.mark.parametrize("changed_at", [20, 139])
def test_editable_main_logits_are_causal_across_csa_and_hca(changed_at: int):
    torch.manual_seed(9)
    profile = tiny_profile()
    model = TinyEditableV4(profile).eval()
    tokens = torch.randint(0, profile["vocab_size"], (1, 140))
    changed = tokens.clone()
    changed[:, changed_at:] = (changed[:, changed_at:] + 31) % profile["vocab_size"]
    with torch.no_grad():
        before = model(tokens)["out"][:, :, 0, :]
        after = model(changed)["out"][:, :, 0, :]
    torch.testing.assert_close(before[:, :changed_at], after[:, :changed_at], atol=0, rtol=0)


def test_one_token_mtp_path_is_empty_without_breaking_inference():
    profile = tiny_profile()
    model = TinyEditableV4(profile).eval()
    tokens = torch.randint(0, profile["vocab_size"], (2, 1))
    with torch.no_grad():
        result = model(tokens)
    assert result["out"].shape == (2, 1, 2, profile["vocab_size"])
    torch.testing.assert_close(result["out"][:, 0, 1, :], torch.zeros_like(result["out"][:, 0, 1, :]))
    assert torch.isfinite(result["out"]).all() and torch.isfinite(result["loss"])


def test_next_token_selects_main_logits_and_original_tokenizer_covers_128k_ids():
    projection = build("next-token", {})
    logits = torch.randn(2, 3, 2, 17, requires_grad=True)
    selected = projection(logits)
    assert selected.shape == (2, 3, 17)
    torch.testing.assert_close(selected, logits[:, :, 0, :])
    selected.sum().backward()
    torch.testing.assert_close(logits.grad[:, :, 0, :], torch.ones_like(logits.grad[:, :, 0, :]))
    torch.testing.assert_close(logits.grad[:, :, 1, :], torch.zeros_like(logits.grad[:, :, 1, :]))

    tokenizer = json.loads((DATASET / "tokenizer.json").read_text())
    ids = set(tokenizer["model"]["vocab"].values())
    ids.update(item["id"] for item in tokenizer.get("added_tokens", []))
    assert ids == set(range(129280))
