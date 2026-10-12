from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_MAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "implmentation"
    / "miniDeepSeekv4"
    / "main.py"
)
_SPEC = importlib.util.spec_from_file_location("minideepseekv4_consumer", _MAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MAIN)


class _Tokenizer:
    def __init__(self):
        self.encoded = []
        self.decoded = None

    def encode(self, text, *, add_special_tokens):
        self.encoded.append((text, add_special_tokens))
        return type("Encoding", (), {"ids": [1, 2, 3]})()

    def decode(self, ids, *, skip_special_tokens):
        self.decoded = (list(ids), skip_special_tokens)
        return "decoded"


class _Model:
    def __init__(self):
        self.inputs = []

    def logits(self, input_ids):
        self.inputs.append(input_ids.tolist()[0])
        next_id = 8 if len(self.inputs) == 1 else 9
        logits = torch.zeros((1, input_ids.shape[1], 16))
        logits[0, -1, next_id] = 1
        return logits


def test_greedy_generation_preserves_token_ids_and_stops_at_eos():
    model = _Model()
    tokenizer = _Tokenizer()

    result = _MAIN.generate(
        model,
        tokenizer,
        "prompt",
        max_new_tokens=5,
        context=3,
        eos_token_id=9,
    )

    assert result == "decoded"
    assert tokenizer.encoded == [("prompt", False)]
    assert tokenizer.decoded == ([1, 2, 3, 8, 9], True)
    assert model.inputs == [[1, 2, 3], [2, 3, 8]]
