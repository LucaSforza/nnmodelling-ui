from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "examples/implementation/llm/download_wheel.py"
SPEC = importlib.util.spec_from_file_location("llm_wheel_downloader", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def test_downloader_saves_only_a_hash_verified_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"verified wheel bytes"
    monkeypatch.setattr(downloader, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: io.BytesIO(payload))
    destination = tmp_path / "model.whl"

    downloader.download("http://example.test", destination)

    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".whl.download").exists()


def test_downloader_preserves_previous_file_on_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "model.whl"
    destination.write_bytes(b"previous verified wheel")
    monkeypatch.setattr(downloader, "SHA256", hashlib.sha256(b"expected").hexdigest())
    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: io.BytesIO(b"wrong artifact"))

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloader.download("http://example.test", destination)

    assert destination.read_bytes() == b"previous verified wheel"
    assert not destination.with_suffix(".whl.download").exists()
