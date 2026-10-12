from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / "examples/implementation/vae"
_SPEC = importlib.util.spec_from_file_location("vae_download_wheel", CONSUMER / "download_wheel.py")
assert _SPEC and _SPEC.loader
downloader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(downloader)


def artifact_for(payload: bytes) -> dict[str, str]:
    return {
        "job_id": "verified-job-id",
        "filename": "nnm_mnist_vae-0.1.0-py3-none-any.whl",
        "distribution": "nnm_mnist_vae",
        "module": "nnmodel_verified_job_id",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_download_hash_mismatch_preserves_existing_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "wheel.whl"
    destination.write_bytes(b"old verified wheel")

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def read(self, _size: int) -> bytes:
            nonlocal payload
            chunk, payload = payload, b""
            return chunk

    payload = b"unverified replacement"
    monkeypatch.setattr(downloader, "urlopen", lambda *_args, **_kwargs: Response())
    metadata = artifact_for(b"expected bytes")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloader.download("http://127.0.0.1:8765", destination, metadata)

    assert destination.read_bytes() == b"old verified wheel"
    assert list(tmp_path.iterdir()) == [destination]


def test_download_uses_exact_job_and_atomically_installs_verified_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified wheel bytes"
    metadata = artifact_for(payload)
    destination = tmp_path / metadata["filename"]
    requested: list[str] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def read(self, _size: int) -> bytes:
            nonlocal payload
            chunk, payload = payload, b""
            return chunk

    monkeypatch.setattr(downloader, "urlopen", lambda url, **_kwargs: (requested.append(url), Response())[1])
    downloader.download("http://backend/", destination, metadata)

    assert requested == [f"http://backend/v1/jobs/{metadata['job_id']}/wheel"]
    assert destination.read_bytes() == b"verified wheel bytes"
    assert list(tmp_path.iterdir()) == [destination]


def test_model_consumer_uses_public_operations_and_checks_shapes() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def inference(self, value: np.ndarray) -> np.ndarray:
            self.calls.append("inference")
            assert value.shape == (28, 28)
            return np.full((28, 28), 0.25, dtype=np.float32)

        def infer(self, value: np.ndarray) -> np.ndarray:
            self.calls.append("infer")
            return self.inference(value)

        def encode(self, value: np.ndarray) -> np.ndarray:
            self.calls.append("encode")
            assert value.shape == (28, 28)
            return np.zeros((1, 32), dtype=np.float32)

        def decode(self, latent: np.ndarray) -> np.ndarray:
            self.calls.append("decode")
            assert latent.shape == (1, 32)
            return np.full((28, 28), 0.25, dtype=np.float32)

    model = FakeModel()
    raw_image = np.zeros((28, 28), dtype=np.uint8)
    with pytest.MonkeyPatch.context() as scoped:
        scoped.syspath_prepend(str(CONSUMER))
        scoped.setitem(sys.modules, "download_wheel", downloader)
        main_spec = importlib.util.spec_from_file_location("_vae_consumer_main_test", CONSUMER / "main.py")
        assert main_spec and main_spec.loader
        consumer_main = importlib.util.module_from_spec(main_spec)
        main_spec.loader.exec_module(consumer_main)
        outputs = consumer_main.run_model(model, raw_image, [np.zeros((1, 32), dtype=np.float32)])

    assert outputs["encoded"].shape == (1, 32)
    assert outputs["reconstruction"].shape == (28, 28)
    assert outputs["prior_images"][0].shape == (28, 28)
    assert np.isfinite(outputs["inference"]).all()
    assert model.calls.count("encode") == 2
    assert model.calls.count("decode") == 2


def test_artifact_metadata_rejects_unfrozen_hash(tmp_path: Path) -> None:
    artifact = artifact_for(b"wheel")
    artifact["sha256"] = "PENDING_TRAINING_ARTIFACT"
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        downloader.load_artifact(path)
