from __future__ import annotations

import hashlib
import io

import pytest

from backend.remote_data import acquire, validate_remote_files


def _manifest(*, revision="a" * 40, path="corpus/train.bin", destination="train.bin", data=b"dataset"):
    return {"remoteFiles": {"repo": "owner/public-data", "revision": revision, "files": [
        {"path": path, "destination": destination, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}}


def test_remote_file_manifest_accepts_pinned_public_hf_records_in_order():
    assert validate_remote_files(_manifest()) == [{
        "path": "corpus/train.bin", "destination": "train.bin", "size": 7,
        "sha256": hashlib.sha256(b"dataset").hexdigest(), "repo": "owner/public-data",
        "revision": "a" * 40,
    }]


@pytest.mark.parametrize("changes", [
    {"revision": "main"},
    {"path": "../train.bin"},
    {"destination": "/tmp/train.bin"},
])
def test_remote_file_manifest_rejects_mutable_or_unsafe_references(changes):
    with pytest.raises(ValueError):
        validate_remote_files(_manifest(**changes))


def test_acquire_streams_hash_checked_file_and_reuses_verified_copy(tmp_path, monkeypatch):
    data = b"dataset payload"
    record = validate_remote_files(_manifest(data=data))[0]
    requests = []

    class Response(io.BytesIO):
        def geturl(self):
            return "https://huggingface.co/datasets/owner/public-data/file"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def open_url(request, timeout):
        requests.append((request.full_url, timeout))
        return Response(data)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    root = tmp_path / "dataset-data"
    acquire([record], root)
    assert (root / "train.bin").read_bytes() == data
    assert requests[0][0].startswith("https://huggingface.co/datasets/owner/public-data/resolve/")
    acquire([record], root)
    assert len(requests) == 1


def test_acquire_never_publishes_incomplete_file(tmp_path, monkeypatch):
    record = validate_remote_files(_manifest())[0]

    class Response(io.BytesIO):
        def geturl(self):
            return "https://huggingface.co/datasets/owner/public-data/file"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response(b"wrong"))
    root = tmp_path / "dataset-data"
    with pytest.raises(ValueError, match="verification"):
        acquire([record], root)
    assert not (root / "train.bin").exists()
    assert list(root.glob("*.download")) == []
