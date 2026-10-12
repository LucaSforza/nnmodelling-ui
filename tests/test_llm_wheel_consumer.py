from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile
import os
import subprocess
import sysconfig

import pytest


SCRIPT = Path(__file__).parents[1] / "examples/implementation/llm/download_wheel.py"
SPEC = importlib.util.spec_from_file_location("llm_wheel_downloader", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def _wheel_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nnm_tiny_decoder_llm-0.1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: nnm_tiny_decoder_llm\nVersion: 0.1.0\n\n")
    return buffer.getvalue()


def test_downloader_saves_a_hash_verified_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _wheel_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: io.BytesIO(payload))
    destination = tmp_path / "model.whl"

    received = downloader.download("http://example.test", destination, sha256=digest)

    assert received == digest
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".whl.download").exists()


def test_downloader_accepts_selected_job_without_hash_and_encodes_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _wheel_bytes()
    requests = []

    def open_url(url, timeout):
        requests.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(downloader, "urlopen", open_url)
    destination = tmp_path / "selected.whl"

    downloader.download("http://example.test", destination, "job with space", None)

    assert requests == ["http://example.test/v1/jobs/job%20with%20space/wheel"]
    assert destination.read_bytes() == payload


def test_downloader_preserves_previous_file_on_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "model.whl"
    destination.write_bytes(b"previous verified wheel")
    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: io.BytesIO(_wheel_bytes()))

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloader.download("http://example.test", destination, sha256=hashlib.sha256(b"expected").hexdigest())

    assert destination.read_bytes() == b"previous verified wheel"
    assert not destination.with_suffix(".whl.download").exists()


def test_downloader_rejects_a_different_distribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("other-0.1.dist-info/METADATA", "Name: other\nVersion: 0.1\n")
    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: io.BytesIO(payload.getvalue()))
    destination = tmp_path / "model.whl"

    with pytest.raises(ValueError, match="distribution identity"):
        downloader.download("http://example.test", destination, "new-job", None)

    assert not destination.exists()


def test_installer_discovers_the_import_package_in_a_historical_wheel(tmp_path: Path) -> None:
    import base64
    import csv
    entries = {
        "nnmodel_historical_job/__init__.py": b"from .model import Model\n",
        "nnmodel_historical_job/model.py": b"class Model:\n    marker = 'historical'\n    def __init__(self, weights_path=None): pass\n",
        "nnm_tiny_decoder_llm-0.1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: nnm_tiny_decoder_llm\nVersion: 0.1.0\n\n",
        "nnm_tiny_decoder_llm-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record_path = "nnm_tiny_decoder_llm-0.1.0.dist-info/RECORD"
    rows = []
    for name, payload in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        rows.append((name, f"sha256={digest}", str(len(payload))))
    rows.append((record_path, "", ""))
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    entries[record_path] = record.getvalue().encode()
    wheel = tmp_path / downloader.FILENAME
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    environment = tmp_path / "legacy-venv"
    subprocess.run(["uv", "venv", "--system-site-packages", str(environment)], check=True, capture_output=True, text=True)
    python = environment / "bin" / "python"
    installer = SCRIPT.with_name("install_wheel.py")
    env = {**os.environ, "PYTHONPATH": sysconfig.get_paths()["purelib"]}
    subprocess.run([str(python), str(installer), str(wheel)], check=True, capture_output=True, text=True, env=env)
    main_script = SCRIPT.with_name("main.py")
    result = subprocess.run(
        [str(python), "-c", "import importlib.util, runpy, sys; assert importlib.util.find_spec('nnm_tiny_decoder_llm') is None; sys.argv = [sys.argv[1], '--tokens', '0']; runpy.run_path(sys.argv[0], run_name='__main__')", str(main_script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ROMEO:")
