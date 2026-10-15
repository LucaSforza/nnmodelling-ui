from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.prepare_worker import PreparationError, WorkerPreparation, _resource_dependencies
from backend.user_config import validate_profile


def _profile() -> dict:
    return validate_profile({
        "backend": {"executor": "slurm", "container_runtime": "docker"},
        "slurm": {
            "host": "cluster", "root": "/cluster/jobs", "image": "/cluster/images/worker.sif",
            "partition": "cpu", "cpus": 2, "memory": "4G", "time": "00:30:00",
            "gpus": 0, "ssh_executable": "ssh",
        },
    })


def _repository(root: Path) -> Path:
    (root / "backend").mkdir(parents=True)
    (root / "python/nnmodelling-runtime/src/nnmodelling_runtime").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[dependency-groups]\ndev = ["pytest>=8"]\nexamples = ["Pillow>=11,<13"]\n', encoding="utf-8"
    )
    (root / "uv.lock").write_text("frozen workspace lock", encoding="utf-8")
    (root / ".dockerignore").write_text(".git\n.venv\n", encoding="utf-8")
    (root / "backend/Dockerfile").write_text(
        "ENV UV_PROJECT_ENVIRONMENT=/opt/nnmodelling/venv\nRUN uv sync --frozen --group examples\n", encoding="utf-8"
    )
    (root / "backend/app.py").write_text("source", encoding="utf-8")
    (root / "backend/.env").write_text("SECRET=should-not-copy", encoding="utf-8")
    (root / "backend/private.pem").write_text("private key", encoding="utf-8")
    (root / "python/nnmodelling-runtime/pyproject.toml").write_text('[project]\nname="runtime"\n', encoding="utf-8")
    (root / "python/nnmodelling-runtime/src/nnmodelling_runtime/__init__.py").write_text("", encoding="utf-8")
    (root / ".aws").mkdir()
    (root / ".aws/credentials").write_text("credentials", encoding="utf-8")
    return root


class FakePreparation(WorkerPreparation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands: list[list[str]] = []
        self.remote_commands: list[str] = []
        self.uploads: list[tuple[str, int]] = []
        self.context: Path | None = None
        self.context_files: dict[str, str | bool] = {}
        self.submitted = False
        self.fail_at: str | None = None
        self.image_hash = hashlib.sha256(b"verified sif").hexdigest()

    def _prerequisites(self):
        return

    def _prepare_remote_stage(self):
        self.remote_stage = "/cluster/jobs/.nnmodelling-worker-test"
        self._ssh("mkdir private stage")

    def _command(self, args, *, timeout):
        self.commands.append(args)
        executable = Path(args[0]).name
        if [executable, *args[1:2]] == ["docker", "--version"]:
            return "Docker version 27.0"
        if executable == "uv" and args[1:2] == ["lock"]:
            self.context = Path(args[args.index("--project") + 1])
            return ""
        if executable == "docker" and args[1:2] == ["save"]:
            output = Path(args[args.index("--output") + 1])
            output.write_bytes(b"docker archive bytes")
        if executable == "docker" and args[1:2] == ["build"]:
            context = Path(args[-1])
            self.context_files = {
                "pyproject": (context / "pyproject.toml").read_text(encoding="utf-8"),
                "dockerfile": (context / "Dockerfile").read_text(encoding="utf-8"),
                "env": (context / "backend/.env").exists(),
                "key": (context / "backend/private.pem").exists(),
                "lock": (context / "uv.lock").read_text(encoding="utf-8"),
            }
        return ""

    def _ssh(self, command, *, timeout=60, input_data=None, ignore_cancel=False):
        self.remote_commands.append(command)
        if self.fail_at == "ssh":
            raise PreparationError("SSH transport failed")
        if self.fail_at == "scancel" and command.startswith("scancel "):
            raise PreparationError("scancel transport failed")
        if "result.sha256" in command:
            return self.image_hash
        return ""

    def _stream_upload(self, source, destination, label):
        size = source.stat().st_size
        self.uploads.append((label, size))
        self.emit("transfer", f"Transferring {label}.", current=size // 2, total=size)
        if self.fail_at == "upload":
            raise PreparationError("Cluster transfer failed after a partial upload.")
        self.emit("transfer", f"Transferring {label}.", current=size, total=size)
        return size, hashlib.sha256(source.read_bytes()).hexdigest()

    def _verify_remote_archive(self, archive):
        if self.fail_at == "hash":
            raise PreparationError("Transferred worker archive SHA256 does not match the local archive.")

    def _submit_conversion(self, archive_size):
        self.submitted = True
        if self.fail_at == "scheduler":
            raise PreparationError("SIF conversion allocation 42 ended in FAILED (1:0).")
        self.scheduler_id = "42"
        (self.local_stage / "scheduler.id").write_text("42\n", encoding="ascii")
        self._wait_for_conversion("42")

    def _wait_for_conversion(self, job_id):
        return


def _events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]


def test_prepare_uses_frozen_context_progress_and_immutable_hash_name(tmp_path, capsys):
    repo = _repository(tmp_path / "repo")
    project = tmp_path / "project"
    (project / "packages/custom").mkdir(parents=True)
    (project / "model.json").write_text(json.dumps({"manifest": {"customPackages": [
        {"path": "packages/custom"}], "customDatasets": []}}), encoding="utf-8")
    (project / "packages/custom/pyproject.toml").write_text('[project]\ndependencies=["requests>=2.31"]\n', encoding="utf-8")
    prep = FakePreparation(_profile(), repository=repo, project=project)

    image = prep.run()

    assert image == f"/cluster/images/worker-{prep.image_hash}.sif"
    assert prep.submitted
    assert ["uv", "lock", "--project", str(prep.context)] in prep.commands
    assert any(Path(command[0]).name == "docker" and command[1:2] == ["build"] for command in prep.commands)
    assert any("--format" not in command for command in prep.commands if Path(command[0]).name == "docker" and command[1:2] == ["save"])
    context_pyproject = prep.context_files["pyproject"]
    assert "requests>=2.31" in context_pyproject
    assert "frozen workspace lock" == prep.context_files["lock"]
    assert "UV_PROJECT_ENVIRONMENT=/opt/nnmodelling/venv" in prep.context_files["dockerfile"]
    assert not prep.context_files["env"]
    assert not prep.context_files["key"]
    assert not (prep.context / ".aws/credentials").exists()
    assert {label for label, _ in prep.uploads} == {"source", "worker image"}
    events = _events(capsys)
    assert events[-1]["phase"] == "success" and events[-1]["image"] == image
    assert any(event["phase"] == "transfer" and event["current"] > 0 for event in events)


def test_check_mode_probes_without_build_or_image_mutation(tmp_path, capsys):
    repo = _repository(tmp_path / "repo")
    prep = FakePreparation(_profile(), repository=repo, check_only=True)
    prep._probe_configured = lambda: prep.remote_commands.append("training-layout-probe")

    assert prep.run() is None

    assert not prep.submitted
    assert not any(Path(command[0]).name == "docker" and command[1:2] == ["build"] for command in prep.commands)
    assert not any(Path(command[0]).name == "uv" and command[1:2] == ["lock"] for command in prep.commands)
    assert "training-layout-probe" in prep.remote_commands
    event = _events(capsys)[-1]
    assert event["phase"] == "success" and "image" not in event


def test_check_probe_supports_legacy_path_and_checks_declared_distributions(tmp_path):
    prep = FakePreparation(_profile(), repository=_repository(tmp_path), check_only=True)
    prep.project_dependencies = ["Pillow>=11"]

    command = prep._probe_command(prep.slurm["image"], "/cluster/jobs/source", legacy=True)

    assert "else exec python3" in command
    assert "'Pillow'" in command
    assert "importlib.metadata.distribution(name)" in command
    assert subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True).returncode == 0


def test_conversion_uses_archive_without_fakeroot_and_integer_disk_bytes(tmp_path):
    prep = WorkerPreparation(_profile(), repository=_repository(tmp_path))
    prep.remote_stage = "/cluster/jobs/.nnmodelling-worker-test"
    prep.local_stage = tmp_path
    commands = []
    def fake_ssh(command, **kwargs):
        commands.append((command, kwargs))
        return "42\n" if "scheduler.id" in command and "cat --" in command else ""

    prep._ssh = fake_ssh
    prep._wait_for_conversion = lambda _job_id: None

    prep._submit_conversion(8)

    scripts = [kwargs["input_data"].decode() for _, kwargs in commands if kwargs.get("input_data")]
    script = next(value for value in scripts if "#SBATCH --partition" in value)
    assert 'singularity build "$sif" "docker-archive:$archive"' in script
    assert "printf \"%.0f\", $4 * 1024" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0


@pytest.mark.parametrize("failure", ["ssh", "upload", "hash", "scheduler"])
def test_failed_preparation_reports_error_and_preserves_configured_image(tmp_path, capsys, failure):
    repo = _repository(tmp_path / "repo")
    previous_image = "/cluster/images/worker.sif"
    prep = FakePreparation(_profile(), repository=repo)
    prep.fail_at = failure

    with pytest.raises(PreparationError):
        prep.run()

    events = _events(capsys)
    assert events[-1]["phase"] == "error"
    assert events[-1]["error"]
    assert prep.slurm["image"] == previous_image


def test_cancel_recovers_only_the_owned_allocation_receipt(tmp_path):
    prep = FakePreparation(_profile(), repository=tmp_path)
    prep.local_stage = tmp_path
    prep.remote_stage = "/cluster/jobs/.nnmodelling-worker-owned"
    commands = []

    def fake_ssh(command, *, timeout=60, input_data=None, ignore_cancel=False):
        commands.append(command)
        return "73109\n" if "scheduler.id" in command else ""

    prep._ssh = fake_ssh
    prep.cancel()
    prep._cancel_recorded_allocation()

    assert any(command == "scancel 73109" for command in commands)
    assert not any("scancel" in command and "73109" not in command for command in commands)
    assert prep.scheduler_id == "73109"
    assert (tmp_path / "scheduler.id").read_text() == "73109\n"


def test_gpu_preparation_rejects_the_cpu_workspace_lock(tmp_path, capsys):
    repo = _repository(tmp_path / "repo")
    profile = _profile()
    profile["slurm"]["gpus"] = 1
    prep = FakePreparation(profile, repository=repo)

    with pytest.raises(PreparationError, match="CPU-only"):
        prep.run()

    assert not prep.submitted
    assert _events(capsys)[-1]["phase"] == "error"


def test_check_does_not_require_local_builder_or_uv(tmp_path, monkeypatch):
    import backend.prepare_worker as prepare_worker

    prep = WorkerPreparation(_profile(), repository=_repository(tmp_path), check_only=True)
    commands = []
    prep._ssh = lambda command, **kwargs: commands.append(command) or ""
    monkeypatch.setattr(prepare_worker.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)

    prep._prerequisites()

    assert not any("test -w /cluster/images" in command for command in commands)
    assert not any("mkdir -m 700 -p" in command for command in commands)


def test_project_dependency_errors_never_echo_authenticated_urls(tmp_path):
    project = tmp_path / "project"
    package = project / "packages/private"
    package.mkdir(parents=True)
    (project / "model.json").write_text(json.dumps({"manifest": {
        "customPackages": [{"path": "packages/private"}], "customDatasets": []
    }}), encoding="utf-8")
    (package / "pyproject.toml").write_text(
        '[project]\ndependencies=["private-package @ https://alice:top-secret@example.test/private.whl"]\n',
        encoding="utf-8",
    )

    with pytest.raises(PreparationError) as error:
        _resource_dependencies(project)

    assert "top-secret" not in str(error.value)
    assert "alice" not in str(error.value)


def test_project_dependency_metadata_cannot_be_a_symlink(tmp_path):
    project = tmp_path / "project"
    package = project / "packages/private"
    package.mkdir(parents=True)
    (project / "model.json").write_text(json.dumps({"manifest": {
        "customPackages": [{"path": "packages/private"}], "customDatasets": []
    }}), encoding="utf-8")
    outside = tmp_path / "external-pyproject.toml"
    outside.write_text('[project]\ndependencies=[]\n', encoding="utf-8")
    (package / "pyproject.toml").symlink_to(outside)

    with pytest.raises(PreparationError, match="cannot be a symlink"):
        _resource_dependencies(project)


def test_process_capture_is_bounded_and_keeps_small_receipts():
    prep = WorkerPreparation(_profile())
    receipt = prep._run_process([sys.executable, "-c", "print('receipt-73109')"], timeout=5)
    assert receipt == "receipt-73109\n"

    output = prep._run_process([sys.executable, "-c", "print('x' * 1000000)"], timeout=5)
    assert len(output.encode()) <= 64 * 1024
    assert output.endswith("x\n")
    with pytest.raises(PreparationError) as error:
        prep._run_process([sys.executable, "-c", "import sys; sys.stderr.write('e' * 1000000); sys.exit(1)"], timeout=5)
    assert len(str(error.value)) <= 2000


def test_cancel_transport_runs_even_after_cancel_flag_is_set():
    prep = FakePreparation(_profile())
    prep.scheduler_id = "73109"
    prep.cancelled = True
    calls = []

    def fake_ssh(command, **kwargs):
        calls.append((command, kwargs))
        if not kwargs.get("ignore_cancel"):
            raise PreparationError("cancelled transport")
        return ""

    prep._ssh = fake_ssh
    prep._cancel_recorded_allocation()

    assert calls == [("scancel 73109", {"timeout": 15, "ignore_cancel": True})]
    assert prep.cancel_requested_id == "73109"
    assert prep._run_process([sys.executable, "-c", "print('scancelled')"], timeout=5,
                             ignore_cancel=True) == "scancelled\n"


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_monitor_transport_failure_cancels_only_owned_allocation(tmp_path, capsys, cancel_fails):
    class LostMonitor(FakePreparation):
        def _wait_for_conversion(self, job_id):
            raise PreparationError("cluster connection dropped")

    prep = LostMonitor(_profile(), repository=_repository(tmp_path))
    prep.fail_at = "scancel" if cancel_fails else None

    with pytest.raises(PreparationError):
        prep.run()

    assert "scancel 42" in prep.remote_commands
    event = _events(capsys)[-1]
    assert event["phase"] == "error"
    if cancel_fails:
        assert "allocation 42 may still be active" in event["message"]
        assert "scancel failed" in event["message"]
        receipt = Path(event["message"].split("Receipt saved to ", 1)[1].split(".", 1)[0])
        assert receipt.read_text() == "42\n"
        assert receipt.stat().st_mode & 0o777 == 0o600
        receipt.unlink()
    else:
        assert "Cancellation requested for owned allocation 42" in event["message"]


def test_submission_transport_failure_cancels_recorded_allocation(tmp_path, capsys):
    class LostSubmission(FakePreparation):
        def _submit_conversion(self, archive_size):
            self.scheduler_id = "42"
            (self.local_stage / "scheduler.id").write_text("42\n", encoding="ascii")
            raise PreparationError("SSH timed out after Slurm accepted the submission")

    prep = LostSubmission(_profile(), repository=_repository(tmp_path))

    with pytest.raises(PreparationError):
        prep.run()

    assert "scancel 42" in prep.remote_commands
    event = _events(capsys)[-1]
    assert "Cancellation requested for owned allocation 42" in event["message"]
