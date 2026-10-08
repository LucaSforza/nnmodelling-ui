from __future__ import annotations

import io
import json
import tarfile
import asyncio

import pytest

from backend import config
import backend.app as service
from backend.runner import JobRunner, container_status
from backend.slurm import SlurmError, SlurmJobRunner, slurm_configuration
import backend.slurm as slurm
from backend.store import JobStore


def _store_job(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create({"manifest": {"schemaVersion": 2}, "nodes": [], "edges": []}, {}, {}, {}, "owner")
    return store, job


def _configure(monkeypatch):
    monkeypatch.setattr(config, "SLURM_HOST", "cluster")
    monkeypatch.setattr(config, "SLURM_ROOT", "/cluster/jobs")
    monkeypatch.setattr(config, "SLURM_IMAGE", "/home/user/worker.sif")
    monkeypatch.setattr(config, "SLURM_PARTITION", "students")
    monkeypatch.setattr(config, "SLURM_CPUS", 2)
    monkeypatch.setattr(config, "SLURM_MEMORY", "4G")
    monkeypatch.setattr(config, "SLURM_TIME", "00:30:00")
    monkeypatch.setattr(config, "SLURM_SSH", "ssh")


def _output_tar(metrics: dict, *, weights: bytes = b"weights", wheel: bytes = b"wheel") -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in (("metrics.json", json.dumps(metrics).encode()),
                           ("weights.safetensors", weights), ("nnm_model-0.1.0-py3-none-any.whl", wheel),
                           ("worker.log", b"training finished")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def test_config_rejects_unsafe_remote_paths_and_options(monkeypatch):
    monkeypatch.setattr(config, "SLURM_HOST", "cluster; touch /tmp/no")
    monkeypatch.setattr(config, "SLURM_ROOT", "/home/user/../other")
    monkeypatch.setattr(config, "SLURM_IMAGE", "relative.sif")
    monkeypatch.setattr(config, "SLURM_PARTITION", "students;id")
    assert not slurm_configuration()["valid"]


def test_stage_and_batch_command_freeze_source_and_quote_remote_paths(tmp_path, monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(config, "SLURM_ROOT", "/cluster/job root")
    store, job = _store_job(tmp_path)
    runner = SlurmJobRunner(store)
    calls = []

    def fake_ssh(command, *, input=None, timeout=30, binary_output=False):
        calls.append((command, input, timeout, binary_output))
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner, "_ssh_run", fake_ssh)
    stage = runner._stage(job["id"])
    assert stage == f"/cluster/job root/{job['id']}"
    assert "mkdir -p -- '/cluster/job root' && mkdir -- '/cluster/job root/" in calls[0][0]
    with tarfile.open(fileobj=io.BytesIO(calls[1][1]), mode="r:*") as archive:
        names = set(archive.getnames())
    assert "snapshot/project.json" in names
    assert "source/backend/worker.py" in names
    assert "source/python/nnmodelling-runtime/src" in names

    script = runner._batch_script(job["id"], stage)
    assert "singularity exec --cleanenv --containall --no-home --net --network none" in script
    assert "--bind '/cluster/job root/" in script
    assert "--env PYTHONPATH=/app:/app/python/nnmodelling-runtime/src" in script
    assert "--partition students --time 00:30:00 --mem 4G --cpus-per-task 2" in runner._sbatch_arguments(job["id"], stage)
    with pytest.raises(SlurmError, match="Invalid job identity"):
        runner._remote_path("12345")


def test_successful_remote_job_publishes_metrics_and_retrieves_safe_artifacts(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store, job = _store_job(tmp_path)
    runner = SlurmJobRunner(store)
    monkeypatch.setattr("backend.slurm._POLL_SECONDS", 0)
    metrics = {"epochs": [{"epoch": 1, "training_loss": 1.0, "validation_loss": 0.5}],
               "steps": [{"step": 1, "epoch": 1, "training_loss": 1.0, "validation_loss": 0.5}], "test_loss": 0.25}
    states = iter(["RUNNING", ""])

    def fake_ssh(command, *, input=None, timeout=30, binary_output=False):
        stdout = ""
        if command.startswith("scheduler_id=$(sbatch "):
            stdout = "12345\n"
        elif command.startswith("squeue "):
            stdout = next(states)
        elif command.startswith("sacct "):
            stdout = "12345|COMPLETED|0:0\n"
        elif command.startswith("[ ! -L "):
            stdout = json.dumps(metrics)
        elif command.startswith("cd "):
            stdout = _output_tar(metrics)
        return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(runner, "_ssh_run", fake_ssh)
    monkeypatch.setattr(runner, "_ssh_run_at", lambda command, *_args, **_kwargs: type(
        "Result", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    runner._execute(job["id"])
    record = store.internal(job["id"])
    assert record["status"] == "completed", record.get("error")
    assert record["slurm"]["scheduler_id"] == "12345"
    assert store.get(job["id"], "owner")["metrics"] == metrics
    assert store.artifact(job["id"], "owner", "weights").read_bytes() == b"weights"
    assert store.artifact(job["id"], "owner", "wheel").read_bytes() == b"wheel"


def test_remote_output_rejects_traversal_links_and_oversize(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store, job = _store_job(tmp_path)
    runner = SlurmJobRunner(store)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(SlurmError, match="unsafe path"):
        runner._extract_output_archive(job["id"], stream.getvalue())


def test_cancel_and_restart_only_target_persisted_scheduler_job(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store, job = _store_job(tmp_path)
    runner = SlurmJobRunner(store)
    store.transition(job["id"], "queued", status="running", cancel_requested=True,
                     slurm={"scheduler_id": "7654", "remote_path": f"/cluster/jobs/{job['id']}",
                            "configuration": {"host": "cluster", "ssh_executable": "ssh",
                                              "root": "/cluster/jobs"}})
    commands = []
    contacted = []
    monkeypatch.setattr(runner, "_ssh_run_at", lambda command, host, ssh, **_kwargs: (commands.append(command) or contacted.append((host, ssh)) or type(
        "Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()))
    monkeypatch.setattr(config, "SLURM_HOST", "different-cluster")
    runner.cancel(job["id"])
    assert commands == ["scancel 7654"]
    assert contacted == [("cluster", "ssh")]

    restarted = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(SlurmJobRunner, "_ssh_run_at", lambda self, command, *_args, **_kwargs: (commands.append(command) or type(
        "Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()))
    SlurmJobRunner(restarted)
    assert commands[-1] == "scancel 7654"
    assert restarted.get(job["id"], "owner")["status"] == "failed"


def test_lost_sbatch_response_recovers_receipt_and_cancels(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store, job = _store_job(tmp_path)
    runner = SlurmJobRunner(store)
    commands = []

    def fake_ssh(command, **kwargs):
        if command.startswith("scheduler_id=$(sbatch "):
            raise SlurmError("SSH command timed out after 30 seconds.")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def fake_at(command, *_args, **_kwargs):
        commands.append(command)
        output = "98765\n" if command.startswith("cat -- ") else ""
        return type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    monkeypatch.setattr(runner, "_ssh_run", fake_ssh)
    monkeypatch.setattr(runner, "_ssh_run_at", fake_at)
    runner._execute(job["id"])
    record = store.internal(job["id"])
    assert commands == [f"cat -- /cluster/jobs/{job['id']}/scheduler.id", "scancel 98765"]
    assert record["slurm"]["scheduler_id"] == "98765"
    assert record["status"] == "failed"


def test_health_probe_checks_remote_prerequisites_with_batch_ssh(monkeypatch):
    _configure(monkeypatch)
    calls = []
    monkeypatch.setattr(slurm.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(slurm.subprocess, "run", lambda argv, **kwargs: (calls.append((argv, kwargs)) or type(
        "Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()))
    health = slurm.slurm_status()
    assert health["available"] is True
    argv, kwargs = calls[0]
    assert "BatchMode=yes" in argv and "ConnectTimeout=10" in argv
    assert "StrictHostKeyChecking=no" not in argv
    assert "/home/user/worker.sif" in argv[-1]
    assert all(value in argv[-1] for value in ("sbatch", "squeue", "sacct", "scancel", "singularity", "python3"))
    assert kwargs["timeout"] <= 30


def test_health_marks_cluster_unavailable_on_remote_probe_failure(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(slurm.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(slurm.subprocess, "run", lambda *_args, **_kwargs: type(
        "Result", (), {"returncode": 1, "stdout": "", "stderr": "image unreadable"})())
    health = slurm.slurm_status()
    assert health["available"] is False
    assert "image unreadable" in health["error"]


def test_timeout_cancels_persisted_scheduler_and_fails_visibly(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store, job = _store_job(tmp_path)
    runner = SlurmJobRunner(store)
    monkeypatch.setattr(runner, "_job_timeout_seconds", lambda: 0)
    commands = []

    def fake_ssh(command, **_kwargs):
        commands.append(command)
        stdout = "12345\n" if command.startswith("scheduler_id=$(sbatch ") else ""
        return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(runner, "_ssh_run", fake_ssh)
    monkeypatch.setattr(runner, "_ssh_run_at", lambda command, *_args, **_kwargs: (commands.append(command) or type(
        "Result", (), {"returncode": 0, "stdout": "12345\n" if command.startswith("cat -- ") else "", "stderr": ""})()))
    runner._execute(job["id"])
    record = store.internal(job["id"])
    assert record["status"] == "failed"
    assert "time limit" in record["error"]
    assert "scancel 12345" in commands


def test_docker_runner_and_health_object_remain_compatible(tmp_path, monkeypatch):
    _configure(monkeypatch)
    store, _job = _store_job(tmp_path)
    assert isinstance(JobRunner(store), JobRunner)
    monkeypatch.setattr("backend.runner.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("backend.runner.subprocess.run", lambda *_a, **_k: type(
        "Result", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    status = container_status()
    assert status["executor"] == "docker" and status["available"]


def test_health_keeps_container_field_for_selected_slurm_executor(monkeypatch):
    monkeypatch.setattr(config, "EXECUTOR", "slurm")
    async def inline_to_thread(function):
        return function()

    monkeypatch.setattr(service.asyncio, "to_thread", inline_to_thread)
    monkeypatch.setattr(service, "slurm_status", lambda: {
        "available": True, "runtime": "singularity", "executor": "slurm", "error": None})
    health = asyncio.run(service.health())
    assert health["status"] == "ok"
    assert health["container"]["executor"] == "slurm"
