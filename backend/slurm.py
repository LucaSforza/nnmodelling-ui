from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import tarfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from . import config
from .runner import JobRunner
from .remote_data import records_from_snapshot
from .store import JobStore, _read_metrics, utc_now


_ALIAS = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]*$")
_PARTITION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_MEMORY = re.compile(r"^[1-9][0-9]*(?:[KMGTP])?$", re.IGNORECASE)
_TIME = re.compile(r"^(?:[0-9]+-)?[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$")
_JOB_ID = re.compile(r"^[0-9]+$")
_POLL_SECONDS = 5.0
_SSH_TIMEOUT = 30
_ARTIFACT_TRANSFER_TIMEOUT = 24 * 60 * 60
_MAX_TRANSFER_BYTES = config.MAX_ARTIFACT_BYTES
_MAX_ARCHIVE_BYTES = config.MAX_ARTIFACT_BYTES + 64 * 1024 * 1024
_MAX_CHECKPOINT_MANIFEST_BYTES = config.MAX_METRICS_BYTES
_REMOTE_METRICS_MISSING = 44


class SlurmError(RuntimeError):
    pass


class _BoundedReader:
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self.source = source
        self.limit = limit
        self.read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        value = self.source.read(size)
        self.read_bytes += len(value)
        if self.read_bytes > self.limit:
            raise SlurmError("Remote output archive exceeds the configured artifact transfer limit.")
        return value


def slurm_configuration() -> dict[str, Any]:
    """Validate private executor settings without contacting a cluster."""
    errors: list[str] = []
    if not _ALIAS.fullmatch(config.SLURM_HOST):
        errors.append("NNMODELLING_SLURM_HOST must be an SSH alias or hostname.")
    if not _safe_remote_path(config.SLURM_ROOT):
        errors.append("NNMODELLING_SLURM_ROOT must be an absolute remote path without '..'.")
    if not _safe_remote_path(config.SLURM_IMAGE):
        errors.append("NNMODELLING_SLURM_IMAGE must be an absolute remote path without '..'.")
    if not _PARTITION.fullmatch(config.SLURM_PARTITION):
        errors.append("NNMODELLING_SLURM_PARTITION contains unsupported characters.")
    if config.SLURM_CPUS < 1 or config.SLURM_CPUS > 256:
        errors.append("NNMODELLING_SLURM_CPUS must be between 1 and 256.")
    if not 0 <= config.SLURM_GPUS <= 8:
        errors.append("NNMODELLING_SLURM_GPUS must be between 0 and 8.")
    if not _MEMORY.fullmatch(config.SLURM_MEMORY):
        errors.append("NNMODELLING_SLURM_MEMORY must be a Slurm memory value such as 4G.")
    if not _TIME.fullmatch(config.SLURM_TIME):
        errors.append("NNMODELLING_SLURM_TIME must use HH:MM:SS or D-HH:MM:SS.")
    ssh = Path(config.SLURM_SSH)
    if not config.SLURM_SSH or (not ssh.is_absolute() and ssh.name != config.SLURM_SSH) or ".." in ssh.parts:
        errors.append("NNMODELLING_SSH_EXECUTABLE must name an executable on PATH or an absolute path.")
    return {"valid": not errors, "errors": errors, "root": config.SLURM_ROOT,
            "image": config.SLURM_IMAGE, "host": config.SLURM_HOST}


def _safe_remote_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (path.is_absolute() and ".." not in path.parts and not any(
        char in value for char in ":,\\") and not any(ord(char) < 32 for char in value))


def slurm_status() -> dict[str, Any]:
    settings = slurm_configuration()
    executable = shutil.which(config.SLURM_SSH) if settings["valid"] else None
    error = "; ".join(settings["errors"]) or None
    if settings["valid"] and executable is None:
        error = f"SSH executable '{config.SLURM_SSH}' was not found on PATH."
    elif settings["valid"]:
        command = " && ".join((
            f"test -f {shlex.quote(config.SLURM_IMAGE)} && test -r {shlex.quote(config.SLURM_IMAGE)}",
            f"test -d {shlex.quote(config.SLURM_ROOT)} && test -w {shlex.quote(config.SLURM_ROOT)}",
            "command -v sbatch squeue sacct scancel singularity python3 >/dev/null",
        ))
        try:
            result = subprocess.run(
                [executable, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", config.SLURM_HOST, command],
                capture_output=True, text=True, timeout=25, check=False,
            )
            if result.returncode:
                error = (result.stderr or result.stdout).strip()[-1500:] or "Remote Slurm prerequisites are unavailable."
        except (OSError, subprocess.TimeoutExpired) as probe_error:
            error = f"Slurm SSH health probe failed: {probe_error}"
    return {"available": settings["valid"] and executable is not None and not error,
            "runtime": "singularity", "executor": "slurm", "error": error or None}


class SlurmJobRunner(JobRunner):
    """Runs the ordinary backend worker in a bounded remote Slurm allocation."""

    def __init__(self, store: JobStore) -> None:
        self._active_jobs: set[str] = set()
        self._active_lock = threading.RLock()
        super().__init__(store)

    def shutdown(self) -> None:
        self._stopping.set()
        with self._active_lock:
            active = list(self._active_jobs)
        for job_id in active:
            error = self._stop_container(job_id)
            message = "Backend shutdown interrupted this Slurm job."
            if error:
                message += " " + error
            self.store.transition(job_id, "running", status="failed", error=message, finished_at=utc_now())
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)

    def cancel(self, job_id: str) -> None:
        raw = self.store.internal(job_id)
        if raw is None or raw.get("status") != "running":
            return
        error = self._stop_container(job_id)
        if error:
            self.store.update(job_id, error=error)

    def _stop_container(self, job_id: str) -> str | None:
        raw = self.store.internal(job_id)
        remote = raw.get("slurm") if raw else None
        scheduler_id = remote.get("scheduler_id") if isinstance(remote, dict) else None
        if isinstance(remote, dict) and remote.get("attempt_active") is False:
            return None
        execution = remote.get("configuration") if isinstance(remote, dict) else None
        stage = remote.get("remote_path") if isinstance(remote, dict) else None
        if not isinstance(execution, dict) or not isinstance(stage, str):
            return "Recorded Slurm execution configuration is missing; refusing to contact the current cluster."
        host, ssh, root = execution.get("host"), execution.get("ssh_executable"), execution.get("root")
        if (not isinstance(host, str) or not _ALIAS.fullmatch(host)
                or not isinstance(ssh, str) or not ssh
                or (not Path(ssh).is_absolute() and Path(ssh).name != ssh)
                or not isinstance(root, str) or not PurePosixPath(root).is_absolute()
                or stage != str(PurePosixPath(root) / job_id)):
            return "Recorded Slurm execution configuration is invalid; refusing to cancel an unrelated job."
        if not scheduler_id:
            try:
                receipt = self._ssh_run_at(f"cat -- {shlex.quote(stage + '/scheduler.id')}", host, ssh,
                                           timeout=_SSH_TIMEOUT)
            except SlurmError:
                with self._active_lock:
                    submitting = job_id in self._active_jobs
                return None if submitting else "Could not recover the recorded Slurm scheduler identity."
            if receipt.returncode:
                with self._active_lock:
                    submitting = job_id in self._active_jobs
                return None if submitting else "No remote scheduler receipt exists for this Slurm job."
            scheduler_id = receipt.stdout.strip().splitlines()[-1] if receipt.stdout.strip() else ""
            if not scheduler_id:
                return "The remote Slurm scheduler receipt is empty."
            remote["scheduler_id"] = scheduler_id
            scheduler_ids = remote.setdefault("scheduler_ids", [])
            if scheduler_id not in scheduler_ids:
                scheduler_ids.append(scheduler_id)
            self.store.update(job_id, slurm=remote)
        if not _JOB_ID.fullmatch(str(scheduler_id)):
            return "Recorded Slurm scheduler ID is invalid; refusing to cancel an unrelated job."
        try:
            result = self._ssh_run_at(f"scancel {scheduler_id}", host, ssh, timeout=_SSH_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired, SlurmError) as error:
            return f"Could not cancel Slurm job {scheduler_id}: {error}"
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            return f"Could not cancel Slurm job {scheduler_id}: {detail or f'exit {result.returncode}'}"
        return None

    def _execute(self, job_id: str) -> None:
        raw = self.store.internal(job_id)
        if self._stopping.is_set() or raw is None:
            return
        if self.store.transition(job_id, "queued", status="running", started_at=utc_now(), error=None) is None:
            return
        with self._active_lock:
            self._active_jobs.add(job_id)
        scheduler_id: str | None = None
        remote_record: dict[str, Any] | None = None
        cancel_error: str | None = None
        try:
            stage = self._remote_path(job_id)
            remote_record = {"scheduler_id": None, "scheduler_ids": [], "attempt_active": False, "remote_path": stage,
                             "configuration": self._execution_configuration()}
            self.store.update(job_id, slurm=remote_record)
            self._stage(job_id)
            raw = self.store.internal(job_id)
            if self._stopping.is_set() or (raw and raw.get("cancel_requested")):
                self.store.transition(job_id, "running", status="cancelled", finished_at=utc_now())
                return

            if records_from_snapshot(self.store.job_dir(job_id) / "snapshot"):
                self._prepare_remote_data(job_id, stage, remote_record)
                raw = self.store.internal(job_id)
                if self._stopping.is_set() or (raw and raw.get("cancel_requested")):
                    self.store.transition(job_id, "running", status="cancelled", finished_at=utc_now())
                    return

            remote_record["attempt_active"] = True
            remote_record["phase"] = "training"
            remote_record["scheduler_id"] = None
            self.store.update(job_id, slurm=remote_record)
            result = self._ssh_run(self._submit_command(job_id, stage),
                                   input=self._batch_script(job_id, stage), timeout=_SSH_TIMEOUT)
            if result.returncode:
                raise SlurmError(f"sbatch failed: {(result.stderr or result.stdout).strip()[-2000:]}")
            submitted = result.stdout.strip().splitlines()
            scheduler_id = submitted[-1].split(";", 1)[0] if submitted else ""
            if not _JOB_ID.fullmatch(scheduler_id):
                raise SlurmError(f"sbatch returned an invalid scheduler identity: {result.stdout.strip()!r}")
            remote_record["scheduler_id"] = scheduler_id
            remote_record.setdefault("scheduler_ids", []).append(scheduler_id)
            self.store.update(job_id, slurm=remote_record)

            prior_step = int(remote_record.get("global_step", 0))
            while True:
                cancellation_sent = False
                while True:
                    raw = self.store.internal(job_id)
                    if not cancellation_sent and (self._stopping.is_set() or (raw and raw.get("cancel_requested"))):
                        cancel_error = self._stop_container(job_id)
                        cancellation_sent = True
                    state, exit_code = self._scheduler_state(scheduler_id)
                    self._publish_metrics(job_id, stage)
                    if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                                 "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
                        break
                    time.sleep(_POLL_SECONDS)

                remote_record["attempt_active"] = False
                self.store.update(job_id, slurm=remote_record)
                raw = self.store.internal(job_id)
                if self._stopping.is_set():
                    self.store.transition(job_id, "running", status="failed",
                                          error="Backend shutdown interrupted this Slurm job." + (f" {cancel_error}" if cancel_error else ""),
                                          finished_at=utc_now())
                    break
                if raw and (raw.get("cancel_requested") or raw.get("status") == "cancelled"):
                    self.store.transition(job_id, "running", status="cancelled", finished_at=utc_now())
                    break
                if state == "FAILED" and exit_code == "75:0":
                    prior_step = self._validate_checkpoint(stage, job_id, scheduler_id, prior_step)
                    remote_record["global_step"] = prior_step
                    self.store.update(job_id, slurm=remote_record)
                    raw = self.store.internal(job_id)
                    if self._stopping.is_set() or (raw and raw.get("cancel_requested")):
                        self.store.transition(job_id, "running", status="cancelled", finished_at=utc_now())
                        break
                    remote_record["attempt_active"] = True
                    remote_record["scheduler_id"] = None
                    self.store.update(job_id, slurm=remote_record)
                    result = self._ssh_run(self._submit_command(job_id, stage),
                                           input=self._batch_script(job_id, stage), timeout=_SSH_TIMEOUT)
                    if result.returncode:
                        raise SlurmError(f"sbatch failed: {(result.stderr or result.stdout).strip()[-2000:]}")
                    submitted = result.stdout.strip().splitlines()
                    scheduler_id = submitted[-1].split(";", 1)[0] if submitted else ""
                    if not _JOB_ID.fullmatch(scheduler_id):
                        raise SlurmError(f"sbatch returned an invalid scheduler identity: {result.stdout.strip()!r}")
                    remote_record["scheduler_id"] = scheduler_id
                    remote_record.setdefault("scheduler_ids", []).append(scheduler_id)
                    self.store.update(job_id, slurm=remote_record)
                    continue
                self._retrieve_output(job_id, stage)
                if state == "COMPLETED" and exit_code == "0:0":
                    self._finish(job_id)
                    break
                detail = self._local_log_tail(job_id)
                status = f"Slurm job {scheduler_id} finished in state {state or 'UNKNOWN'} with exit code {exit_code or 'unknown'}."
                self.store.transition(job_id, "running", status="failed", error=(detail or status)[-3000:], finished_at=utc_now())
                break
        except (OSError, subprocess.TimeoutExpired, SlurmError, tarfile.TarError, ValueError) as error:
            if remote_record is not None:
                cancel_error = self._stop_container(job_id) or cancel_error
            raw = self.store.internal(job_id)
            if raw and raw.get("status") == "running":
                message = f"Slurm training failed: {error}"
                if cancel_error:
                    message += " " + cancel_error
                self.store.transition(job_id, "running", status="failed", error=message[-3000:], finished_at=utc_now())
        finally:
            with self._active_lock:
                self._active_jobs.discard(job_id)

    def _execution_configuration(self) -> dict[str, Any]:
        return {"host": config.SLURM_HOST, "ssh_executable": config.SLURM_SSH,
                "root": config.SLURM_ROOT, "image": config.SLURM_IMAGE,
                "partition": config.SLURM_PARTITION, "cpus": config.SLURM_CPUS,
                "memory": config.SLURM_MEMORY, "time": config.SLURM_TIME,
                "gpus": config.SLURM_GPUS}

    def _remote_path(self, job_id: str) -> str:
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError
        except (ValueError, AttributeError, TypeError):
            raise SlurmError("Invalid job identity for remote staging.")
        settings = slurm_configuration()
        if not settings["valid"]:
            raise SlurmError("; ".join(settings["errors"]))
        return str(PurePosixPath(config.SLURM_ROOT) / job_id)

    def _stage(self, job_id: str) -> str:
        stage = self._remote_path(job_id)
        root = str(PurePosixPath(stage).parent)
        mkdir = (f"umask 077; mkdir -p -- {shlex.quote(root)} && mkdir -- {shlex.quote(stage)} && "
                 f"mkdir -- {shlex.quote(stage + '/output')} {shlex.quote(stage + '/source')} "
                 f"{shlex.quote(stage + '/dataset-data')}")
        self._checked_remote(mkdir, "Could not prepare remote job directory.")
        archive = self._source_archive(job_id)
        command = f"tar -xf - -C {shlex.quote(stage)} --no-same-owner --no-same-permissions"
        result = self._ssh_run(command, input=archive, timeout=max(_SSH_TIMEOUT, 120))
        if result.returncode:
            raise SlurmError(f"Could not stage immutable job snapshot and worker source: {(result.stderr or result.stdout).strip()[-1500:]}")
        return stage

    def _prepare_remote_data(self, job_id: str, stage: str, record: dict[str, Any]) -> None:
        """Acquire pinned corpus files in a separately tracked networked allocation."""
        record.update(phase="preparing-data", scheduler_id=None, attempt_active=True)
        preparation = record.setdefault("preparation", {"scheduler_id": None, "scheduler_ids": []})
        self.store.update(job_id, slurm=record)
        result = self._ssh_run(self._submit_command(job_id, stage, preparation=True),
                               input=self._preparation_script(stage), timeout=_SSH_TIMEOUT)
        if result.returncode:
            raise SlurmError(f"Dataset preparation sbatch failed: {(result.stderr or result.stdout).strip()[-2000:]}")
        lines = result.stdout.strip().splitlines()
        scheduler_id = lines[-1].split(";", 1)[0] if lines else ""
        if not _JOB_ID.fullmatch(scheduler_id):
            raise SlurmError(f"Dataset preparation returned an invalid scheduler identity: {result.stdout.strip()!r}")
        record["scheduler_id"] = scheduler_id
        preparation["scheduler_id"] = scheduler_id
        preparation.setdefault("scheduler_ids", []).append(scheduler_id)
        record.setdefault("scheduler_ids", []).append(scheduler_id)
        self.store.update(job_id, slurm=record)
        cancelled = False
        while True:
            raw = self.store.internal(job_id)
            if not cancelled and (self._stopping.is_set() or (raw and raw.get("cancel_requested"))):
                error = self._stop_container(job_id)
                if error:
                    raise SlurmError(error)
                cancelled = True
            state, exit_code = self._scheduler_state(scheduler_id)
            if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                         "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
                break
            time.sleep(_POLL_SECONDS)
        record["attempt_active"] = False
        preparation["state"] = state
        preparation["exit_code"] = exit_code
        self.store.update(job_id, slurm=record)
        if cancelled:
            return
        if state != "COMPLETED" or exit_code != "0:0":
            raise SlurmError(f"Dataset preparation allocation {scheduler_id} finished in state {state or 'UNKNOWN'} with exit code {exit_code or 'unknown'}.")

    def _preparation_script(self, stage: str) -> str:
        snapshot, source, data_root = stage + "/snapshot", stage + "/source", stage + "/dataset-data"
        return "\n".join((
            "#!/bin/bash", "set -eu",
            f"singularity exec --cleanenv --containall --no-home --bind {shlex.quote(snapshot + ':/job:ro')} "
            f"--bind {shlex.quote(source + ':/app:ro')} --bind {shlex.quote(data_root + ':/dataset-data:rw')} "
            f"{shlex.quote(config.SLURM_IMAGE)} env PYTHONPATH=/app "
            "python -m backend.remote_data --snapshot /job --data-root /dataset-data",
            ""))

    def _source_archive(self, job_id: str) -> bytes:
        repo = config.REPOSITORY_ROOT
        job_dir = self.store.job_dir(job_id)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            self._add_tree(archive, job_dir / "snapshot", "snapshot")
            self._add_tree(archive, repo / "backend", "source/backend")
            self._add_tree(archive, repo / "python/nnmodelling-runtime/src", "source/python/nnmodelling-runtime/src")
        return stream.getvalue()

    @staticmethod
    def _add_tree(archive: tarfile.TarFile, root: Path, archive_name: str) -> None:
        if not root.is_dir():
            raise SlurmError(f"Required worker source directory is missing: {root}")
        for path in [root, *sorted(root.rglob("*"))]:
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            if path.is_symlink():
                raise SlurmError(f"Refusing to stage symbolic link: {path}")
            relative = path.relative_to(root)
            name = archive_name if not relative.parts else f"{archive_name}/{relative.as_posix()}"
            if path.is_dir():
                info = tarfile.TarInfo(name.rstrip("/") + "/")
                info.type = tarfile.DIRTYPE
                info.mode = 0o555
                archive.addfile(info)
            elif path.is_file():
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)

    def _sbatch_arguments(self, job_id: str, stage: str) -> str:
        seconds = self._allocation_seconds()
        reserve = (min(300, max(1, seconds // 4)) if config.SLURM_GPUS else
                   min(60, max(1, seconds // 10)))
        values = ["--partition", config.SLURM_PARTITION, "--time", config.SLURM_TIME,
                  "--mem", config.SLURM_MEMORY, "--cpus-per-task", str(config.SLURM_CPUS),
                  "--signal", f"B:USR1@{reserve}",
                  "--job-name", f"nnm-{job_id[:8]}", "--output", stage + "/output/worker-%j.log",
                  "--error", stage + "/output/worker-%j.log"]
        if config.SLURM_GPUS:
            values.extend(["--gres", f"gpu:{config.SLURM_GPUS}"])
        return " ".join(shlex.quote(value) for value in values)

    def _submit_command(self, job_id: str, stage: str, *, preparation: bool = False) -> str:
        receipt = shlex.quote(stage + "/scheduler.id")
        temporary = shlex.quote(stage + "/scheduler.id.tmp")
        arguments = self._sbatch_arguments(job_id, stage)
        if preparation:
            args = ["--partition", config.SLURM_PARTITION, "--time", config.SLURM_TIME,
                    "--mem", config.SLURM_MEMORY, "--cpus-per-task", str(config.SLURM_CPUS),
                    "--job-name", f"nnm-data-{job_id[:8]}", "--output", stage + "/output/preparation-%j.log",
                    "--error", stage + "/output/preparation-%j.log"]
            arguments = " ".join(shlex.quote(value) for value in args)
        return (f"scheduler_id=$(rm -f -- {receipt} {temporary} && "
                f"sbatch --parsable {arguments}) && "
                f"printf '%s\\n' \"$scheduler_id\" > {temporary} && "
                f"mv -- {temporary} {receipt} && printf '%s\\n' \"$scheduler_id\"")

    def _batch_script(self, job_id: str, stage: str) -> str:
        snapshot, source, output = (stage + "/snapshot", stage + "/source", stage + "/output")
        return "\n".join((
                "#!/bin/bash", "set -eu",
            f"output={shlex.quote(output)}",
            'trap \'tmp="$output/.checkpoint-request.${SLURM_JOB_ID}.tmp"; '
            'printf "%s\\n" "$SLURM_JOB_ID" > "$tmp"; '
            'mv -f -- "$tmp" "$output/checkpoint-request.${SLURM_JOB_ID}"\' USR1',
            "set +e",
            "singularity exec --cleanenv --containall --no-home --net --network none "
            + ("--nv " if config.SLURM_GPUS else "") +
            "--env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 "
            "--env PYTHONPATH=/app:/app/python/nnmodelling-runtime/src "
            '--env NNMODELLING_SCHEDULER_ID=${SLURM_JOB_ID} '
            '--env NNMODELLING_CHECKPOINT_REQUEST=/output/checkpoint-request.${SLURM_JOB_ID} '
            '--env NNMODELLING_DATASET_DATA_ROOT=/dataset-data '
            + ("--env NNMODELLING_WORKER_DEVICE=cuda " if config.SLURM_GPUS else "") +
            f"--bind {shlex.quote(stage + '/dataset-data:/dataset-data:ro')} "
            f"--bind {shlex.quote(snapshot + ':/job:ro')} "
            f"--bind {shlex.quote(source + ':/app:ro')} "
            f"--bind {shlex.quote(output + ':/output:rw')} {shlex.quote(config.SLURM_IMAGE)} "
            f"python -m backend.worker --snapshot /job --output /output --job-id {shlex.quote(job_id)} &",
            "worker_pid=$!", "worker_status=0",
            'while kill -0 "$worker_pid" 2>/dev/null; do wait "$worker_pid"; worker_status=$?; done',
            'exit "$worker_status"',
            ""))

    def _allocation_seconds(self) -> int:
        value = config.SLURM_TIME
        days = 0
        if "-" in value:
            day_text, value = value.split("-", 1)
            days = int(day_text)
        hours, minutes, seconds = (int(part) for part in value.split(":"))
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def _validate_checkpoint(self, stage: str, job_id: str, scheduler_id: str, prior_step: int) -> int:
        output = shlex.quote(stage + "/output")
        code = (
            "import hashlib,json,os,pathlib,sys; "
            "root=pathlib.Path(sys.argv[1]); m=root/'checkpoint.json'; t=root/'checkpoint.safetensors'; "
            "assert not m.is_symlink() and not t.is_symlink() and m.is_file() and t.is_file(); "
            f"assert m.stat().st_size <= {_MAX_CHECKPOINT_MANIFEST_BYTES}; "
            "v=json.loads(m.read_text()); "
            f"assert type(v.get('version')) is int and v.get('version') == 1 and v.get('job_id') == {job_id!r} and v.get('scheduler_id') == {scheduler_id!r}; "
            f"assert type(v.get('global_step')) is int and v['global_step'] > {prior_step}; "
            f"assert type(v.get('tensor_size')) is int and v['tensor_size'] == t.stat().st_size <= {_MAX_TRANSFER_BYTES}; "
            "h=hashlib.sha256(); f=t.open('rb'); "
            "[h.update(b) for b in iter(lambda:f.read(1048576),b'')]; "
            "assert h.hexdigest() == v.get('tensor_sha256'); print(v['global_step'])"
        )
        result = self._ssh_run(f"python3 -c {shlex.quote(code)} {output}", timeout=180)
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-1500:]
            raise SlurmError(f"Slurm handoff checkpoint failed remote validation: {detail or 'missing or invalid files'}")
        try:
            step = int(result.stdout.strip())
        except ValueError as error:
            raise SlurmError("Slurm handoff checkpoint validation returned invalid progress.") from error
        return step

    def _ssh_argv(self, host: str | None = None, ssh_executable: str | None = None) -> list[str]:
        ssh_executable = ssh_executable or config.SLURM_SSH
        executable = shutil.which(ssh_executable)
        if executable is None:
            raise SlurmError(f"SSH executable '{ssh_executable}' was not found on PATH.")
        return [executable, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host or config.SLURM_HOST]

    def _ssh_run(self, remote_command: str, *, input: str | bytes | None = None,
                 timeout: int = _SSH_TIMEOUT, binary_output: bool = False) -> subprocess.CompletedProcess:
        return self._ssh_run_at(remote_command, config.SLURM_HOST, config.SLURM_SSH,
                                input=input, timeout=timeout, binary_output=binary_output)

    def _ssh_run_at(self, remote_command: str, host: str, ssh_executable: str, *,
                    input: str | bytes | None = None, timeout: int = _SSH_TIMEOUT,
                    binary_output: bool = False) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self._ssh_argv(host, ssh_executable) + [remote_command], input=input, capture_output=True,
                                  text=not binary_output and (isinstance(input, str) or input is None),
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise SlurmError(f"SSH command timed out after {timeout} seconds.") from error
        except OSError as error:
            raise SlurmError(f"SSH command could not start: {error}") from error

    def _checked_remote(self, command: str, description: str) -> str:
        result = self._ssh_run(command, timeout=_SSH_TIMEOUT)
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-1500:]
            raise SlurmError(f"{description} {detail}".strip())
        return result.stdout

    def _scheduler_state(self, scheduler_id: str) -> tuple[str, str]:
        active = self._ssh_run(f"squeue --noheader --jobs {scheduler_id} --format=%T", timeout=_SSH_TIMEOUT)
        if active.returncode == 0 and active.stdout.strip():
            return active.stdout.strip().splitlines()[0].strip().upper(), ""
        accounting = self._ssh_run(
            f"sacct --noheader --parsable2 --jobs {scheduler_id} --format=JobIDRaw,State,ExitCode", timeout=_SSH_TIMEOUT)
        if accounting.returncode:
            raise SlurmError(f"Could not read Slurm job state: {(accounting.stderr or accounting.stdout).strip()[-1500:]}")
        for line in accounting.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 3 and fields[0].strip() == scheduler_id:
                return fields[1].split()[0].split("+", 1)[0].upper(), fields[2].strip()
        return "", ""

    def _publish_metrics(self, job_id: str, stage: str) -> None:
        path = shlex.quote(stage + "/output/metrics.json")
        command = (f"[ ! -L {path} ] || {{ echo 'metrics symlink rejected' >&2; exit 45; }}; "
                   f"[ -f {path} ] || exit {_REMOTE_METRICS_MISSING}; size=$(wc -c < {path}) || exit $?; "
                   f"[ \"$size\" -le {config.MAX_METRICS_BYTES} ] || {{ echo 'metrics exceed limit' >&2; exit 42; }}; "
                   f"cat -- {path}")
        result = self._ssh_run(command, timeout=_SSH_TIMEOUT)
        if result.returncode == _REMOTE_METRICS_MISSING:
            return
        if result.returncode == 42:
            raise SlurmError("Remote metrics exceed the 64 MiB publication limit.")
        if result.returncode == 45:
            raise SlurmError("Remote metrics file is a symbolic link; refusing retrieval.")
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise SlurmError(f"Could not read remote metrics: {detail or f'exit {result.returncode}'}")
        data = result.stdout.encode("utf-8")
        if len(data) > config.MAX_METRICS_BYTES:
            raise SlurmError("Remote metrics exceed the 64 MiB publication limit.")
        output = self.store.job_dir(job_id) / "output"
        destination = output / "metrics.json"
        with tempfile.NamedTemporaryFile(dir=output, prefix=".metrics-", delete=False) as target:
            temporary = Path(target.name)
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        try:
            if _read_metrics(temporary) is None:
                raise SlurmError("Remote metrics file is not a complete valid publication.")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _retrieve_output(self, job_id: str, stage: str) -> None:
        output = shlex.quote(stage + "/output")
        command = (
            f"cd {output} && total=0 && count=0 && set -- && "
            "for f in metrics.json metrics.tmp weights.safetensors worker-*.log worker.log preparation-*.log *.whl; do "
            "if [ -e \"$f\" ] || [ -L \"$f\" ]; then "
            "[ ! -L \"$f\" ] || { echo 'linked output rejected' >&2; exit 43; }; "
            f"count=$((count + 1)); [ \"$count\" -le {config.MAX_FILES} ] || {{ echo 'too many output files' >&2; exit 46; }}; "
            "size=$(stat -c %s -- \"$f\") || exit $?; total=$((total + size)); set -- \"$@\" \"$f\"; "
            f"fi; done && [ \"$total\" -le {_MAX_TRANSFER_BYTES} ] || "
            "{ echo 'output exceeds transfer limit' >&2; exit 44; }; "
            "if [ \"$#\" -gt 0 ]; then tar -cf - -- \"$@\"; else tar -cf - --files-from /dev/null; fi")
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(self._ssh_argv() + [command], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=errors)
            try:
                if process.stdout is None:
                    raise SlurmError("SSH output stream was not available.")
                self._extract_output_archive(job_id, process.stdout, streaming=True)
                status = process.wait(timeout=_ARTIFACT_TRANSFER_TIMEOUT)
            except Exception:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
            if status:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()[-1500:]
                if status == 44:
                    raise SlurmError("Remote output files exceed the configured 64 GiB artifact limit.")
                if status == 46:
                    raise SlurmError("Remote output contains too many files.")
                raise SlurmError(f"Could not retrieve Slurm output: {detail or f'exit {status}'}")

    def _extract_output_archive(self, job_id: str, archive: bytes | BinaryIO, *, streaming: bool = False) -> None:
        if isinstance(archive, bytes) and len(archive) > _MAX_ARCHIVE_BYTES:
            raise SlurmError("Remote output archive exceeds the configured artifact limit.")
        output = self.store.job_dir(job_id) / "output"
        seen: set[str] = set()
        total = 0
        stream = archive if not isinstance(archive, bytes) else io.BytesIO(archive)
        if streaming:
            stream = _BoundedReader(stream, _MAX_ARCHIVE_BYTES)
        with tarfile.open(fileobj=stream, mode="r|*" if streaming else "r:*") as source:
            for member in source:
                if member.isdir() and member.name in {".", "./"}:
                    continue
                pure = PurePosixPath(member.name)
                name = pure.name
                if "\\" in member.name or pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                    raise SlurmError(f"Remote output archive contains an unsafe path: {member.name!r}")
                if not member.isfile() or member.issym() or member.islnk():
                    raise SlurmError(f"Remote output archive contains a non-regular file: {member.name!r}")
                if name == "metrics.tmp":
                    if member.size < 0 or member.size > config.MAX_METRICS_BYTES:
                        raise SlurmError("Remote transient metrics exceed the 64 MiB publication limit.")
                    continue
                if (name not in {"metrics.json", "weights.safetensors", "worker.log"}
                        and not re.fullmatch(r"(?:worker|preparation)-[0-9]+\.log", name) and not name.endswith(".whl")):
                    raise SlurmError(f"Remote output archive contains an unexpected file: {name!r}")
                if name in seen or member.size < 0 or member.size > _MAX_TRANSFER_BYTES:
                    raise SlurmError(f"Remote output archive contains a duplicate or oversized file: {name!r}")
                if len(seen) >= config.MAX_FILES:
                    raise SlurmError("Remote output archive contains too many files.")
                total += member.size
                if total > _MAX_TRANSFER_BYTES:
                    raise SlurmError("Remote output files exceed the configured 64 GiB artifact limit.")
                data = source.extractfile(member)
                if data is None:
                    raise SlurmError(f"Could not read remote output file {name!r}.")
                destination = output / name
                with tempfile.NamedTemporaryFile(dir=output, prefix=f".{name}.", delete=False) as target:
                    temporary = Path(target.name)
                    shutil.copyfileobj(data, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                try:
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                seen.add(name)

    def _local_log_tail(self, job_id: str) -> str:
        raw = self.store.internal(job_id) or {}
        slurm = raw.get("slurm") if isinstance(raw.get("slurm"), dict) else {}
        scheduler_id = slurm.get("scheduler_id")
        name = f"worker-{scheduler_id}.log" if isinstance(scheduler_id, str) and _JOB_ID.fullmatch(scheduler_id) else "worker.log"
        path = self.store.safe_output_file(job_id, name)
        if path is None:
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
        except OSError:
            return ""
