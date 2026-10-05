from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config
from .store import JobStore, utc_now


def container_status() -> dict[str, Any]:
    executable = shutil.which(config.CONTAINER_RUNTIME)
    if executable is None:
        return {"available": False, "runtime": config.CONTAINER_RUNTIME, "error": f"Container runtime '{config.CONTAINER_RUNTIME}' was not found on PATH."}
    try:
        result = subprocess.run([executable, "info"], capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "runtime": config.CONTAINER_RUNTIME, "error": f"Container runtime check failed: {error}"}
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        return {"available": False, "runtime": config.CONTAINER_RUNTIME, "error": detail or "Container daemon is unavailable."}
    return {"available": True, "runtime": config.CONTAINER_RUNTIME, "error": None}


@lru_cache(maxsize=4)
def _is_podman(executable: str) -> bool:
    if "podman" in Path(executable).resolve().name.lower():
        return True
    try:
        with Path(executable).open("rb") as source:
            return b"podman" in source.read(4096).lower()
    except OSError:
        return False


class JobRunner:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        stop_errors = {}
        for job_id in store.interrupted_running():
            error = self._stop_container(job_id)
            if error:
                stop_errors[job_id] = error
        for job_id in store.recover(stop_errors):
            self._queue.put(job_id)

    def start(self) -> None:
        if self._thread is None and not self._stopping.is_set():
            self._thread = threading.Thread(target=self._run, name="nnmodelling-job-runner", daemon=True)
            self._thread.start()

    def shutdown(self) -> None:
        self._stopping.set()
        with self._lock:
            active = list(self._processes.items())
        for job_id, process in active:
            error = self._stop_container(job_id)
            if error:
                self.store.update(job_id, error=error)
            elif process.poll() is None:
                process.terminate()
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                error = self._stop_container(job_id)
                if error:
                    self.store.update(job_id, error=error)

    def _stop_container(self, job_id: str) -> str | None:
        executable = shutil.which(config.CONTAINER_RUNTIME)
        if executable is not None:
            try:
                result = subprocess.run([executable, "stop", "--time", "5", f"nnmodelling-{job_id}"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=8, check=False)
                if result.returncode:
                    return f"Container runtime could not stop nnmodelling-{job_id} (exit {result.returncode})."
            except (OSError, subprocess.TimeoutExpired) as error:
                return f"Container runtime could not stop nnmodelling-{job_id}: {error}"
        else:
            return "Container runtime disappeared while the job was running."
        return None

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                if self._stopping.is_set():
                    continue
                self._execute(job_id)
            except Exception as error:
                if job_id is not None:
                    self.store.update(job_id, status="failed", error=f"Backend worker failed: {error}", finished_at=utc_now())
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        raw = self.store.internal(job_id)
        if self._stopping.is_set() or raw is None:
            return
        if self.store.transition(job_id, "queued", status="running", started_at=utc_now(), error=None) is None:
            return
        runtime = container_status()
        if not runtime["available"]:
            raw = self.store.internal(job_id)
            if raw and raw.get("cancel_requested"):
                self.store.transition(job_id, "running", status="cancelled", error=None, finished_at=utc_now())
            else:
                self.store.transition(job_id, "running", status="failed", error=runtime["error"], finished_at=utc_now())
            return
        raw = self.store.internal(job_id)
        if self._stopping.is_set():
            self.store.transition(job_id, "running", status="queued", started_at=None)
            return
        if raw is None or raw.get("cancel_requested"):
            self.store.transition(job_id, "running", status="cancelled", error=None, finished_at=utc_now())
            return
        executable = shutil.which(config.CONTAINER_RUNTIME)
        assert executable is not None
        job_dir = self.store.job_dir(job_id)
        snapshot = (job_dir / "snapshot").resolve()
        output = (job_dir / "output").resolve()
        command = [executable, "run", "--rm", "--name", f"nnmodelling-{job_id}", "--network", "none",
                   "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                   "--pids-limit", "128", "--memory", "4g", "--cpus", "2",
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m"]
        # Rootless Podman's keep-id makes the writable output mount owned by the caller.
        if _is_podman(executable):
            command.extend(["--userns=keep-id", f"--user={os.getuid()}:{os.getgid()}"])
        else:
            command.extend([f"--user={os.getuid()}:{os.getgid()}"])
        command.extend(["-v", f"{snapshot}:/job:ro,Z", "-v", f"{output}:/output:rw,Z",
                        config.CONTAINER_IMAGE, "python", "-m", "backend.worker",
                        "--snapshot", "/job", "--output", "/output", "--job-id", job_id])
        log_path = output / "worker.log"
        try:
            with log_path.open("wb") as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                with self._lock:
                    self._processes[job_id] = process
                raw = self.store.internal(job_id)
                if self._stopping.is_set() or (raw and raw.get("cancel_requested")):
                    stop_error = self._stop_container(job_id)
                    if stop_error:
                        self.store.update(job_id, error=stop_error)
                    elif process.poll() is None:
                        process.terminate()
                result = process.wait(timeout=24 * 60 * 60)
        except subprocess.TimeoutExpired:
            stop_error = self._stop_container(job_id)
            if process.poll() is None:
                process.kill()
            process.wait()
            reason = "Training exceeded the 24-hour job limit."
            if stop_error:
                reason += " " + stop_error
            self.store.transition(job_id, "running", status="failed", error=reason, finished_at=utc_now())
        except OSError as error:
            self.store.transition(job_id, "running", status="failed", error=f"Could not start training container: {error}", finished_at=utc_now())
        else:
            raw = self.store.internal(job_id)
            if raw is not None and (raw["status"] == "cancelled" or raw.get("cancel_requested")):
                self.store.transition(job_id, "running", status="cancelled", finished_at=utc_now())
                return
            if result != 0:
                detail = ""
                try:
                    safe_log = self.store.safe_output_file(job_id, "worker.log")
                    if safe_log is not None:
                        detail = safe_log.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
                except OSError:
                    pass
                self.store.transition(job_id, "running", status="failed", error=(detail or f"Training container exited with status {result}.")[-3000:], finished_at=utc_now())
            else:
                self._finish(job_id)
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _finish(self, job_id: str) -> None:
        metrics, error = self.store.validated_result(job_id)
        if metrics is None or not metrics["epochs"] or not _finite(metrics["test_loss"]):
            detail = error or "worker completed without finite final metrics"
            self.store.transition(job_id, "running", status="failed", error=f"Training result was incomplete: {detail}", finished_at=utc_now())
            return
        if not self.store.complete(job_id, metrics):
            current = self.store.internal(job_id)
            if current and current.get("status") == "running":
                self.store.transition(job_id, "running", status="cancelled", finished_at=utc_now())


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
