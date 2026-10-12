from __future__ import annotations

import os
from pathlib import Path

from .user_config import load_profile


def _integer_environment(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return -1


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_, _PROFILE = load_profile()
_BACKEND_PROFILE = _PROFILE["backend"]
_SLURM_PROFILE = _PROFILE["slurm"]


def _setting(environment: str, section: dict[str, object], key: str, default: object) -> object:
    if environment in os.environ:
        return os.environ[environment]
    return section.get(key, default)


JOB_ROOT = Path(str(_setting("NNMODELLING_JOB_ROOT", _BACKEND_PROFILE, "job_root", Path.home() / ".local/share/nnmodelling/jobs"))).expanduser()
CORE_ROOT = Path(os.environ.get("NNMODELLING_CORE_ROOT", REPOSITORY_ROOT / "stereotype-packages/core")).resolve()
CONTAINER_IMAGE = str(_setting("NNMODELLING_WORKER_IMAGE", _BACKEND_PROFILE, "worker_image", "nnmodelling-worker:local"))
CONTAINER_RUNTIME = str(_setting("NNMODELLING_CONTAINER_RUNTIME", _BACKEND_PROFILE, "container_runtime", "docker"))
EXECUTOR = str(_setting("NNMODELLING_EXECUTOR", _BACKEND_PROFILE, "executor", "docker")).strip().lower()
SLURM_HOST = str(_setting("NNMODELLING_SLURM_HOST", _SLURM_PROFILE, "host", ""))
SLURM_ROOT = str(_setting("NNMODELLING_SLURM_ROOT", _SLURM_PROFILE, "root", ""))
SLURM_IMAGE = str(_setting("NNMODELLING_SLURM_IMAGE", _SLURM_PROFILE, "image", ""))
SLURM_PARTITION = str(_setting("NNMODELLING_SLURM_PARTITION", _SLURM_PROFILE, "partition", ""))
SLURM_CPUS = int(_setting("NNMODELLING_SLURM_CPUS", _SLURM_PROFILE, "cpus", 2))
SLURM_MEMORY = str(_setting("NNMODELLING_SLURM_MEMORY", _SLURM_PROFILE, "memory", "4G"))
SLURM_TIME = str(_setting("NNMODELLING_SLURM_TIME", _SLURM_PROFILE, "time", "00:30:00"))
SLURM_GPUS = _integer_environment("NNMODELLING_SLURM_GPUS", int(_SLURM_PROFILE["gpus"]))
SLURM_SSH = str(_setting("NNMODELLING_SSH_EXECUTABLE", _SLURM_PROFILE, "ssh_executable", "ssh"))
AUTH_TOKEN = os.environ.get("NNMODELLING_BEARER_TOKEN")
MAX_BUNDLE_BYTES = int(os.environ.get("NNMODELLING_MAX_BUNDLE_BYTES", 46 * 1024 * 1024))
MAX_PROJECT_BYTES = int(os.environ.get("NNMODELLING_MAX_PROJECT_BYTES", 8 * 1024 * 1024))
MAX_FILES = int(os.environ.get("NNMODELLING_MAX_FILES", 10000))
MAX_FILE_BYTES = int(os.environ.get("NNMODELLING_MAX_FILE_BYTES", 128 * 1024 * 1024))
MAX_ARTIFACT_BYTES = int(os.environ.get("NNMODELLING_MAX_ARTIFACT_BYTES", 64 * 1024 * 1024 * 1024))
MAX_METRICS_BYTES = 64 * 1024 * 1024
