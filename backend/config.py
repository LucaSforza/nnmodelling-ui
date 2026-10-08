from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = Path(os.environ.get("NNMODELLING_JOB_ROOT", Path.home() / ".local/share/nnmodelling/jobs")).expanduser()
CORE_ROOT = Path(os.environ.get("NNMODELLING_CORE_ROOT", REPOSITORY_ROOT / "stereotype-packages/core")).resolve()
CONTAINER_IMAGE = os.environ.get("NNMODELLING_WORKER_IMAGE", "nnmodelling-worker:local")
CONTAINER_RUNTIME = os.environ.get("NNMODELLING_CONTAINER_RUNTIME", "docker")
EXECUTOR = os.environ.get("NNMODELLING_EXECUTOR", "docker").strip().lower()
SLURM_HOST = os.environ.get("NNMODELLING_SLURM_HOST", "")
SLURM_ROOT = os.environ.get("NNMODELLING_SLURM_ROOT", "")
SLURM_IMAGE = os.environ.get("NNMODELLING_SLURM_IMAGE", "")
SLURM_PARTITION = os.environ.get("NNMODELLING_SLURM_PARTITION", "students")
SLURM_CPUS = int(os.environ.get("NNMODELLING_SLURM_CPUS", "2"))
SLURM_MEMORY = os.environ.get("NNMODELLING_SLURM_MEMORY", "4G")
SLURM_TIME = os.environ.get("NNMODELLING_SLURM_TIME", "00:30:00")
SLURM_SSH = os.environ.get("NNMODELLING_SSH_EXECUTABLE", "ssh")
AUTH_TOKEN = os.environ.get("NNMODELLING_BEARER_TOKEN")
MAX_BUNDLE_BYTES = int(os.environ.get("NNMODELLING_MAX_BUNDLE_BYTES", 46 * 1024 * 1024))
MAX_PROJECT_BYTES = int(os.environ.get("NNMODELLING_MAX_PROJECT_BYTES", 8 * 1024 * 1024))
MAX_FILES = int(os.environ.get("NNMODELLING_MAX_FILES", 10000))
MAX_FILE_BYTES = int(os.environ.get("NNMODELLING_MAX_FILE_BYTES", 128 * 1024 * 1024))
MAX_METRICS_BYTES = 64 * 1024 * 1024
