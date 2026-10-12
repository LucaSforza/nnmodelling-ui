"""Shared, validated per-user service profile for the backend and Qt launcher."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, dict[str, Any]] = {
    "backend": {
        "executor": "docker",
        "job_root": str(Path.home() / ".local/share/nnmodelling/jobs"),
        "container_runtime": "docker",
        "worker_image": "nnmodelling-worker:local",
    },
    "slurm": {
        "host": "",
        "root": "",
        "image": "",
        "partition": "",
        "cpus": 2,
        "memory": "4G",
        "time": "00:30:00",
        "gpus": 0,
        "ssh_executable": "ssh",
    },
}
PROFILE_KEYS = {section: set(fields) for section, fields in DEFAULTS.items()}


class ProfileError(ValueError):
    """A profile error safe to show without exposing any configured values."""


def config_path(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    explicit = env.get("NNMODELLING_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    root = env.get("XDG_CONFIG_HOME")
    return (Path(root).expanduser() if root else Path.home() / ".config") / "nnmodelling" / "config.json"


def _profile_path(value: str) -> str:
    if value == "~":
        return str(Path.home())
    if value.startswith("~/"):
        return str(Path.home() / value[2:])
    return value


def validate_profile(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ProfileError("Configuration profile root must be an object.")
    for section, value in raw.items():
        if section not in PROFILE_KEYS:
            continue
        if not isinstance(value, dict):
            raise ProfileError("Configuration profile contains an invalid object.")
        if set(value) - PROFILE_KEYS[section]:
            raise ProfileError("Configuration profile contains unknown profile keys.")

    profile = {section: dict(fields) for section, fields in DEFAULTS.items()}
    for section in PROFILE_KEYS:
        incoming = raw.get(section, {})
        if not isinstance(incoming, dict):
            raise ProfileError("Configuration profile contains an invalid object.")
        profile[section].update(incoming)

    backend, slurm = profile["backend"], profile["slurm"]
    if backend["executor"] not in ("docker", "slurm"):
        raise ProfileError("Configuration profile has an invalid executor.")
    for section in PROFILE_KEYS:
        for key in PROFILE_KEYS[section]:
            value = profile[section][key]
            if key in ("cpus", "gpus"):
                is_integer = type(value) is int or (
                    isinstance(value, float) and math.isfinite(value) and value.is_integer()
                )
                if not is_integer:
                    raise ProfileError("Configuration profile has an invalid integer setting.")
                profile[section][key] = int(value)
            elif not isinstance(value, str):
                raise ProfileError("Configuration profile has an invalid string setting.")
            elif any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ProfileError("Configuration profile has an invalid string setting.")
    if not 1 <= slurm["cpus"] <= 256 or not 0 <= slurm["gpus"] <= 8:
        raise ProfileError("Configuration profile has an out-of-range resource setting.")
    backend["job_root"] = _profile_path(backend["job_root"])
    return profile


def load_profile(path: Path | None = None, *, environ: dict[str, str] | None = None) -> tuple[Path, dict[str, dict[str, Any]]]:
    env = os.environ if environ is None else environ
    selected = path or config_path(env)
    explicitly_selected = bool(env.get("NNMODELLING_CONFIG")) if path is None else True
    if not selected.exists():
        if explicitly_selected:
            raise ProfileError("The selected configuration profile does not exist.")
        return selected, validate_profile({})
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError) as error:
        raise ProfileError("Configuration profile is unreadable or malformed JSON.") from error
    return selected, validate_profile(raw)


def _reject_constant(_value: str) -> None:
    raise ValueError("Non-standard JSON value.")
