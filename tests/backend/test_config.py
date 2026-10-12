from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.user_config import DEFAULTS, ProfileError, config_path, load_profile


def test_default_profile_uses_xdg_and_has_no_cluster_defaults(tmp_path):
    xdg = tmp_path / "xdg"
    path, profile = load_profile(environ={"XDG_CONFIG_HOME": str(xdg)})

    assert path == xdg / "nnmodelling" / "config.json"
    assert profile == DEFAULTS
    assert profile["backend"]["executor"] == "docker"
    assert profile["slurm"]["host"] == ""
    assert profile["slurm"]["root"] == ""
    assert profile["slurm"]["image"] == ""
    assert profile["slurm"]["partition"] == ""
    assert not path.exists()


def test_explicit_profile_loads_defaults_and_preserves_unrelated_root_values(tmp_path):
    explicit = tmp_path / "alternate.json"
    explicit.write_text(json.dumps({
        "backend": {"executor": "slurm"},
        "unrelated": {"nested": [1, {"enabled": True}]},
    }), encoding="utf-8")
    path, profile = load_profile(environ={"NNMODELLING_CONFIG": str(explicit)})

    assert path == explicit
    assert profile["backend"]["executor"] == "slurm"
    assert profile["slurm"] == DEFAULTS["slurm"]


def test_profile_rejects_unknown_profile_keys_and_invalid_values(tmp_path):
    explicit = tmp_path / "alternate.json"
    cases = [
        ({"backend": {"unexpected": "private-value"}}, "unknown profile keys"),
        ({"backend": {"executor": "unknown"}}, "invalid executor"),
        ({"slurm": {"cpus": True}}, "integer setting"),
        ({"slurm": {"cpus": 257}}, "out-of-range"),
        ({"slurm": {"gpus": -1}}, "out-of-range"),
        ({"slurm": {"host": "bad\nhost"}}, "string setting"),
        ({"backend": {"job_root": None}}, "string setting"),
        ({"backend": {"job_root": True}}, "string setting"),
        ({"backend": {"job_root": []}}, "string setting"),
        ({"backend": []}, "invalid object"),
    ]
    for value, error in cases:
        explicit.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ProfileError, match=error) as raised:
            load_profile(environ={"NNMODELLING_CONFIG": str(explicit)})
        assert "private-value" not in str(raised.value)


def test_explicit_missing_and_malformed_profiles_fail(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ProfileError, match="does not exist"):
        load_profile(environ={"NNMODELLING_CONFIG": str(missing)})
    assert not missing.exists()

    missing.write_text('{"backend":', encoding="utf-8")
    with pytest.raises(ProfileError, match="malformed JSON"):
        load_profile(environ={"NNMODELLING_CONFIG": str(missing)})


def test_profile_path_precedence(tmp_path):
    explicit = tmp_path / "explicit.json"
    assert config_path({"NNMODELLING_CONFIG": str(explicit),
                        "XDG_CONFIG_HOME": str(tmp_path / "xdg")}) == explicit
    assert config_path({"XDG_CONFIG_HOME": str(tmp_path / "xdg")}) == tmp_path / "xdg/nnmodelling/config.json"


def test_environment_overrides_profile_for_standalone_service(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "backend": {"executor": "slurm", "job_root": "/profile/jobs"},
        "slurm": {"gpus": 2},
    }), encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NNMODELLING_")}
    environment.update({
        "NNMODELLING_CONFIG": str(path),
        "NNMODELLING_EXECUTOR": "docker",
        "NNMODELLING_SLURM_GPUS": "5",
    })
    result = subprocess.run(
        [sys.executable, "-c", "from backend import config; print(config.EXECUTOR, config.JOB_ROOT, config.SLURM_GPUS)"],
        capture_output=True, text=True, check=True, env=environment,
    )
    assert result.stdout.strip() == "docker /profile/jobs 5"


def test_profile_expands_tilde_job_root(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"backend":{"job_root":"~/.local/share/nnmodelling/jobs"}}', encoding="utf-8")
    _, profile = load_profile(path)
    assert profile["backend"]["job_root"] == str(Path.home() / ".local/share/nnmodelling/jobs")
