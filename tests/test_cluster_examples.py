from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools/train_cluster_examples.py"
spec = importlib.util.spec_from_file_location("cluster_examples", SCRIPT)
cluster = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cluster)


def args(tmp_path: Path, models: Path):
    return cluster.parser().parse_args(["--models-dir", str(models), "--output-dir", str(tmp_path / "out")])


def test_discovers_five_supported_examples_and_excludes_deepseek():
    projects = cluster.discover_projects(REPOSITORY / "examples/models")
    names = [name for name, _root, _project in projects]
    assert names == ["local-training", "mnist-mlp", "mnist-vae", "rnn-sine", "tiny-decoder-llm"]


def test_snapshot_includes_inference_asset_and_excludes_model_json():
    root = REPOSITORY / "examples/models/tiny-decoder-llm"
    project = json.loads((root / "model.json").read_text())
    files = cluster.project_files(root, project)
    assert "model.json" not in files
    assert "datasets/llm.tokens-1.0.0/vocabulary.json" in files
    assert "datasets/llm.tokens-1.0.0/data.json" in files
    assert "datasets/llm.tokens-1.0.0/manifest.json" in files


def test_rejects_symlinks_inside_manifest_resource(tmp_path: Path):
    resource = tmp_path / "datasets/data"
    resource.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (resource / "data.json").symlink_to(outside)
    project = {"manifest": {"customDatasets": [{"path": "datasets/data"}], "customPackages": []}}
    with pytest.raises(cluster.TrainingError, match="symlink"):
        cluster.project_files(tmp_path, project)


@pytest.mark.parametrize("value", [".", "a/..", "../a", "a//b", "a/./b"])
def test_rejects_paths_normalized_by_pure_posix_path(value: str):
    with pytest.raises(cluster.TrainingError, match="Unsafe project resource path"):
        cluster.safe_relative(value)


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), 0.5])
def test_poll_interval_must_be_finite_and_at_least_one(tmp_path: Path, interval: float):
    options = args(tmp_path, tmp_path / "models")
    options.poll_interval = interval
    with pytest.raises(cluster.TrainingError, match="poll-interval must be finite"):
        cluster.validate_args(options)


def test_rejects_symlink_project_directories_and_model_files(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    make_project(tmp_path / "target_models")
    target = tmp_path / "target_models/example"
    (models / "linked-project").symlink_to(target, target_is_directory=True)
    with pytest.raises(cluster.TrainingError, match="project directory cannot be a symlink"):
        cluster.discover_projects(models)

    (models / "linked-project").unlink()
    local_project = models / "local-project"
    local_project.mkdir()
    (local_project / "model.json").symlink_to(target / "model.json")
    with pytest.raises(cluster.TrainingError, match="model.json cannot be a symlink"):
        cluster.discover_projects(models)


def test_refuses_non_slurm_health_before_submission(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    make_project(models)
    submitted = []
    monkeypatch.setattr(cluster, "get_json", lambda *_: {"container": {"executor": "docker", "available": True}})
    monkeypatch.setattr(cluster, "request", lambda *a, **kw: submitted.append(a) or (202, {}, b"{}"))
    with pytest.raises(cluster.TrainingError, match="executor='slurm'"):
        cluster.run(args(tmp_path, models))
    assert not submitted


def test_terminal_failure_is_recorded_and_raised(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    make_project(models)
    monkeypatch.setattr(cluster, "get_json", lambda *_: {"container": {"executor": "slurm", "available": True}})

    def fake_request(_base, method, route, **_kwargs):
        assert method == "POST" and route == "/v1/jobs"
        return 202, {}, json.dumps({"id": "job-1", "status": "failed", "metrics": {}, "error": "worker failed"}).encode()

    monkeypatch.setattr(cluster, "request", fake_request)
    options = args(tmp_path, models)
    options.wait = True
    with pytest.raises(cluster.TrainingError, match="worker failed"):
        cluster.run(options)
    report = json.loads((options.output_dir / "report.json").read_text())
    assert report["jobs"][0]["status"] == "failed"
    assert report["jobs"][0]["job_id"] == "job-1"


def test_resume_observes_existing_job_without_duplicate_submission(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    make_project(models)
    options = args(tmp_path, models)
    options.wait = True
    posts = []
    jobs = iter([
        {"id": "job-1", "status": "queued", "metrics": {}, "error": None},
        {"id": "job-1", "status": "completed", "metrics": {"test_loss": 0.1}, "error": None},
    ])
    monkeypatch.setattr(cluster, "get_json", lambda _base, route: {"container": {"executor": "slurm", "available": True}})

    def fake_request(_base, method, route, **_kwargs):
        if method == "POST":
            posts.append(route)
            return 202, {}, json.dumps(next(jobs)).encode()
        raise AssertionError(f"unexpected request {method} {route}")

    monkeypatch.setattr(cluster, "request", fake_request)
    # First execution persists the accepted submission and returns without polling.
    options.wait = False
    assert cluster.run(options) == 0
    # Resume switches on waiting and polls the existing job without a second POST.
    options.wait = True
    monkeypatch.setattr(cluster, "get_json", lambda _base, route: next(jobs) if route.endswith("job-1") else {"container": {"executor": "slurm", "available": True}})
    monkeypatch.setattr(cluster, "download_artifact", lambda _base, _job, kind, _directory: {"filename": kind, "path": kind, "sha256": "0" * 64})
    assert cluster.run(options) == 0
    assert posts == ["/v1/jobs"]
    report = json.loads((options.output_dir / "report.json").read_text())
    assert report["jobs"][0]["status"] == "completed"
    assert set(report["jobs"][0]["artifacts"]) == {"weights", "wheel"}


def make_project(models: Path) -> None:
    project = models / "example"
    project.mkdir(parents=True)
    (project / "model.json").write_text(json.dumps({
        "manifest": {"schemaVersion": 2, "id": "example", "customDatasets": [], "customPackages": []},
        "nodes": [], "edges": [],
    }))
