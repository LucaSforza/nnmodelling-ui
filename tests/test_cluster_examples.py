from __future__ import annotations

import importlib.util
import hashlib
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


def test_discovers_all_bundled_projects():
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
    assert "execution-model.json" not in files


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


def schema_three_project(directory: Path) -> tuple[dict, dict]:
    directory.mkdir(parents=True)
    source = {
        "manifest": {"schemaVersion": 3, "id": "compiled-example", "metavariables": [{"name": "width"}],
                     "customDatasets": [], "customPackages": []},
        "nodes": [], "edges": [],
    }
    execution = {"manifest": {"schemaVersion": 2, "id": "compiled-example",
                              "customDatasets": [], "customPackages": []},
                 "nodes": [], "edges": []}
    model_bytes = json.dumps(source, separators=(",", ":")).encode()
    (directory / "model.json").write_bytes(model_bytes)
    wrapper = {"source_sha256": hashlib.sha256(model_bytes).hexdigest(), "execution_project": execution}
    (directory / "execution-model.json").write_text(json.dumps(wrapper), encoding="utf-8")
    return source, execution


def test_schema_three_execution_artifact_is_required_and_source_pinned(tmp_path: Path):
    directory = tmp_path / "model"
    source, _execution = schema_three_project(directory)
    model_bytes = (directory / "model.json").read_bytes()
    artifact = directory / "execution-model.json"
    artifact.unlink()
    with pytest.raises(cluster.TrainingError, match="required native execution artifact"):
        cluster.load_execution_artifact(directory, source, model_bytes)

    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    artifact.symlink_to(other)
    with pytest.raises(cluster.TrainingError, match="symlink"):
        cluster.load_execution_artifact(directory, source, model_bytes)
    artifact.unlink()

    artifact.write_text("not json", encoding="utf-8")
    with pytest.raises(cluster.TrainingError, match="malformed"):
        cluster.load_execution_artifact(directory, source, model_bytes)
    artifact.write_text(json.dumps({"source_sha256": "0" * 64, "execution_project": {}}), encoding="utf-8")
    with pytest.raises(cluster.TrainingError, match="does not match"):
        cluster.load_execution_artifact(directory, source, model_bytes)

    artifact.write_text(json.dumps({"source_sha256": hashlib.sha256(model_bytes).hexdigest(),
                                    "execution_project": {"manifest": {"schemaVersion": 3}}}), encoding="utf-8")
    with pytest.raises(cluster.TrainingError, match="concrete schema-2"):
        cluster.load_execution_artifact(directory, source, model_bytes)


def test_submission_includes_compiled_project_and_resume_hashes_companion(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    source, execution = schema_three_project(models / "compiled-example")
    posts = []
    monkeypatch.setattr(cluster, "get_json", lambda *_: {"container": {"executor": "slurm", "available": True}})

    def fake_request(_base, method, route, **kwargs):
        if method == "POST" and route == "/v1/jobs":
            posts.append(kwargs["payload"])
            return 202, {}, json.dumps({"id": "job-compiled", "status": "queued", "metrics": {}, "error": None}).encode()
        raise AssertionError(f"unexpected {method} {route}")

    monkeypatch.setattr(cluster, "request", fake_request)
    options = args(tmp_path, models)
    assert cluster.run(options) == 0
    body = posts[0]
    assert body["project"] == source
    assert body["execution_project"] == execution
    assert "execution-model.json" not in body["files"]

    wrapper = json.loads((models / "compiled-example/execution-model.json").read_text())
    wrapper["execution_project"]["nodes"].append({"id": "changed"})
    (models / "compiled-example/execution-model.json").write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(cluster.TrainingError, match="incompatible"):
        cluster.run(options)
    assert len(posts) == 1


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
