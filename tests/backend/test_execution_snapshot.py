from __future__ import annotations

import base64
import copy
import json

import pytest
from fastapi import HTTPException

from backend.validation import resolve_core_packages, validate_execution_project, validate_project
from backend.store import JobStore
from backend.worker import _materialize_project


def _model():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    example = root / "examples/models/local-training"
    source = json.loads((example / "model.json").read_text(encoding="utf-8"))
    files = {
        path.relative_to(example).as_posix(): path.read_bytes()
        for path in (example / "datasets/tiny-regression").rglob("*") if path.is_file()
    }
    source["manifest"]["schemaVersion"] = 3
    source["manifest"]["metavariables"] = [{"name": "width", "type": "integer", "value": 1}]
    source["nodes"][1]["data"]["params"]["in_features"] = {"$ref": "width"}
    execution = copy.deepcopy(source)
    execution["manifest"]["schemaVersion"] = 2
    execution["manifest"].pop("metavariables")
    execution["nodes"][1]["data"]["params"]["in_features"] = 1
    return source, execution, files


def test_compiled_snapshot_validates_and_worker_uses_execution_graph(tmp_path):
    source, execution, files = _model()
    core = resolve_core_packages(execution, files)
    validate_project(source, files, core, allow_authored=True)
    validate_execution_project(source, execution, files, core)

    store = JobStore(tmp_path / "jobs")
    job = store.create(source, files, {}, core, "owner", execution_project=execution)
    restored = store.snapshot_response(job["id"], "owner")
    assert restored["project"] == source
    assert restored["files"] == {name: base64.b64encode(data).decode("ascii") for name, data in files.items()}

    project = tmp_path / "worker-project"
    project.mkdir()
    _materialize_project(store.job_dir(job["id"]) / "snapshot", project)
    effective = json.loads((project / "model.json").read_text(encoding="utf-8"))
    assert effective == execution
    assert effective["nodes"][1]["data"]["params"]["in_features"] == 1


@pytest.mark.parametrize("change", ["literal", "nested-literal", "position", "edge", "manifest"])
def test_compiled_snapshot_rejects_source_mismatch(change):
    source, execution, files = _model()
    if change == "literal":
        source["nodes"][1]["data"]["params"]["bias"] = False
    elif change == "nested-literal":
        source["nodes"][1]["data"]["params"]["custom"] = {"gain": {"$ref": "width"}, "locked": 4}
        execution["nodes"][1]["data"]["params"]["custom"] = {"gain": 1, "locked": 5}
    elif change == "position":
        execution["nodes"][1]["position"]["x"] += 20
    elif change == "edge":
        execution["edges"].pop()
    else:
        execution["manifest"]["name"] = "different"
    core = resolve_core_packages(execution, files)
    validate_project(source, files, core, allow_authored=True)
    with pytest.raises(HTTPException, match="execution_project"):
        validate_execution_project(source, execution, files, core)


def test_compiled_snapshot_rejects_malformed_or_unresolved_bindings():
    source, execution, files = _model()
    source["nodes"][1]["data"]["params"]["in_features"] = {"$ref": "width", "extra": True}
    core = resolve_core_packages(execution, files)
    with pytest.raises(HTTPException):
        validate_project(source, files, core, allow_authored=True)

    source, execution, files = _model()
    execution["nodes"][1]["data"]["params"]["in_features"] = {"$lua": "meta.width"}
    core = resolve_core_packages(execution, files)
    validate_project(source, files, core, allow_authored=True)
    with pytest.raises(HTTPException, match="unresolved"):
        validate_execution_project(source, execution, files, core)
