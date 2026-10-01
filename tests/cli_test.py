"""End-to-end nnmodelctl against the real offscreen Qt application."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def main():
    ui, cli = map(Path, sys.argv[1:3])
    with tempfile.TemporaryDirectory(prefix="nn-cli-", dir="/tmp/opencode") as directory:
        root = Path(directory)
        endpoint = root / "control.sock"
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        with (root / "ui.log").open("w+") as log:
            process = subprocess.Popen([str(ui), "--socket", str(endpoint)], env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 10
                while not endpoint.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        raise AssertionError("UI service failed to start: " + log.read())
                    time.sleep(0.02)

                def command(operation, args=None, expected=0, stdin=False):
                    payload = json.dumps(args or {})
                    result = subprocess.run(
                        [str(cli), "--socket", str(endpoint), operation, "-" if stdin else payload],
                        input=payload if stdin else None, text=True, capture_output=True, timeout=15,
                    )
                    assert result.returncode == expected, (operation, result.returncode, result.stdout, result.stderr)
                    response = json.loads(result.stdout)
                    assert response["ok"] == (expected == 0), response
                    return response.get("result")

                assert command("project.snapshot") is None
                command("project.create", {"parent": str(root), "id": "authored", "name": "Authored"})
                seeded = command("project.snapshot")
                assert len(seeded["nodes"]) == 2
                output = next(n["id"] for n in seeded["nodes"] if n["package"]["id"] == "core.output")
                loss_output = next(n["id"] for n in seeded["nodes"] if n["package"]["id"] == "core.loss-output")
                diagnostics = command("analysis.diagnostics")
                assert set(diagnostics) == {"available", "complete", "problems", "tensors"}
                assert diagnostics["available"] is True
                assert diagnostics["complete"] is False
                assert any(p["node"] is None and p["category"] == "incomplete" for p in diagnostics["problems"])
                definition = {
                    "name": "Local shape rule", "description": "Pass-through only", "kind": "layer",
                    "view": {"color": "#4779c4", "width": 220, "height": 100},
                    "parameters": {"features": {"type": "integer", "minimum": 1, "default": 8, "position": "bottom"}},
                    "outputs": [{"id": "prediction", "type": "output"}, {"id": "objective", "type": "loss"}],
                }
                command("stereotype.create", {
                    "id": "local.shape", "version": "0.1.0", "definition": definition,
                    "inference": "return function(context, parameters, services) local t,e=tensor.create({},context.inputs[1].dtype); if e then return {status='error',message=e} end; return {status='success',outputs={prediction=context.inputs[1],objective=t}} end",
                    "dependencies": {},
                }, stdin=True)
                for invalid_outputs in [
                    [],
                    [{"id": "one", "type": "output"}, {"id": "two", "type": "output"}],
                    [{"id": "same", "type": "output"}, {"id": "same", "type": "loss"}],
                    [{"id": "unknown", "type": "tensor"}],
                ]:
                    command("stereotype.create", {
                        "id": "local.invalid", "version": "0.1.0",
                        "definition": dict(definition, outputs=invalid_outputs),
                        "inference": "return function(context) return {status='success',output=context.inputs[1]} end",
                    }, expected=1)
                command("dataset.create", {
                    "id": "local.images", "version": "0.1.0", "select": True,
                    "definition": {"name": "Images", "batch": {
                        "inputs": {"image": {"dtype": "float32", "shape": ["B", 8]}}, "targets": {},
                    }},
                })
                command("node.add", {"id": "input", "package": "core.input", "version": "0.1.0"})
                command("node.add", {"id": "local", "package": "local.shape", "version": "0.1.0", "y": 200})
                command("edge.connect", {"id": "edge", "source": "input", "sourceHandle": "out", "target": "local", "targetHandle": "in"})
                command("edge.connect", {"id": "prediction-output", "source": "local", "sourceHandle": "prediction", "target": output, "targetHandle": "in"})
                command("edge.connect", {"id": "objective-output", "source": "local", "sourceHandle": "objective", "target": loss_output, "targetHandle": "in"})
                command("edge.connect", {"id": "wrong-loss", "source": "local", "sourceHandle": "objective", "target": output, "targetHandle": "in"}, expected=1)
                command("edge.connect", {"id": "wrong-output", "source": "local", "sourceHandle": "prediction", "target": loss_output, "targetHandle": "in"}, expected=1)
                command("node.parameter", {"id": "local", "key": "features", "value": "16"})
                command("node.parameter", {"id": "input", "key": "binding", "value": "missing-input"})
                before_diagnostics = command("project.snapshot")
                diagnostics = command("analysis.diagnostics")
                after_diagnostics = command("project.snapshot")
                assert set(diagnostics) == {"available", "complete", "problems", "tensors"}
                assert diagnostics["available"] is True
                assert diagnostics["complete"] is False
                assert before_diagnostics["dirty"] == after_diagnostics["dirty"]
                assert diagnostics == command("analysis.diagnostics")
                problems = {problem["node"]: problem for problem in diagnostics["problems"]}
                assert problems["input"]["code"] == "model.incomplete"
                assert problems["input"]["causeNode"] is None
                assert problems["local"]["code"] == "model.blocked"
                assert problems["local"]["causeNode"] == "input"
                for problem in problems.values():
                    assert {"code", "category", "severity", "node", "scope", "package",
                            "file", "line", "message", "causeNode"} == set(problem)
                    assert set(problem["package"]) == {"id", "version"}
                    assert isinstance(problem["line"], int)
                assert command("ui.inspect")["currentScope"] == ""
                command("node.parameter", {"id": "input", "key": "binding", "value": "image"})
                restored = command("analysis.diagnostics")
                assert restored["problems"] == []
                assert {tensor["node"] for tensor in restored["tensors"]} == {"input", "local", output, loss_output}
                assert len(restored["tensors"]) == 5
                local_tensors = {t["handle"]: t for t in restored["tensors"] if t["node"] == "local"}
                assert local_tensors["prediction"]["type"] == "output" and local_tensors["prediction"]["shape"] == ["B", "8"]
                assert local_tensors["objective"]["type"] == "loss" and local_tensors["objective"]["shape"] == []
                assert all(t["handle"] is None and t["type"] is None for t in restored["tensors"] if t["node"] in {output, loss_output})
                command("node.parameter", {"id": "local", "key": "features", "value": "0"}, expected=1)
                command("ui.scope", {"id": False}, expected=1)
                command("unknown.operation", expected=1)
                snapshot = command("project.snapshot")
                assert len(snapshot["nodes"]) == 4 and len(snapshot["edges"]) == 3
                authored_package = next(p for p in snapshot["packages"] if p["id"] == "local.shape")
                assert authored_package["outputs"] == definition["outputs"]
                assert next(n for n in snapshot["nodes"] if n["id"] == "local")["parameters"]["features"] == 16
                command("project.save")
                saved = root / "authored" / "model.json"
                assert saved.read_text().startswith("{\n  ") and saved.read_text().endswith("\n")
                command("project.close")
                command("project.open", {"path": str(saved.parent)})
                assert len(command("project.snapshot")["nodes"]) == 4
                info = command("ui.inspect")
                assert {"palette", "resources", "inspector"} <= {w["id"] for w in info["widgets"]}
                command("node.move", {"id": "local", "x": 40, "y": 200})
                command("project.open", {"path": str(saved.parent)}, expected=1)
                assert command("project.snapshot")["dirty"] is True
                command("project.save")

                # Two-output subflows spawn mapped terminals only, not Input or edges.
                subflow_definition = dict(definition, name="Typed subflow", kind="subflow", parameters={})
                command("stereotype.create", {
                    "id": "local.subflow", "version": "0.1.0", "definition": subflow_definition,
                    "inference": "return function(context, parameters, services) return services.infer_subflow(context.inputs[1]) end",
                })
                command("node.add", {"id": "nested", "package": "local.subflow", "version": "0.1.0", "y": 400})
                children = [n for n in command("project.snapshot")["nodes"] if n["scope"] == "nested"]
                assert len(children) == 2
                terminals = {n["boundaryHandle"]: n for n in children}
                assert set(terminals) == {"prediction", "objective"}
                assert terminals["prediction"]["package"]["id"] == "core.output"
                assert terminals["objective"]["package"]["id"] == "core.loss-output"
                command("node.boundary", {"id": terminals["prediction"]["id"], "handle": "objective"}, expected=1)
                command("node.boundary", {"id": output, "handle": "prediction"}, expected=1)
                command("node.boundary", {"id": terminals["prediction"]["id"], "handle": "prediction"})
                command("node.add", {"id": "nested-input", "package": "core.input", "version": "0.1.0", "scope": "nested"})
                command("node.add", {"id": "nested-mse", "package": "core.mse-loss", "version": "0.1.0", "scope": "nested", "y": 180})
                for edge_id, source, handle, target in [
                    ("local-nested", "local", "prediction", "nested"),
                    ("nested-normal", "nested-input", "out", terminals["prediction"]["id"]),
                    ("nested-mse-input", "nested-input", "out", "nested-mse"),
                    ("nested-loss", "nested-mse", "loss", terminals["objective"]["id"]),
                ]:
                    command("edge.connect", {"id": edge_id, "source": source, "sourceHandle": handle, "target": target, "targetHandle": "in"})
                typed_nested = command("analysis.diagnostics")
                assert typed_nested["complete"] is True and typed_nested["problems"] == []
                nested_tensors = {t["handle"]: t for t in typed_nested["tensors"] if t["node"] == "nested"}
                assert nested_tensors["prediction"]["shape"] == ["B", "8"]
                assert nested_tensors["objective"]["shape"] == []
                command("project.save")
                command("project.close")
                command("project.open", {"path": str(saved.parent)})
                persisted = [n for n in command("project.snapshot")["nodes"] if n["scope"] == "nested" and n["boundaryHandle"]]
                assert {n["boundaryHandle"] for n in persisted} == {"prediction", "objective"}
                assert command("analysis.diagnostics")["complete"] is True

                # A partially connected client cannot block other UI/CLI work.
                idle = socket.socket(socket.AF_UNIX)
                idle.connect(str(endpoint))
                idle.sendall(b'{"operation":')
                command("project.snapshot")
                idle.close()
                malformed = socket.socket(socket.AF_UNIX)
                malformed.settimeout(5)
                malformed.connect(str(endpoint))
                malformed.sendall(b"{bad}\n")
                assert json.loads(malformed.recv(4096))["ok"] is False
                malformed.close()

                command("project.create", {"parent": str(root), "id": "vae", "name": "VAE", "template": "mnist-vae"})
                snapshot = command("project.snapshot")
                assert len(snapshot["nodes"]) == 23
                assert sum(n["package"]["id"] == "core.subflow-proxy" for n in snapshot["nodes"]) == 2
                vae_diagnostics_before = command("project.snapshot")
                vae_diagnostics = command("analysis.diagnostics")
                vae_diagnostics_after = command("project.snapshot")
                assert vae_diagnostics["available"] is True
                assert vae_diagnostics["complete"] is True
                assert vae_diagnostics["problems"] == []
                assert len(vae_diagnostics["tensors"]) == 23
                assert vae_diagnostics_before["dirty"] == vae_diagnostics_after["dirty"]
                mean = next(n for n in snapshot["nodes"] if n["id"] == "mean")
                old_features = mean["parameters"]["in_features"]
                command("node.parameter", {"id": "mean", "key": "in_features", "value": "999"})
                nested_errors = command("analysis.diagnostics")
                nested_problems = {problem["node"]: problem for problem in nested_errors["problems"]}
                assert nested_problems["mean"]["code"] == "model.semantic"
                assert nested_problems["mean"]["causeNode"] is None
                assert any(p["causeNode"] == "mean" and p["node"] in {"encoder", "gaussian", "encoder-output"}
                           for p in nested_errors["problems"])
                command("node.parameter", {
                    "id": "mean", "key": "in_features", "value": str(old_features),
                })
                restored_vae = command("analysis.diagnostics")
                assert restored_vae["problems"] == [] and len(restored_vae["tensors"]) == 23
                command("project.save")
                nested = next(n for n in snapshot["nodes"] if n["scope"] == "encoder")
                command("ui.reveal", {"id": nested["id"]})
                before_revealed_diagnostics = command("ui.inspect")["currentScope"]
                command("analysis.diagnostics")
                assert command("ui.inspect")["currentScope"] == before_revealed_diagnostics
                assert command("ui.inspect")["currentScope"] == "encoder"
                command("ui.reveal", {"id": "no-such-node"}, expected=1)
                assert command("ui.inspect")["currentScope"] == "encoder"
                command("ui.scope", {"id": "encoder"})
                command("project.create", {"parent": str(root), "id": "vae-second", "name": "Second VAE", "template": "mnist-vae"})
                assert command("ui.inspect")["currentScope"] == ""
                command("ui.scope", {"id": "encoder"})
                command("ui.arrange")
                capture = root / "encoder.png"
                command("ui.screenshot", {"path": str(capture)})
                assert capture.read_bytes().startswith(b"\x89PNG")
                assert command("ui.inspect")["currentScope"] == "encoder"
                command("ui.scope", {"id": "decoder"})
                command("ui.scope", {"id": ""})
                command("project.save")
                command("project.close")
                # Empty graph still has an explicit graph-level Incomplete result.
                command("project.create", {"parent": str(root), "id": "empty", "name": "Empty"})
                for node in command("project.snapshot")["nodes"]:
                    command("node.remove", {"id": node["id"]})
                empty_diagnostics = command("analysis.diagnostics")
                assert empty_diagnostics["complete"] is False and empty_diagnostics["tensors"] == []
                assert any(p["node"] is None and p["package"] is None and
                           p["code"] == "model.incomplete" for p in empty_diagnostics["problems"])
                command("project.close", {"discard": True})
                print("CLI/UI resource authoring, VAE scopes and screenshot: ok")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
