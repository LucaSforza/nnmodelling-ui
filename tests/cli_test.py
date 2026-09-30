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
                definition = {
                    "name": "Local shape rule", "description": "Pass-through only", "kind": "layer",
                    "view": {"color": "#4779c4", "width": 220, "height": 100},
                    "parameters": {"features": {"type": "integer", "minimum": 1, "default": 8, "position": "bottom"}},
                }
                command("stereotype.create", {
                    "id": "local.shape", "version": "0.1.0", "definition": definition,
                    "inference": "return function(context, parameters, services) return {status='success',output=context.inputs[1]} end",
                    "dependencies": {},
                }, stdin=True)
                command("dataset.create", {
                    "id": "local.images", "version": "0.1.0", "select": True,
                    "definition": {"name": "Images", "batch": {
                        "inputs": {"image": {"dtype": "float32", "shape": ["B", 8]}}, "targets": {},
                    }},
                })
                command("node.add", {"id": "input", "package": "core.input", "version": "0.1.0"})
                command("node.add", {"id": "local", "package": "local.shape", "version": "0.1.0", "y": 200})
                command("edge.connect", {"id": "edge", "source": "input", "sourceHandle": "out", "target": "local", "targetHandle": "in"})
                command("node.parameter", {"id": "local", "key": "features", "value": "16"})
                command("node.parameter", {"id": "local", "key": "features", "value": "0"}, expected=1)
                command("ui.scope", {"id": False}, expected=1)
                command("unknown.operation", expected=1)
                snapshot = command("project.snapshot")
                assert len(snapshot["nodes"]) == 2 and len(snapshot["edges"]) == 1
                assert next(n for n in snapshot["nodes"] if n["id"] == "local")["parameters"]["features"] == 16
                command("project.save")
                saved = root / "authored" / "model.json"
                assert saved.read_text().startswith("{\n  ") and saved.read_text().endswith("\n")
                command("project.close")
                command("project.open", {"path": str(saved.parent)})
                assert len(command("project.snapshot")["nodes"]) == 2
                info = command("ui.inspect")
                assert {"palette", "resources", "inspector"} <= {w["id"] for w in info["widgets"]}
                command("node.move", {"id": "local", "x": 40, "y": 200})
                command("project.open", {"path": str(saved.parent)}, expected=1)
                assert command("project.snapshot")["dirty"] is True
                command("project.save")

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
                assert len(snapshot["nodes"]) == 21
                assert sum(n["package"]["id"] == "core.subflow-proxy" for n in snapshot["nodes"]) == 2
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
