---
name: nnmodelling-backend
description: Run or diagnose the local NNModelling FastAPI training service and its isolated CPU worker containers.
---

# NNModelling local backend

Use the repository's pinned uv workspace. The API runs on loopback by default at `http://127.0.0.1:8765`; OpenAPI docs are at `/docs`.

```sh
just backend-sync
just backend-image
just backend-run
```

`backend-run` stays in the foreground. Stop it with Ctrl-C. Set `NNMODELLING_BACKEND_HOST`, `NNMODELLING_BACKEND_PORT`, or `NNMODELLING_JOB_ROOT` to change the bind address, port, or persistent job directory. Keep the host on loopback for this single-owner local service. Set `NNMODELLING_BEARER_TOKEN` before startup to require `Authorization: Bearer …`; the token is held in process memory and is never stored in job snapshots. Use `NNMODELLING_CONTAINER_RUNTIME` when the Docker-compatible CLI is not named `docker`. The worker image name can be changed with `NNMODELLING_WORKER_IMAGE` and must match at build and run time.

`GET /health` reports whether the container engine is reachable. Jobs and immutable snapshots persist under `~/.local/share/nnmodelling/jobs` by default. `GET /v1/jobs` lists jobs; `GET /v1/jobs/{id}` shows state and epoch metrics; `POST /v1/jobs/{id}/cancel` requests cancellation. Completed jobs expose `weights.safetensors` and a standalone wheel at their `/weights` and `/wheel` routes. Worker output is in each job's `output/worker.log`.

Build the unpublished runtime SDK wheel for local resource-authoring environments with `just runtime-wheel`; install the generated wheel with `uv add /path/to/nnmodelling_runtime-*.whl`. The runtime is part of this workspace and is not published to a package registry by these commands.

Worker images start with the workspace runtime, CPU PyTorch, safetensors, and NumPy. A resource that declares additional Python packages must have those packages baked into the worker image before its job is submitted. For example, create a local `Dockerfile.worker-extra` beside the repository Dockerfile:

```dockerfile
FROM nnmodelling-worker:local
USER 0
RUN uv pip install --python /app/.venv/bin/python 'Pillow==11.3.0'
USER 65532:65532
```

Build the base image with `just backend-image`, then build the extension as `docker build -f Dockerfile.worker-extra -t nnmodelling-worker:pillow .`. Set `NNMODELLING_WORKER_IMAGE=nnmodelling-worker:pillow` when starting the service (for example, `NNMODELLING_WORKER_IMAGE=nnmodelling-worker:pillow just backend-run`). Keep resource dependencies pinned in the image recipe and rebuild after changing them. Training containers run without network access; a missing dependency appears as a failed job with the worker import error.
