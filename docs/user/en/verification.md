# Guide verification

Verification performed on 6 October 2026. Descriptions follow the KB contracts and current implementation; images show the real Qt client, controlled with computer use through the local noVNC bridge.

## Checks performed

- Client build: `CCACHE_DISABLE=1 just build` succeeded.
- Core suite: 13 of 13 tests passed; GUI suite: 6 of 6 passed. The `just test` and `just test-ui` recipes first found that fixtures expected the temporary `/tmp/opencode` directory. After creating it, the tests were rerun with `ctest --test-dir build/core --output-on-failure` and `ctest --test-dir build/qt -L gui --output-on-failure`.
- GUI navigation through menus, scopes, Inspector, datasets, stereotype references, join handles, diagnostics and the Training panel. Demonstration edits were undone; the project was a local copy of the example.
- Isolated backend on loopback, port 8766, with a dedicated temporary store. Health, GUI submission, history, metrics and wheel download were checked against the real service.
- Job `86caba53-0690-4c79-b948-4273ed1426b4`: three epochs, batch 16, learning rate 0.001, seed 0, publish every step. It completed with 12 updates, final mean training loss 3.462321, validation loss 3.387799 and test loss 3.345268.
- Wheel downloaded from the client: the model loaded without the backend and the tutorial inference loop ran successfully for 40 characters. For this check the wheel was extracted to a temporary directory and imported with workspace dependencies; installation in a fresh venv is the suggested reader procedure and was not separately tested.
- The server chapter's API example was tried against an isolated backend: submit a short job, wait for completion, and retrieve its snapshot, weights and wheel.
- HTML regenerated and checked with `build.py --check`: pages, local links, anchors and images. Both languages and same-chapter switching were verified; desktop and mobile layouts were checked through computer use. Shell and Python command syntax was also checked in the translation.

## Wheel filename check (6 October 2026)

An export of the tiny LLM graph with temporary project ID `llm` and previously
trained weights from historical job `86caba53-0690-4c79-b948-4273ed1426b4`
produced `nnm_llm-0.1.0-py3-none-any.whl`, retaining module
`nnmodel_job_86caba53_0690_4c79_b948_4273ed1426b4`. The wheel ran without the
backend and produced an eight-character continuation. No new training, worker
image rebuild or screenshot capture was performed; the tutorial images remain
the historical ones identified above. The backend (15 tests), runtime (15)
and examples (32) suites passed.

## Observed limits

The full-corpus, 20-epoch path is documented by the example files and preparation script; it was not rerun during this session.

The contract describes Network 3D, but this interface does not expose its controls. The guide reports this without describing inaccessible features.

The stereotype form was captured in full using a 1600-pixel-high virtual display. During tests to close this form through the bridge, Escape terminated the Qt session. Creation and cancellation are described from code; they are not reported as successful GUI operations. The illustrated dataset form is the edit form: creation shares its fields and tables, with the differences noted in the text.

The checks did not modify original examples or the model excluded from the request. Original screenshots, the demonstration copy and artifacts remain outside versioned files. Temporary verification services were stopped when work ended.
