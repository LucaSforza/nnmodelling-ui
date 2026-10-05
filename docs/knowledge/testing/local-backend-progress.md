# Local backend implementation

Accepted 2026-10-05. Contract: ../contracts/backend.md. UML:
../uml/backend.md. User explicitly requests GPT-6 Luna high workers;
principal owns KB/contracts and review, workers own all implementation.

1. Python worker: reusable generic dataset SDK, stereotype ABI, generic PyTorch
   scope DAG, prediction/objective execution and standalone safetensors wheel.
   Owns python/nnmodelling-runtime and runtime tests.
2. Backend worker: uv workspace, FastAPI protocol/persistence, container queue,
   training and artifacts, just recipes, local start and backend management skill.
   Owns backend, workspace/lock/build recipes, backend tests and local example.
3. Native worker: Qt Network backend/job/loss/download/restore UI and C uv
   resource scaffolds, transaction and HTTP tests. Owns Qt, C resources and CMake.

No shared Git staging/commits; preserve unrelated user tools directory. Review
cross-worker ABI before integration, then core/Qt/Python tests, standalone wheel,
actual container job and running local service. Missing environment capabilities
must be recorded, not replaced with alternate production execution paths.

Baseline 2026-10-05: just test passed 13/13 before implementation. Local Docker
CLI is Podman 5.8.7 compatibility; rootless runtime available when executed
outside filesystem sandbox. Compiler/cache similarly requires normal escalation.

## Completed verification (2026-10-05)

Core suite passed 13/13 after resource-authoring changes. Qt suite passed 6/6
after the final Training toolbar and metric-scroll changes. ASan/UBSan/leak
core suite passed 13/13, including allocation-failure coverage. Final combined
Python suite passed 14/14 (8 backend, 6 runtime); Python 3.14 emitted upstream
FastAPI asyncio.iscoroutinefunction deprecation warnings. Skill quick_validate
and git diff --check passed.

Runtime tests cover real MNIST graph forward/backward, ordered joins, typed
multi-output branches, repeat parameter independence, nested scopes and loss
pruning. Two generated wheels install and infer together in a separate target
without the public SDK source, preserve inference assets, exclude training data
and support alternate safetensors. Generic interpreted DAG remains the accepted
first implementation; performance TODO is in contracts/backend.md and beside
the executor.

Built CPU image nnmodelling-worker:local with Podman 5.8.7, Python 3.12,
torch 2.14.1+cpu, FastAPI 0.115.14, safetensors 0.8.0 and NumPy 2.5.3.
Image ID: 719bbda48af890ef2449eb0a85bc3dc78681a00f63c24d2b84cd4f33628f48ee.
Qt submitted job 5405e245-2203-4be0-a91b-136d783d39d0 using
examples/local-training: three epochs, batch size 2, learning rate 0.05, seed 0.
Actual container completed with training loss 16.1427804 to 12.9074019,
validation loss 81.8691406 to 67.3522186, test loss 137.6125488.
Downloaded wheel installed and Model().infer(1.0) returned [1.0331940651].
Separate 10,000-epoch job 5b0204ff-836f-45b2-85aa-5f69ff4f7ea2 reached
running, accepted HTTP cancellation, finalized cancelled and left no container.

Actual Qt/noVNC trial used a fresh rebuilt executable and disposable project.
Training button opens the backend dialog; connection, states, epoch/final losses,
safetensors/wheel downloads and restore into a new project directory succeeded.
Supplemental comparisons confirmed downloaded artifact bytes and all four
restored resource files equal the API snapshot; model JSON equals the historical
job and differs from the subsequently edited working diagram.
After moving and saving a graph node, historical restore displayed its original
position. Polling preserves metric scroll. Local evidence screenshots are
.computer-use/backend-training.jpg and .computer-use/backend-restored.jpg;
test bridge was stopped and its temporary browser tab closed.

OpenAPI service is launched detached locally on 127.0.0.1:8765, with process ID
recorded in .computer-use/backend.pid and log in .computer-use/backend.log.
Startup/shutdown and
dependency-image instructions are in .agents/skills/nnmodelling-backend/SKILL.md.
Remote multi-user hosting and distributed/GPU scheduling remain deferred.
