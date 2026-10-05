# Training dashboard and executable examples

Accepted 2026-10-05: contracts/backend.md defines graph visualization, optimizer
step publication, dataset preparation and executable example semantics.

Order: principal updates contracts first; Luna organizes sequence diagrams and
validates/formats Mermaid; principal reviews diagrams against source/contract;
then complete and test every example, actual container training including a small
Tiny Shakespeare LLM, native dashboard trial, first commit. Accepted user follow-up:
only after that commit, train the same LLM on the entire Tiny Shakespeare corpus
for 20 epochs with disjoint train/validation/test splits, record reproducible
configuration and actual results, then create a second commit.

Documentation implementation may be delegated by the explicit current user
request; principal retains normative decisions and final corrections. Dedicated
uml/sequences/ directory separates lifecycle, editing/analysis, resource/IPC,
training/container/curves/export and 3D flows. sequences.md remains an index with
compatible existing anchors. Each sequence names actual submodules, operation
ordering, failures, ownership/lifetimes and source entrypoints. Accepted future
dashboard behavior is identified until implementation arrives. No invented direct
backend-to-UI push or Python execution in the native client.

Mermaid tool: reuse installed Mermaid parser in tools/mermaid-check/node_modules;
add a small reproducible CLI/package manifest/lock and just recipe if no existing
wrapper exists. Validate Mermaid fences and normalize their formatting with an
explicit write option. Commit tool source/metadata only, never node_modules.
Both delegated author and principal must run it and repair actual parse errors.

File ownership: documentation worker owns uml sequence documents and Mermaid
tool/test recipe; backend worker owns step metrics/API/worker/tests; Qt worker
owns dashboard/widgets/tests. Example/runtime worker follows after UML review,
owns examples, generic runtime fixes and example tests. Shared justfile changes
serialize; Git staging and final commit serialize after verification. Principal
edits only KB, preserving unrelated user changes.

## Acceptance matrix

For each example: native open and Lua diagnostics, every active resource's Python
entrypoint, raw-input adapter round trip, finite scalar objective/backward with
registered trainable parameters, independent inference, and standalone wheel
installation/inference without training payloads. Run a real isolated CPU job
for each; retain job identifiers, actual split counts, step/epoch metrics, final
test loss and downloadable artifacts. LLM must use real Tiny Shakespeare and
multiple epochs; a toy run proves execution, not language-model quality.

Check paired points at requested optimizer cadence plus unique epoch-end flush,
sample-weighted losses, legacy epoch-only jobs, and unchanged stochastic training
weights when reporting cadence changes. Test API invalid configuration and
malformed/oversize metrics. Qt checks published-step parsing, cadence payload,
series controls, linear/log ranges and empty/single/legacy plots. Fresh native
browser trial must show history, both curves, controls, log and test loss.

Review all sequence diagrams against actual source ordering/ownership, validate
all KB Mermaid fences through the committed wrapper, and test formatter
idempotence/parser failures. Final gates: core, Qt, backend, runtime/examples,
Mermaid tool tests, and git diff --check. Preserve baseline public behavior and
core package bytes; no package-ID workaround in the generic executor.

## Verification record, 2026-10-05

Integrated Python gates: 45 passed. Fresh native gates: core 13/13 and Qt 6/6.
Mermaid parser: 35 diagrams in 39 KB files; formatter/parser regressions: 2/2.
All five numerical graphs perform finite forward/backward and update weights;
all five exported wheels install and infer outside the checkout without the
public runtime SDK. Native open/Lua diagnostics passed for all five examples.
The LLM trial exposed missing view metadata in newly authored resources; repaired
definitions now open and analyze every nested scope with zero problems. Confirmed
head output [B,128,16], causal scores [B,128,128], logits [B,128,65], scalar loss.

Actual Qt browser trial, fresh build: Training opens the dashboard, readable
container readiness replaces raw JSON, history remains visible, legacy epoch
curves and new optimizer-step curves render, training toggle and logarithmic
scale work, and final test loss/logs remain visible. Container proofs below use
three epochs, batch 8, Adam learning rate 0.001, seed 0, cadence 1.

| Example | Job ID | Published steps | Test loss |
| --- | --- | ---: | ---: |
| local-training | 5f8ee99c-b407-4fba-b5f6-53b0cdb5ec6c | 3 | 182.9770813 |
| mnist-mlp | 97f32945-7ddc-4f10-8f67-820025a04839 | 24 | 1.8929758668 |
| mnist-vae | 7008f46a-db6e-4ab7-91f1-4214030bd3cf | 24 | 0.2163545266 |
| rnn-sine | 72ed8275-62c2-4759-a8c5-0c629a5f65ca | 24 | 0.0633328445 |
| tiny-decoder-llm | da96709c-b022-4556-add1-fe0c74c609d5 | 24 | 3.066873312 |

Completed jobs above have exact model/resource snapshot checks and successful
weights/wheel downloads. Trained wheels infer raw numeric/PIL/sequence inputs
from isolated installation targets, with both bundled and explicit alternate
weights paths; the public runtime SDK cannot be resolved in those processes.
The LLM wheel accepts raw text and returns decoded text. All five jobs retain
three epoch summaries; cadence 1 publishes exactly steps 1 through the last
optimizer update, without duplicate epoch-tail points.
Third-party dependencies come from the locked workspace. Screenshot evidence
is local/ignored at `.computer-use/training-dashboard.jpg` (MNIST MLP curves).
The browser test project was closed through the GUI and its bridge stopped.

Worker image: `961783c493737136ddf4459f8fa81fe812b0c4d4bef826062273dee18a0fdd77`.
Service runs on loopback port 8765; container isolation remains the production
path. Test job snapshots are immutable and separate from later editor changes.
