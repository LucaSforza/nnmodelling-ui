# UI, resources, VAE and CLI implementation

Accepted 2026-09-30. Principal owns contracts, examples, CLI/service, integration
tests and review. Three disjoint native background workers:

1. C project/catalog/application: pretty JSON, position metadata, resource
   transactions, dataset selection and VAE template creation.
2. C inference: recursive bounded subflow service and focused inference tests.
3. Qt: contrast, positioned cards, resource forms, VAE action, scope controls
   and narrow local automation integration.

Principal provides complete VAE project-owned assets, implements C11 Unix
service/nnmodelctl, tests resources and end-to-end CLI, and runs just test,
just test-ui, just test-swarm and git diff --check. Review rendered evidence.
No backend/training, no core asset changes, one final commit.

## Completion

Implemented all three bounded worker assignments and principal integration.
Reviewed resource rollback/default validation, recursive inference status classes
and budgets, true key/value badges, native dark-desktop contrast and CLI dirty
replacement semantics. All eight C tests and three GUI/CLI tests pass; native
Wayland launch and root/encoder/decoder captures were inspected. Clang ASan/UBSan
and leak detection pass all C tests; GCC sanitizer runtimes are unavailable.
Full command/evidence record: [QA results](../knowledge/testing/qa-results.md).
