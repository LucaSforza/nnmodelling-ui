# Qt frontend migration

Accepted 2026-09-29, in progress.

1. Assess existing C model/application rules and SDL graphical responsibilities.
2. Update contracts/UML for strict Qt GUI -> C application -> C core dependency.
3. Extract validated C application API and add independent core tests.
4. Implement retained Qt scene, scope-aware subflows, palette/inspector/resources.
5. Replace SDL build with optional Qt target and C-only library/test configuration.
6. Run core/GUI gates, inspect screenshots, remove replaced SDL code, record QA.

Existing compiler/IR and cross-scope execution are not implemented in this
checkout; migration preserves existing semantics rather than inventing them.
Backend, training and command service remain deferred.
