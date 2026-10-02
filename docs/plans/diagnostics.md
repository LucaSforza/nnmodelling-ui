# Diagnostics and common helpers

UML maintenance 2026-10-02: the original diagnostics diagram referenced below
has been consolidated into ../knowledge/uml/editor.md (ownership), metamodel.md
(report types) and sequences.md (analysis/navigation); the contract is unchanged.

Accepted 2026-09-30. User subsequently requested Luna implementation workers;
principal retains KB/UML, coordination and final review. SDS adoption withdrawn.

1. Establish contracts/diagnostics.md and uml/diagnostics.md before code.
2. Remove SDS; centralize bounded errors, remove custom JSON text builders,
   make partial allocation cleanup safe and share strict join-handle parsing.
3. Extend C outcomes with Lua syntax category, location and root causes;
   application owns cache and exposes identical read-only CLI diagnostics.
4. Replace success list with categorized/navigable model problems, retain types
   in inspector, mark cards, show command rejection dialogs, retain authoring.
5. Add regression tests, run core/Qt/CLI and sanitizer checks, inspect rendering,
    review diff and commit. Record actual results below when complete.

Disjoint ownership: core worker (utils/inference/application and their tests),
Qt worker (Qt frontend and Qt window test), test worker (build/vendor removal,
allocation and CLI tests). Principal owns project/catalog cleanup, automation/CLI
dispatch/help and KB. Test-worker permissions blocked C source edits; ownership
was reassigned without bypassing tooling permissions.

Completed 2026-09-30. Principal reviewed all worker results, fixed partial
project-allocation publication/cleanup and shared command response serialization,
and reran `just test` (10/10), `just test-ui` (3/3), Clang ASan/UBSan with leak
checks (10/10) and diff-check. Visible native Wayland and small offscreen review
found and fixed cramped diagnostics, missing blocked-child grouping and Lua
reason elision. Evidence and remaining coverage limits are recorded in
../knowledge/testing/qa-results.md. SDS is removed, not merely hidden in headers.
