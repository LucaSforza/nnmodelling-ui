# Resource authoring and local automation

Project resources are written by C project code only. Qt forms and
`nnmodelctl` submit typed payloads to the same `NNApplication` operations.
Sources: `src/application/resources.c`, `src/project/resources.c`,
`src/catalog/`, `src/automation/`, `src/nnmodelctl/` and the Qt resource panels.

## Create a dataset or stereotype

```mermaid
  sequenceDiagram
    actor Author as User or local model author
    participant Entry as Qt form or nnmodelctl
    participant App as NNApplication
    participant Project as NNProject resource transaction
    participant Catalog as Package and dataset catalogs
    Author->>Entry: submit typed definition and identity
    Entry->>App: createDataset or createStereotype
    App->>Project: validate safe ID, path, schema and destination
    Project->>Project: exclusively create new resource directory and files
    Note over Project: New files are rollback-owned until model save succeeds
    Project->>Catalog: build candidate active catalogs
    Catalog->>Catalog: resolve unique dependencies and validate graph references
    alt candidate valid
      Catalog-->>Project: validated candidate resources
      Project->>Project: temporarily swap candidate catalogs and manifest references
      Project->>Project: atomically save candidate model.json
      Project->>Project: retain new state and release old catalogs on success
      Project-->>App: committed resource
      App-->>Entry: success and all current graph edits are saved
    else schema, dependency or write failure
      Catalog-->>Project: explicit error
      Project->>Project: restore old catalogs, references and dirty state if swapped
      Project->>Project: remove only files created by this transaction
      Project-->>App: failure and active model and previous model.json preserved
      App-->>Entry: visible validation/write error with form fields retained
    end
```

Creation is a history barrier because the operation saves the resource files
and existing graph edits together. Python dataset and stereotype stubs declare
`nnmodelling-runtime` and raise `NotImplementedError` until implemented;
native C and the Qt client never execute those modules. Dataset selection is a
separate exact ID/version operation that updates manifest state and invalidates
the lazy analysis report.

Preflight identity/schema/destination failures return before creating files or
loading candidate catalogs. After directory creation, each write/catalog/save
failure enters rollback. `nn_project_create_dataset` and
`nn_project_create_stereotype` temporarily install candidate resource state for
`nn_project_save`, then retain it only when the atomic model save succeeds.

## Resolve a project on open

```mermaid
  sequenceDiagram
    participant Project as C project loader
    participant Manifest
    participant Packages as PackageCatalog
    participant Datasets as DatasetCatalog
    participant Model as NNModel
    Project->>Manifest: parse schema-v2 references
    Manifest->>Packages: load declared core and project packages
    Packages->>Packages: validate identity, dependency closure and cycles
    Manifest->>Datasets: load declared project datasets
    Datasets->>Datasets: validate exact identity and relative paths
    Packages-->>Project: staged package set or error
    Datasets-->>Project: staged dataset set or error
    Project->>Model: validate nodes, handles, scopes and input bindings
    alt every stage succeeds
      Model-->>Project: candidate graph
      Project->>Project: publish staged project atomically
    else any resource or graph stage fails
      Project->>Project: discard candidate state
    end
```

Catalog activation is limited to declared project resources and the exact
required core closure. Ambient directories are not activated. See
`contracts/model.md` and [project UML](../project.md).

## CLI request and UI introspection

```mermaid
  sequenceDiagram
    actor Author
    participant CLI as nnmodelctl
    participant Socket as AF_UNIX local transport
    participant Adapter as CommandAdapter
    participant App as NNApplication
    participant UI as Qt widgets
    Author->>CLI: operation and JSON arguments
    CLI->>Socket: one bounded newline-delimited request
    Socket->>Adapter: parse and dispatch on Qt application thread
    alt project/model/resource operation
      Adapter->>App: invoke same C operation used by Qt
      App-->>Adapter: result, diagnostics or visible error
    else ui.inspect
      Adapter->>UI: snapshot semantic widget tree
      UI-->>Adapter: stable IDs, roles, labels and current scope
    else ui.screenshot
      Adapter->>UI: refresh and optionally arrange and frame
      UI->>UI: finish layout before capture
      UI-->>Adapter: captured frame or error
    end
    Adapter-->>Socket: JSON success/error response
    Socket-->>CLI: final newline-delimited response
    CLI-->>Author: JSON and exit status
```

The socket is private to the current user and enabled only with `--socket`.
Requests are bounded, dispatched synchronously on the existing application
thread, and cannot create a second graph owner. `ui.inspect` is read-only.
Capture waits for layout/frame completion; optional arrangement persists the
current scope's model positions before capture. Unknown or rejected operations
return errors without partial mutation.
