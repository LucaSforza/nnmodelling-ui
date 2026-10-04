# Project lifecycle sequences

## Startup and editor synchronization

**Purpose.** Show how the Qt entry point creates one C application owner and how
the main window refreshes its graph presentation after a project becomes active.

```mermaid
sequenceDiagram
    actor User
    participant Entry as Qt main.cpp
    participant AppQt as QApplication
    participant Main as MainWindow
    participant App as NNApplication C API
    participant Scene as GraphScene QGraphicsScene
    Entry->>AppQt: construct QApplication
    Entry->>App: nn_app_new(core package root)
    Entry->>Main: construct MainWindow with application
    Entry->>AppQt: enter event loop
    alt project path supplied at launch
        AppQt->>Main: schedule openProject(path)
        Main->>App: nn_app_open(path)
    else interactive launch without a project path
        AppQt->>Main: schedule project chooser
        User->>Main: choose Open or New project
        Main->>App: nn_app_open(path) or nn_app_create(...)
    end
    alt a project becomes active
        App-->>Main: successful open or create
        Main->>Scene: refresh from copied node and edge values
        Note over Scene: Selection, routes, and preview state belong to Qt
    else no project becomes active
        Main->>Scene: keep empty graph view
    end
```

## Open and replace a project

**Purpose.** Trace project replacement through dirty-state handling and staged
loading of model, package, and dataset resources.

```mermaid
sequenceDiagram
    actor User
    participant Main as MainWindow Qt Widgets
    participant App as NNApplication C API
    participant Project as NNProject loader
    participant Catalog as Package catalog
    participant Datasets as Dataset loader
    participant Model as C graph model
    User->>Main: choose Open and project directory
    Main->>Main: confirm Save, Discard, or Cancel when project is dirty
    opt user chose Save
        Main->>App: nn_app_save()
        App-->>Main: saved or error
    end
    alt user cancelled or save failed
        Main-->>User: keep current project open
    else replacement is authorized
        Main->>App: nn_app_open(path)
        App->>Project: build staged project from model.json
        Project->>Catalog: load core and declared project packages
        Catalog->>Catalog: resolve dependencies and validate definitions
        Project->>Datasets: load declared project datasets
        Project->>Model: parse graph and validate package references
        Project-->>App: complete staged project or error
        App->>App: swap active project only after full validation
        App-->>Main: success or error, old project retained on failure
        Main->>Main: clear selection and return to root scope on success
        Main->>Main: refresh panels and fit graph on success
        Main-->>User: report open error on failure
    end
    Note over Main,App: Discard skips saving, Cancel and save failure stop before nn_app_open
```

## Create a project

**Purpose.** Show the UI gathering project identity before C creates and
activates a complete project, including built-in templates.

```mermaid
sequenceDiagram
    actor User
    participant Main as MainWindow Qt Widgets
    participant App as NNApplication C API
    participant Project as NNProject creation
    Main->>Main: confirm Save, Discard, or Cancel when replacing a dirty project
    opt user chose Save
        Main->>App: nn_app_save()
        App-->>Main: saved or error
    end
    alt replacement cancelled or save failed
        Main-->>User: keep current project open
    else replacement authorized
        User->>Main: choose parent folder, ID, name, and template
        Main->>App: nn_app_create(...) or nn_app_create_vae(...)
        App->>Project: create resources and validate staged project
        Project-->>App: complete project or error
        App->>App: replace active project only after successful creation
        App-->>Main: success or error, prior project retained on failure
        Main->>Main: clear selection, refresh panels, and fit graph on success
        Main-->>User: show creation error on failure
    end
```

## Save and close

**Purpose.** Capture the window’s Save, Discard, and Cancel behavior and the
application’s final project release.

```mermaid
sequenceDiagram
    actor User
    participant Main as MainWindow Qt Widgets
    participant App as NNApplication C API
    participant Project as NNProject persistence
    User->>Main: close application window
    alt project is clean
        Main->>App: nn_app_close(discard=false)
        App->>App: release project, report, and edit history
        App-->>Main: success or error
        Main-->>User: close window or show error and keep it open
    else user chooses Save
        Main->>App: nn_app_save()
        App->>Project: write temporary model, flush, atomically replace
        Project-->>App: success or error
        App-->>Main: saved revision or save error
        Main->>App: nn_app_close(false) only after save succeeds
        App->>App: release project, report, and edit history after successful save
        App-->>Main: success or close error
        Main-->>User: close window or keep it open and show error
    else user chooses Discard
        Main->>App: nn_app_close(discard=true)
        App->>App: release project, report, and edit history
        App-->>Main: success or error
        Main-->>User: close window or show error and keep it open
    else user chooses Cancel
        Main->>Main: ignore close event and keep window open
    end
```
