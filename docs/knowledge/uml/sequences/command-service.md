# Local command service sequences

## Request transport

**Purpose.** Trace line-delimited requests from nnmodelctl to the nonblocking
C service polled on the Qt application thread.

```mermaid
sequenceDiagram
    actor Agent as User or LLM
    participant CLI as nnmodelctl C CLI
    participant Socket as AF_UNIX socket
    participant Timer as MainWindow QTimer
    participant Service as NNAutomation C service
    participant Main as MainWindow Qt Widgets
    Agent->>CLI: operation name and JSON arguments
    CLI->>Socket: connect, write request, await response
    Timer->>Service: nn_automation_poll()
    Service->>Socket: accept and read request
    Service->>Service: validate line-delimited JSON
    Service->>Main: dispatch typed command or UI callback
    Main-->>Service: result, snapshot, diagnostics, or screenshot path
    Service->>Socket: write response and close client
    Socket-->>CLI: newline-terminated JSON
    CLI-->>Agent: print result and exit status
    Note over Timer,Service: Polling runs on Qt thread, request handling stays nonblocking
```

## Route a request

**Purpose.** Identify the distinct model, query, and UI paths selected by the
C dispatcher after a request reaches the service.

```mermaid
sequenceDiagram
    participant Service as NNAutomation C service
    participant Dispatch as nn_automation_dispatch C
    participant Commands as nn_automation_execute C
    participant App as NNApplication C API
    participant Main as MainWindow Qt Widgets
    participant Scene as GraphScene Qt
    participant Timer as MainWindow QTimer
    participant Socket as AF_UNIX socket
    Timer->>Service: nn_automation_poll()
    Service->>Dispatch: validated request
    alt model command
        Dispatch->>Commands: execute typed command
        Commands->>App: call same operation as native UI
        App-->>Commands: result or error
        Commands-->>Dispatch: command result
    else project snapshot or analysis query
        Dispatch->>App: read-only snapshot or report query
        App-->>Dispatch: snapshot, diagnostics, or error
    else ui.inspect
        Dispatch->>Main: semantic widget-tree callback
        Main-->>Dispatch: IDs, roles, labels, and current scope
    else ui.screenshot
        Dispatch->>Main: screenshot callback, arrange if requested
        Main->>App: grouped C moves when arrangement requested
        Main->>Scene: refresh graph after pending changes
        Main->>Main: activate layouts, process Qt events, capture window
        Main-->>Dispatch: saved image path or error
    else other UI operation
        Dispatch->>Main: scope, arrange, or reveal callback
        Main-->>Dispatch: result or error
    end
    Dispatch-->>Service: JSON result and changed status
    Service-->>Timer: poll result and changed status
    opt successful mutating model or UI operation
        Timer->>Main: refreshAll()
        Main->>Scene: refreshAll graph and routes
    end
    Service->>Socket: write JSON response and close client
    Note over Dispatch,App: Model commands and read-only queries use the shared C application owner
```
