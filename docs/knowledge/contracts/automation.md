# Local command service (accepted 2026-09-30)

Explicit user request replaces deferred nnmodelctl status. Unix-only local
automation is implemented on Linux; no backend or network server. C11 owns IPC, JSON
dispatch and CLI; Qt runs the service on its application thread and handles
UI introspection/layout/screenshot. One application remains authoritative.

UI starts service only with `--socket PATH`; nnmodelctl requires `--socket PATH`
or NNMODELLING_SOCKET. Socket is AF_UNIX, mode 0600, owned by current UID. Refuse
existing path, symlinks and overlong paths; clean up only own socket on exit.
Requests are newline-delimited JSON, one connection/request, <=2 MiB. Malformed,
oversized, unknown or invalid requests return explicit error without mutation;
idle/partial clients cannot block UI. At most eight concurrent clients, a
five-second request/response deadline and 64 KiB per-client I/O budget per poll
apply. Dispatch is guarded against reentrancy during screenshot event flushing.
Response: {ok:true,result:...} or {ok:false,error:string} plus
final newline. Dispatch is synchronous on application thread; no other graph.
Duplicate object keys, embedded NUL strings and JSON nesting deeper than 64 are
rejected. CLI waits at most ten seconds for I/O and bounds response to 16 MiB.

`nnmodelctl --socket PATH OPERATION [JSON_OBJECT]` sends
{operation:OPERATION,args:JSON_OBJECT}; stdin via `-` is supported. Default args
is {}. Prints JSON; exit 0 success, 1 command failure, 2 usage/transport failure.
CLI help lists operations and resource payloads. Payload keys:

* project.open {path}; project.create {parent,id,name,template:blank|mnist-mlp|
  mnist-vae}; project.save {}; project.close {discard:boolean=false}.
  Open/create reject replacement of a dirty active project without modal UI;
  caller must first save or explicitly close with discard=true. Successful
  replacement resets scope to root. Read-only snapshot/inspect do not change
  selection or dirty state.
* project.snapshot {} returns identity, dirty, active dataset, package identities,
   dataset slots, nodes (IDs, scopes, package refs, positions, parameters,
   boundaryHandle), edges. Packages include normalized typed outputs.
* stereotype.create {id,version,definition:object,inference:string,
  dependencies:object={}} invokes resource transaction.
* dataset.create {id,version,definition:object,select:boolean=false};
  dataset.select {id,version}.
* node.add {id,package,version,scope:string="",x:number=0,y:number=0};
  node.remove {id}; node.move {id,x,y}; node.rename {id,name};
   node.parameter {id,key,value:string} uses same text parser as inspector.
* node.boundary {id,handle} edits nested terminal mapping per typed-outputs.md.
* edge.connect {id,source,sourceHandle,target,targetHandle}; edge.disconnect {id}.
* ui.inspect {} returns stable semantic widget IDs/roles/labels and current scope.
* ui.scope {id:string} navigates root (empty) or existing scope.
* ui.arrange {} arranges current scope vertically and frames the complete
  model-backed layout, following editor.md.
* ui.screenshot {path,arrange:boolean=false} refreshes scene/widgets and commits
  widget layout before capture. Optional arrange=true first arranges current
  scope vertically and frames content; it never performs Input-focused Fit.
* analysis.diagnostics {} returns the same structured C problems and successful
  tensors used by the UI, per diagnostics.md; it is read-only.
* ui.reveal {id} selects/centers an existing node, opening its scope as needed.

Qt refreshes panels after successful commands; invalid scope rejected. GUI
operations use a narrow callback, never Qt types in C ABI. No arbitrary widget
clicking, shell execution, Python, network, secrets or remote training commands.
Resource create saves all current edits as defined in resource-authoring.md.
CLI can start from blank project, inspect UI, author resources, add nodes,
enter subflow scopes, save and capture. The UI must already run; launching
processes from command service is not authorized.
