# Native client and Lua decision

Status: accepted from current request, 2026-09-28.

Superseded graphical decision 2026-09-29: keep all core/application code C11;
use Qt 6 Widgets with C++ only in gui/qt. Keep backend outside this repository for
now; future integration crosses an application adapter. Preserve package assets
and their IDs/semantics. Vendor official Lua 5.5.1 source and compile it into a
static library in this project. Lua is a full production runtime, small enough
to embed, and its MIT license is GPL-compatible. Pin archive digest in
`third_party/lua/README.md`. Do not depend on system Lua or build-time downloads.

Lua package execution will use a restricted host-provided environment, a
per-invocation instruction/memory policy, structured errors, and explicit
`tensor`/`services` bindings. Current bootstrap only proves source linkage; it
does not claim semantic parity. Avoid loading untrusted packages until those
contracts and limits are implemented and tested. Lua standard I/O, OS, package
loader and debug APIs are not exposed to package rules.

Qt 6 Widgets is an external dependency of the optional GUI target only. C core
and tests configure without enabling C++ or discovering Qt. justfile invokes
CMake targets. Public C headers have extern-C guards; no Qt types cross them.

Schema-v2 JSON uses vendored yyjson 0.13.0 (MIT) to avoid a runtime system
parser dependency. Legacy font/rasterization assets remain preserved under third_party; Qt owns
font rendering. Neither asset defines domain semantics.

Source: [Lua official download](https://www.lua.org/download.html),
[Lua embedding/build guidance](https://www.lua.org/manual/5.5/readme.html),
[Lua license](https://www.lua.org/license.html).

Accepted 2026-10-05 user request supersedes keeping backend outside repository:
local Python/FastAPI and container execution live beside native client per
[backend contract](../contracts/backend.md). C11 graph/Lua ownership remains.
