set shell := ["bash", "-ec"]

default: build

# Core configuration does not enable C++ or discover Qt.
core:
    cmake -S . -B build/core -DNN_BUILD_GUI=OFF -DBUILD_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build/core --parallel 4
    ln -sfn build/core/compile_commands.json compile_commands.json

test: core
    ctest --test-dir build/core --output-on-failure -L core

build:
    cmake -S . -B build/qt -DNN_BUILD_GUI=ON -DBUILD_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build/qt --parallel 4
    ln -sfn build/qt/compile_commands.json compile_commands.json

run *args: build
    ./build/qt/nnmodelling-ui {{args}}

test-ui: build
    ctest --test-dir build/qt --output-on-failure -L gui

test-cli: build
    ctest --test-dir build/qt --output-on-failure -R cli_ui

# Count C and C++ source/header lines, excluding tests and generated build trees.
count-lines:
    find . -type d \( -name .git -o -name build -o -name tests \) -prune -o -type f \( -name '*.c' -o -name '*.h' -o -name '*.C' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.h++' -o -name '*.c++' \) -print0 | xargs -0 wc -l

# Repository-local OpenCode V2 development tooling.
swarm-setup:
    node -e 'import("./.opencode/swarm-mailbox/index.js").then(p => console.log(p.default.id))'

test-swarm:
    node --test tests/swarm_mailbox_test.mjs
