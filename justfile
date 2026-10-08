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

backend-sync:
    uv sync --all-packages --group dev --group examples

backend-image:
    "${NNMODELLING_CONTAINER_RUNTIME:-docker}" build -f backend/Dockerfile -t "${NNMODELLING_WORKER_IMAGE:-nnmodelling-worker:local}" .

backend-run:
    uv run --package nnmodelling-backend --no-sync uvicorn backend.app:app --host "${NNMODELLING_BACKEND_HOST:-127.0.0.1}" --port "${NNMODELLING_BACKEND_PORT:-8765}"

# Backend stays local; only isolated workers run on the configured Slurm cluster.
backend-slurm:
    NNMODELLING_EXECUTOR=slurm NNMODELLING_SLURM_HOST="${NNMODELLING_SLURM_HOST:-cluster}" uv run --package nnmodelling-backend --no-sync uvicorn backend.app:app --host "${NNMODELLING_BACKEND_HOST:-127.0.0.1}" --port "${NNMODELLING_BACKEND_PORT:-8765}"

train-cluster-examples:
    uv run --no-project python tools/train_cluster_examples.py --wait

test-vae-consumer:
    cd examples/implementation/vae && uv run --no-project python download_wheel.py
    cd examples/implementation/vae && uv sync --locked
    cd examples/implementation/vae && uv run --locked python main.py

test-backend:
    uv run --package nnmodelling-backend --group dev pytest -q tests/test_backend.py tests/backend

test-runtime:
    uv run --group dev pytest -q tests/test_runtime.py tests/test_operations.py

test-examples:
    uv run --group dev --group examples python -m pytest -q tests/test_examples.py tests/test_example_datasets.py tests/test_example_stereotypes.py tests/test_full_llm_preparation.py tests/test_llm_wheel_consumer.py tests/test_vae_wheel_consumer.py tests/test_cluster_examples.py

test-llm-consumer:
    cd examples/implementation/llm && uv run --no-project python download_wheel.py
    cd examples/implementation/llm && uv sync --locked
    cd examples/implementation/llm && uv run --locked python main.py --tokens 8

runtime-wheel:
    uv build --package nnmodelling-runtime --out-dir dist

# Clang is the default because the local GCC sanitizer runtimes may be absent.
test-sanitize:
    cmake -S . -B build/sanitizers -DNN_BUILD_GUI=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER="${NN_SANITIZER_CC:-clang}" -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
    cmake --build build/sanitizers --parallel 4
    ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build/sanitizers --output-on-failure -L core

# Count project source lines by language and owned subtree.
count-lines:
    @set -euo pipefail; \
    c_output=$(find src -type f \( -name '*.c' -o -name '*.h' \) -print0 | sort -z | xargs -0 -r wc -l); \
    cpp_output=$(find src -type f \( -name '*.C' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.h++' -o -name '*.c++' \) -print0 | sort -z | xargs -0 -r wc -l); \
    python_output=$(find backend python/nnmodelling-runtime -type d \( -name .git -o -name build -o -name .venv -o -name venv -o -name __pycache__ \) -prune -o -type f -name '*.py' -print0 | sort -z | xargs -0 -r wc -l); \
    stereotype_output=$(find stereotype-packages -type d \( -name .git -o -name build -o -name .venv -o -name venv -o -name __pycache__ \) -prune -o -type f \( -name '*.lua' -o -name '*.py' \) -print0 | sort -z | xargs -0 -r wc -l); \
    c_count=$(printf '%s\n' "$c_output" | awk 'NF && $2 != "total" { sum += $1 } END { print sum + 0 }'); \
    cpp_count=$(printf '%s\n' "$cpp_output" | awk 'NF && $2 != "total" { sum += $1 } END { print sum + 0 }'); \
    python_count=$(printf '%s\n' "$python_output" | awk 'NF && $2 != "total" { sum += $1 } END { print sum + 0 }'); \
    stereotype_count=$(printf '%s\n' "$stereotype_output" | awk 'NF && $2 != "total" { sum += $1 } END { print sum + 0 }'); \
    total_c=$((c_count + cpp_count)); \
    printf '%s\n' 'C and C++ source and headers:' 'C files:' "$c_output" 'C++ files:' "$cpp_output"; \
    printf 'C code: %s lines\nC++ code: %s lines\nTotal C code (C + C++): %s + %s = %s lines\n' "$c_count" "$cpp_count" "$c_count" "$cpp_count" "$total_c"; \
    printf '%s\n' 'Python (backend and runtime):' "$python_output" 'Lua and Python (stereotype-packages):' "$stereotype_output"; \
    printf '%s lines of C code + %s lines of backend code + %s lines of stereotype-packages code = %s total lines\n' "$total_c" "$python_count" "$stereotype_count" "$((total_c + python_count + stereotype_count))"

# Repository-local OpenCode V2 development tooling.
swarm-setup:
    node -e 'import("./.opencode/swarm-mailbox/index.js").then(p => console.log(p.default.id))'

test-swarm:
    node --test tests/swarm_mailbox_test.mjs tests/swarm_activity_test.mjs

mermaid-check:
    node tools/mermaid-check/mermaid-check.mjs --check

mermaid-write:
    node tools/mermaid-check/mermaid-check.mjs --write

mermaid-sync:
    npm --prefix tools/mermaid-check ci

test-mermaid-check:
    node --test tests/mermaid_check_test.mjs
