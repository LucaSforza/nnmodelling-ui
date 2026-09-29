set shell := ["bash", "-ec"]

default: build

# Core configuration does not enable C++ or discover Qt.
core:
    cmake -S . -B build/core -DNN_BUILD_GUI=OFF -DBUILD_TESTING=ON
    cmake --build build/core --parallel 4

test: core
    ctest --test-dir build/core --output-on-failure -L core

build:
    cmake -S . -B build/qt -DNN_BUILD_GUI=ON -DBUILD_TESTING=ON
    cmake --build build/qt --parallel 4

run *args: build
    ./build/qt/nnmodelling-ui {{args}}

test-ui: build
    ctest --test-dir build/qt --output-on-failure -L gui
