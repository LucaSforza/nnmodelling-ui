#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'AppImage build requires Linux x86_64.' >&2
    exit 1
fi

if [[ ${NN_APPIMAGE_CONTAINER:-0} != 1 ]]; then
    . /etc/os-release
    if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 22.04 ]]; then
        if ! command -v podman >/dev/null; then
            echo 'Install rootless Podman to build the Ubuntu 22.04 AppImage on this host.' >&2
            exit 1
        fi
        mkdir -p build dist
        echo 'Building AppImage in an Ubuntu 22.04 container...'
        exec podman run --rm --userns=keep-id:uid=0,gid=0 --security-opt label=disable \
            -v "$PWD:/workspace:ro" -v "$PWD/build:/workspace/build:rw" \
            -v "$PWD/dist:/workspace/dist:rw" -w /workspace \
            -e NN_APPIMAGE_CONTAINER=1 -e DEBIAN_FRONTEND=noninteractive \
            docker.io/library/ubuntu:22.04 bash -ec '
                apt-get update -qq
                apt-get install --no-install-recommends -y build-essential cmake git curl file python3 \
                    ca-certificates qt6-base-dev qt6-base-dev-tools libgl1-mesa-dev
                exec bash tools/build-appimage.sh
            '
    fi
fi

build_dir=build/appimage
if [[ ${NN_APPIMAGE_CONTAINER:-0} == 1 ]]; then
    build_dir=build/appimage-ubuntu
fi
for tool in cmake git tar file qmake6; do
    if ! command -v "$tool" >/dev/null; then
        echo "AppImage build requires '$tool' on PATH." >&2
        exit 1
    fi
done

tools_dir="$PWD/build/appimage-tools"
for tool in linuxdeploy linuxdeploy-plugin-qt; do
    if ! command -v "$tool" >/dev/null; then
        if [[ ! -x "$tools_dir/$tool" ]]; then
            if ! command -v curl >/dev/null; then
                echo "Install curl to download $tool, or provide $tool on PATH." >&2
                exit 1
            fi
            mkdir -p "$tools_dir"
            echo "Downloading $tool..."
            url="https://github.com/linuxdeploy/$tool/releases/download/continuous/$tool-x86_64.AppImage"
            if ! curl -fL --retry 3 -o "$tools_dir/$tool.tmp" "$url"; then
                rm -f "$tools_dir/$tool.tmp"
                echo "Failed to download $tool from $url." >&2
                exit 1
            fi
            chmod +x "$tools_dir/$tool.tmp"
            mv -f "$tools_dir/$tool.tmp" "$tools_dir/$tool"
        fi
        export PATH="$tools_dir:$PATH"
    fi
done

cmake -S . -B "$build_dir" -DNN_BUILD_GUI=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel 4
rm -rf "$build_dir/AppDir"
appdir="$PWD/$build_dir/AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/nnmodelling" dist
cp "$build_dir/nnmodelctl" "$appdir/usr/bin/"
# Only tracked assets belong in releases; local datasets and caches must not leak.
git ls-files -z -- stereotype-packages/core examples/models/mnist-mlp examples/models/mnist-vae examples/models/tiny-decoder-llm |
    tar --null -T - -cf - | tar -x -C "$appdir/usr/share/nnmodelling"
export QMAKE="$(command -v qmake6)" APPIMAGE_EXTRACT_AND_RUN=1
# Bundled strip can be older than the host libraries' ELF format.
export NO_STRIP=1
export EXTRA_PLATFORM_PLUGINS=libqoffscreen.so
export OUTPUT="$PWD/dist/NNModelling-x86_64.AppImage"
linuxdeploy --appdir "$appdir" --executable "$build_dir/nnmodelling-ui" --desktop-file packaging/nnmodelling.desktop \
    --icon-file packaging/nnmodelling.svg --plugin qt --output appimage

smoke_dir=$(mktemp -d)
trap 'rm -rf "$smoke_dir"' EXIT
if ! QT_QPA_PLATFORM=offscreen "$OUTPUT" --capture "$smoke_dir/smoke.png" --quit-after-capture ||
        [[ ! -s "$smoke_dir/smoke.png" ]]; then
    echo 'AppImage failed its startup test. Build on Ubuntu 22.04 for the supported release.' >&2
    exit 1
fi
