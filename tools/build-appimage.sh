#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'AppImage build requires Linux x86_64.' >&2
    exit 1
fi
command -v linuxdeploy >/dev/null
command -v linuxdeploy-plugin-qt >/dev/null
command -v qmake6 >/dev/null

cmake -S . -B build/appimage -DNN_BUILD_GUI=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/appimage --parallel 4
rm -rf build/appimage/AppDir
appdir="$PWD/build/appimage/AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/nnmodelling" dist
cp build/appimage/nnmodelctl "$appdir/usr/bin/"
# Only tracked assets belong in releases; local datasets and caches must not leak.
git ls-files -z -- stereotype-packages/core examples/models/mnist-mlp examples/models/mnist-vae examples/models/tiny-decoder-llm |
    tar --null -T - -cf - | tar -x -C "$appdir/usr/share/nnmodelling"
export QMAKE="$(command -v qmake6)" APPIMAGE_EXTRACT_AND_RUN=1
export EXTRA_PLATFORM_PLUGINS=libqoffscreen.so
export OUTPUT="$PWD/dist/NNModelling-x86_64.AppImage"
linuxdeploy --appdir "$appdir" --executable build/appimage/nnmodelling-ui --desktop-file packaging/nnmodelling.desktop \
    --icon-file packaging/nnmodelling.svg --plugin qt --output appimage
