#!/usr/bin/env bash
# Start/stop the real Qt client on Qt's VNC platform plugin, loopback-only.
# Runtime files live in the ignored .computer-use/vnc/ folder.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
dir="$root/.computer-use/vnc"
port="${NN_VNC_PORT:-5998}"
size="${NN_VNC_SIZE:-1600x1000}"
mkdir -p "$dir"
case "${1:-}" in
start)
    shift
    if [[ -f "$dir/qt.pid" ]] && kill -0 "$(cat "$dir/qt.pid")" 2>/dev/null; then
        echo "session already running (pid $(cat "$dir/qt.pid")); run stop first" >&2; exit 1
    fi
    [[ -x "$root/build/qt/nnmodelling-ui" ]] || { echo "run just build first" >&2; exit 1; }
    cc -shared -fPIC "$root/.agents/skills/nnmodelling-browser/scripts/loopback.c" -ldl -o "$dir/loopback.so"
    env NN_SOURCE_DIR="$root" LD_PRELOAD="$dir/loopback.so" \
        QT_QPA_PLATFORM="vnc:size=$size:port=$port" QT_STYLE_OVERRIDE=Fusion \
        nohup "$root/build/qt/nnmodelling-ui" "$@" > "$dir/qt.log" 2>&1 < /dev/null &
    echo $! > "$dir/qt.pid"
    for _ in $(seq 50); do
        if python3 - "$port" <<'PY' 2>/dev/null; then echo "ready: VNC ::1:$port pid $(cat "$dir/qt.pid")"; exit 0; fi
import socket, sys
socket.create_connection(("::1", int(sys.argv[1])), timeout=0.2).close()
PY
        sleep 0.2
    done
    echo "VNC not ready; see $dir/qt.log" >&2; exit 1 ;;
stop)
    if [[ -f "$dir/qt.pid" ]]; then
        pid="$(cat "$dir/qt.pid")"
        if [[ "$(readlink "/proc/$pid/exe" 2>/dev/null)" == "$root/build/qt/nnmodelling-ui"* ]]; then
            kill "$pid"; wait "$pid" 2>/dev/null || true
        fi
        rm -f "$dir/qt.pid"
    fi
    echo stopped ;;
status)
    if [[ -f "$dir/qt.pid" ]] && kill -0 "$(cat "$dir/qt.pid")" 2>/dev/null; then echo "running pid $(cat "$dir/qt.pid")"; else echo "not running"; fi ;;
*) echo "usage: session.sh start [APP ARGS...] | stop | status" >&2; exit 2 ;;
esac
