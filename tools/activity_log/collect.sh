#!/usr/bin/env bash
# Copy the NNModelling activity log to DEST/P-<random hex>.jsonl.
# Usage: collect.sh [--reset] DEST
set -eu
reset=0
dest=
for arg in "$@"; do
  case "$arg" in
    --reset) reset=1 ;;
    *) dest=$arg ;;
  esac
done
[ -n "$dest" ] || { echo "usage: $0 [--reset] DEST" >&2; exit 2; }
src=${NNMODELLING_ACTIVITY_LOG:-$HOME/nnmodelling.log}
[ -f "$src" ] || { echo "log not found: $src" >&2; exit 1; }
mkdir -p "$dest"
name="P-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n').jsonl"
cp "$src" "$dest/$name"
if [ "$reset" = 1 ]; then
  mv "$src" "$src.collected-$(date +%Y%m%d-%H%M%S)"
fi
echo "$name"
