#!/usr/bin/env bash
# Copy the NNModelling activity log to DEST/<id>.log, where ID is the numeric
# participant ID assigned by the research team (digits only; 007 -> 7.log).
# Usage: collect.sh [--reset] DEST [ID]
# Without ID it asks for it. --reset then moves the original log away so the
# next participant starts from an empty one.
set -eu
reset=0
args=()
for arg in "$@"; do
  case "$arg" in
    --reset) reset=1 ;;
    *) args+=("$arg") ;;
  esac
done
usage="usage: $0 [--reset] DEST [ID]"
[ "${#args[@]}" -eq 1 ] || [ "${#args[@]}" -eq 2 ] || { echo "$usage" >&2; exit 2; }
dest=${args[0]}
src=${NNMODELLING_ACTIVITY_LOG:-$HOME/nnmodelling.log}
[ -f "$src" ] || { echo "log not found: $src" >&2; exit 1; }

if [ "${#args[@]}" -eq 2 ]; then
  id=${args[1]}
else
  read -r -p "ID partecipante: " id
fi
id="${id#"${id%%[![:space:]]*}"}"  # trim outer spaces
id="${id%"${id##*[![:space:]]}"}"
[[ "$id" =~ ^[0-9]{1,9}$ ]] || { echo "l'ID deve contenere solo cifre: '$id'" >&2; exit 2; }
id=$((10#$id))
[ "$id" -gt 0 ] || { echo "l'ID non puo essere 0" >&2; exit 2; }
name="$id.log"

mkdir -p "$dest"
[ ! -e "$dest/$name" ] || { echo "esiste gia: $dest/$name" >&2; exit 1; }
cp "$src" "$dest/$name"
if [ "$reset" = 1 ]; then
  mv "$src" "$src.collected-$(date +%Y%m%d-%H%M%S)"
fi
echo "$name"
