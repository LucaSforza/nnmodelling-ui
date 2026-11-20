#!/usr/bin/env python3
"""Split NNModelling activity logs per expert-study task project.

Usage: split_activity_log.py LOG... --out DIR

An app session is assigned to a task by the path of its successful
`project.open` events (task1-mlp or task2-multihead in the
path). A later task project.open inside the same session starts a new
segment; records before the first one go to `unassigned`. Records are copied
verbatim (fields and seq unchanged) to DIR/<task>/<participant>.jsonl, where
participant is the input file stem. Inputs are never modified.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TASKS = ("task1-mlp", "task2-multihead")
UNASSIGNED = "unassigned"
TASK_RE = re.compile("|".join(TASKS))


def split_log(path, out, warn=print):
    """Write one input log into out/<task>/<stem>.jsonl; return {task: record count}."""
    current, handles, counts = {}, {}, {}
    try:
        with open(path, encoding="utf-8", errors="replace") as source:
            for number, line in enumerate(source, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    session = record["session"]
                except (ValueError, KeyError, TypeError):
                    warn(f"{path}:{number}: skipped malformed line")
                    continue
                if record.get("event") == "project.open" and record.get("ok") is True:
                    found = TASK_RE.search(str((record.get("data") or {}).get("path", "")))
                    if found:
                        current[session] = found.group(0)
                task = current.get(session, UNASSIGNED)
                if task not in handles:
                    folder = Path(out) / task
                    folder.mkdir(parents=True, exist_ok=True)
                    handles[task] = open(folder / (Path(path).stem + ".jsonl"), "w", encoding="utf-8")
                handles[task].write(line + "\n")
                counts[task] = counts.get(task, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    for log in args.logs:
        counts = split_log(log, args.out, lambda m: print(m, file=sys.stderr))
        print(f"{log}: " + ", ".join(f"{t}={n}" for t, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
