#!/usr/bin/env python3
"""Split activity logs per task and parse each task with its own task file.

Usage: analyse_activity.py LOG... --out OUT [--idle-seconds N]

Writes OUT/split/<task>/<participant>.jsonl (split_activity_log.py) and the
parser outputs (events/sessions/tasks.csv, summary.md) into OUT/<task>/ using
tools/activity_log/tasks_expert_t{1,2,3}.json. Prints a completion table.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import split_activity_log as split  # noqa: E402

PARSER = HERE.parent / "activity_log" / "parse_activity_log.py"
TASK_FILES = {task: PARSER.with_name(f"tasks_expert_t{n}.json") for n, task in enumerate(split.TASKS, 1)}


def analyse(logs, out, idle_seconds=30):
    out = Path(out)
    for log in logs:
        split.split_log(log, out / "split", lambda m: print(m, file=sys.stderr))
    table = []
    for task, tasks_file in TASK_FILES.items():
        files = sorted((out / "split" / task).glob("*.jsonl"))
        if not files:
            continue
        subprocess.run(
            [sys.executable, str(PARSER), *map(str, files), "--out", str(out / task),
             "--tasks", str(tasks_file), "--idle-seconds", str(idle_seconds)],
            check=True, capture_output=True, text=True)
        with open(out / task / "tasks.csv", newline="", encoding="utf-8") as handle:
            table += [(task, r["participant"], r["task"], r["completed"]) for r in csv.DictReader(handle)]
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--idle-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    for task, participant, sub_goal, completed in analyse(args.logs, args.out, args.idle_seconds):
        print(f"{task:16} {participant:16} {sub_goal:5} {'done' if completed == 'True' else 'open'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
