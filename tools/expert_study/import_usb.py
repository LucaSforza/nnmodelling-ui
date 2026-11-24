#!/usr/bin/env python3
"""Import the data collected on the USB stick into the study folder.

Usage: import_usb.py DATA DEST [--participants CSV] [--grade]

DATA is the stick's nnm-study/data folder (data/<group>/<id>/ written by
collect.py); DEST receives DEST/<group>/<id>/. Identical files are skipped, a
differing existing file stops the import of that participant. Checks numeric
IDs, IDs present in both groups and, with --participants (CSV with columns
id,group), missing or unexpected IDs. --grade writes DEST/grades.csv/.json
(experts) and DEST/analysis/students/ (student activity logs).
Rules: docs/knowledge/testing/expert-study.md (Lab flow).
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from collect_participant import GROUPS, participant_id  # noqa: E402


def canonical(text):
    try:
        return participant_id(text)
    except ValueError:
        return None


def copy_participant(source: Path, target: Path) -> tuple[int, int, list[str]]:
    """(copied, identical, conflicts) for one participant folder; nothing is overwritten."""
    files = [p for p in sorted(source.rglob("*")) if p.is_file()]
    conflicts = [str(p.relative_to(source)) for p in files
                 if (target / p.relative_to(source)).exists()
                 and not filecmp.cmp(p, target / p.relative_to(source), shallow=False)]
    if conflicts:
        return 0, 0, conflicts
    copied = identical = 0
    for path in files:
        dest = target / path.relative_to(source)
        if dest.exists():
            identical += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied += 1
    return copied, identical, []


def read_participants(path: Path) -> dict[str, str]:
    expected = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            pid, group = canonical(row.get("id", "")), (row.get("group") or "").strip()
            if pid is None or group not in GROUPS:
                raise SystemExit(f"{path}:{line}: need a numeric id and group student|expert, got {row}")
            if pid in expected:
                raise SystemExit(f"{path}:{line}: ID {pid} listed twice")
            expected[pid] = group
    return expected


def import_data(data: Path, dest: Path, expected: dict[str, str] | None = None) -> list[str]:
    """Copy every data/<group>/<id>/ into dest; return the problems found (empty when all is well)."""
    problems, seen = [], {}
    for group in GROUPS:
        folder = data / group
        for source in sorted(folder.iterdir(), key=lambda p: (len(p.name), p.name)) if folder.is_dir() else []:
            if not source.is_dir():
                continue
            pid = source.name
            if canonical(pid) != pid:
                problems.append(f"{group}/{pid}: not a numeric participant ID, skipped")
                continue
            if pid in seen:
                problems.append(f"ID {pid} is both {seen[pid]} and {group}: fix it by hand, {group}/{pid} skipped")
                continue
            existing = dest / [g for g in GROUPS if g != group][0] / pid
            if existing.exists():
                problems.append(f"ID {pid} is already in {existing}: {group}/{pid} skipped")
                continue
            seen[pid] = group
            copied, identical, conflicts = copy_participant(source, dest / group / pid)
            if conflicts:
                problems.append(f"{group}/{pid}: differs from the files already in {dest / group / pid} "
                                f"({', '.join(conflicts[:3])}): skipped, nothing overwritten")
            else:
                print(f"{group}/{pid}: {copied} new file(s), {identical} already there")
    if expected is not None:
        for pid, group in sorted(expected.items(), key=lambda item: int(item[0])):
            if pid not in seen and not (dest / group / pid).exists():
                problems.append(f"ID {pid} ({group}) is in the participants list but has no data")
            elif pid in seen and seen[pid] != group:
                problems.append(f"ID {pid} is {group} in the participants list but was collected as {seen[pid]}")
        for pid in sorted(set(seen) - set(expected), key=int):
            problems.append(f"ID {pid} ({seen[pid]}) is not in the participants list")
    return problems


def grade(dest: Path) -> None:
    sys.stdout.flush()  # import messages before the graders' output
    experts = sorted((p for p in (dest / "expert").iterdir() if p.is_dir()), key=lambda p: int(p.name)) \
        if (dest / "expert").is_dir() else []
    if experts:
        subprocess.run([sys.executable, str(HERE / "grade.py"), "--json", str(dest / "grades.json"),
                        "--csv", str(dest / "grades.csv"), *map(str, experts)], check=False)
    logs = sorted((dest / "student").glob("*/*.log"))
    if logs:
        subprocess.run([sys.executable, str(ROOT / "tools/activity_log/parse_activity_log.py"), *map(str, logs),
                        "--out", str(dest / "analysis" / "students")], check=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", metavar="DATA", help="the stick's nnm-study/data folder")
    parser.add_argument("dest", metavar="DEST", help="study folder receiving <group>/<id>/")
    parser.add_argument("--participants", help="CSV with columns id,group")
    parser.add_argument("--grade", action="store_true", help="grade experts and analyse student logs")
    args = parser.parse_args(argv)
    data, dest = Path(args.data), Path(args.dest)
    if not data.is_dir():
        parser.error(f"{data} is not a folder")
    expected = read_participants(Path(args.participants)) if args.participants else None
    problems = import_data(data, dest, expected)
    if args.grade:
        grade(dest)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("Import complete." if not problems else f"Import done with {len(problems)} problem(s), see above.")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
