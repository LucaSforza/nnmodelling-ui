#!/usr/bin/env python3
"""Collect one participant's study data onto the USB stick (standard library only).

Usage: python3 collect.py ID student|expert [--home DIR] [--keep]

Run on the lab machine when the participant has finished. Writes, next to this
script, data/<group>/<id>/ with:
  <id>.log                 NNModelling activity log (~/nnmodelling.log)
  projects/                experts: ~/nnm-expert-study/<id>/projects/
  colab/                   experts: nnm-study-<id>-*.json Colab submissions
Never overwrites collected data. Unless --keep, the log and the collected
submissions are moved aside so the next participant on this machine starts
clean. Rules: docs/knowledge/testing/expert-study.md (Lab flow).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

GROUPS = ("student", "expert")
TASKS = {"T1": "task1-mlp", "T2": "task2-multihead"}
SUBMISSION = re.compile(r"nnm-study-(\w+)-(T[12])-.*\.json$")


def participant_id(text: str) -> str:
    """Canonical numeric participant ID ('007' -> '7'), as study_checks.participant_id."""
    text = str(text).strip()
    if not text.isascii() or not text.isdigit() or int(text) == 0:
        raise ValueError(f"the ID must be the participant number, digits only (got {text!r})")
    return str(int(text))


def download_dirs(home: Path) -> list[Path]:
    """Browser download folders: Italian and English defaults plus the XDG setting."""
    found = [home / "Scaricati", home / "Downloads"]
    try:
        for line in (home / ".config" / "user-dirs.dirs").read_text(encoding="utf-8").splitlines():
            if line.startswith("XDG_DOWNLOAD_DIR="):
                value = line.split("=", 1)[1].strip().strip('"').replace("$HOME", str(home))
                found.append(Path(value))
    except OSError:
        pass
    unique = []
    for folder in found:
        if folder.is_dir() and folder.resolve() not in [u.resolve() for u in unique]:
            unique.append(folder)
    return unique


def submissions(home: Path, pid: str):
    """(mine, others): Colab submission files on this machine for this ID and for other IDs."""
    folders = download_dirs(home) + [home / "nnm-expert-study" / pid / "notebooks"]
    mine, others = [], []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("nnm-study-*.json")):
            match = SUBMISSION.match(path.name)
            if not match:
                continue
            (mine if match.group(1) == pid else others).append(path)
    return mine, others


def submission_task(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("task")
    except (OSError, ValueError, AttributeError):
        match = SUBMISSION.match(path.name)
        return match.group(2) if match else None


def collect(pid: str, group: str, home: Path, data: Path, keep: bool = False) -> int:
    """Copy one participant's files to data/<group>/<pid>/; return 0 when complete, 1 when something is missing."""
    target = data / group / pid
    other_group = data / [g for g in GROUPS if g != group][0] / pid
    if target.exists():
        sys.exit(f"ERROR: {target} already exists: participant {pid} was already collected (nothing changed)")
    if other_group.exists():
        sys.exit(f"ERROR: ID {pid} was already collected as {other_group.parent.name} (nothing changed)")
    log = Path(os.environ.get("NNMODELLING_ACTIVITY_LOG") or home / "nnmodelling.log")
    session = home / "nnm-expert-study" / pid
    mine, others = submissions(home, pid) if group == "expert" else ([], [])
    missing = []

    target.mkdir(parents=True)
    print(f"Participant {pid} ({group}) -> {target}")
    if log.is_file() and log.stat().st_size > 0:
        shutil.copy2(log, target / f"{pid}.log")
        lines = sum(1 for _ in log.open(encoding="utf-8", errors="replace"))
        print(f"  ok  NNModelling log: {lines} lines")
    else:
        missing.append(f"NNModelling log ({log} missing or empty)")

    if group == "expert":
        projects = session / "projects"
        if projects.is_dir():
            shutil.copytree(projects, target / "projects", ignore=shutil.ignore_patterns("__pycache__"))
            for task, name in TASKS.items():
                if (projects / name / "model.json").is_file():
                    print(f"  ok  NNModelling project {task} ({name})")
                else:
                    missing.append(f"NNModelling project {task} ({projects / name})")
        else:
            missing.append(f"NNModelling projects ({projects}); was new_session.py {pid} run on this machine?")
        if mine:
            (target / "colab").mkdir()
            for path in mine:
                shutil.copy2(path, target / "colab" / path.name)
        for task in TASKS:
            count = sum(1 for path in mine if submission_task(path) == task)
            if count:
                print(f"  ok  Colab submission {task}" + (f" ({count} files: only the first counts)" if count > 1 else ""))
            else:
                missing.append(f"Colab submission {task} (nnm-study-{pid}-{task}-*.json not found in "
                               + ", ".join(str(d) for d in download_dirs(home)) + ")")
        for path in others:
            print(f"  note  {path} belongs to another ID: not collected")

    for item in missing:
        print(f"  MISSING  {item}")
    if not any(target.iterdir()):
        target.rmdir()
        print(f"ERROR: nothing found for participant {pid} on this machine: nothing collected, nothing moved")
        return 2
    if not keep:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if log.is_file():
            log.rename(log.with_name(f"{log.name}.collected-{stamp}"))
        for path in mine:
            if path.parent.name == "notebooks":
                continue  # the session folder keeps its copy
            aside = path.parent / "nnm-study-collected"
            aside.mkdir(exist_ok=True)
            path.rename(aside / path.name)
        print("  Log and collected submissions moved aside: the next participant starts clean.")
    print("DONE" if not missing else f"DONE WITH {len(missing)} MISSING ITEM(S): check above before the next participant")
    return 0 if not missing else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("participant", metavar="ID", help="numeric participant ID (digits only)")
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--home", default=str(Path.home()), help="home folder of the lab session (default: yours)")
    parser.add_argument("--data", help="destination data folder (default: data/ next to this script)")
    parser.add_argument("--keep", action="store_true", help="do not move the log and submissions aside")
    args = parser.parse_args(argv)
    try:
        pid = participant_id(args.participant)
    except ValueError as exc:
        parser.error(str(exc))
    data = Path(args.data) if args.data else Path(__file__).resolve().parent / "data"
    return collect(pid, args.group, Path(args.home).expanduser(), data, args.keep)


if __name__ == "__main__":
    sys.exit(main())
