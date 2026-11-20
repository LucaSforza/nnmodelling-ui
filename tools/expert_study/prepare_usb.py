#!/usr/bin/env python3
"""Prepare the expert-study USB stick (docs/knowledge/testing/expert-study.md).

Usage: prepare_usb.py [--force] OUT

Writes OUT/master (start projects, participant notebooks, study_checks.py,
mnist_small.json, task sheet), OUT/facilitator (solutions) and
OUT/new_session.py, which copies master/ into a fresh participant folder.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT / "study" / "experts"
TASKS = ("task1-mlp", "task2-multihead")
DATA = ROOT / "examples" / "test" / "datasets" / "mnist" / "data.json"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")

NEW_SESSION = '''#!/usr/bin/env python3
"""Create a fresh participant folder from master/ (standard library only).

Usage: new_session.py [--dest DIR] PARTICIPANT_ID
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

TASKS = ("task1-mlp", "task2-multihead")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="~/nnm-expert-study", help="folder holding one directory per participant")
    parser.add_argument("participant", metavar="PARTICIPANT_ID")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.participant):
        sys.exit("PARTICIPANT_ID may contain letters, digits, '_', '-' and '.' only")
    master = Path(__file__).resolve().parent / "master"
    if not master.is_dir():
        sys.exit(f"missing {master}")
    target = Path(args.dest).expanduser() / args.participant
    if target.exists():
        sys.exit(f"refusing to overwrite existing participant folder: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(master, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Participant folder: {target}")
    for task in TASKS:
        print(f"NNModelling project: {target / 'projects' / task}")
    print(f"Notebook folder: {target / 'notebooks'}")


if __name__ == "__main__":
    main()
'''


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"missing study material: {source}")
    shutil.copytree(source, target, ignore=IGNORE)


def build_task_sheet(master: Path) -> None:
    tex = STUDY / "tasks" / "expert-tasks.tex"
    target = master / "tasks"
    target.mkdir(parents=True, exist_ok=True)
    if not tex.is_file():
        print(f"warning: {tex} not found, no task sheet", file=sys.stderr)
        return
    if shutil.which("pdflatex"):
        with tempfile.TemporaryDirectory(prefix="nnm-tasks-") as temporary:
            work = Path(temporary)
            shutil.copy(tex, work / tex.name)
            for sibling in tex.parent.iterdir():
                if sibling.is_dir():  # e.g. img/ with the screenshots
                    shutil.copytree(sibling, work / sibling.name)
                elif sibling != tex and sibling.suffix != ".pdf":
                    shutil.copy(sibling, work / sibling.name)
            ok = True
            for _ in range(2):  # second pass resolves references
                done = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                    cwd=work, capture_output=True, text=True, timeout=180,
                )
                ok = ok and done.returncode == 0
            pdf = work / "expert-tasks.pdf"
            if ok and pdf.is_file():
                shutil.copy(pdf, target / "expert-tasks.pdf")
                return
        print("warning: pdflatex failed, copying the .tex instead", file=sys.stderr)
    else:
        print("warning: pdflatex not found, copying the .tex instead", file=sys.stderr)
    shutil.copy(tex, target / tex.name)


def prepare(out: Path, force: bool = False) -> None:
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()) and not force:
        raise SystemExit(f"{out} is not empty (use --force to rebuild master/ and facilitator/)")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("master", "facilitator"):
        shutil.rmtree(out / name, ignore_errors=True)
    master, facilitator = out / "master", out / "facilitator"
    for task in TASKS:
        copy_tree(STUDY / "projects" / task / "start", master / "projects" / task)
        copy_tree(STUDY / "projects" / task / "solution", facilitator / "projects" / task)
    notebooks = master / "notebooks"
    notebooks.mkdir(parents=True)
    participant_notebooks = sorted((STUDY / "notebooks").glob("*.ipynb"))
    if len(participant_notebooks) != len(TASKS):
        raise SystemExit(f"expected {len(TASKS)} notebooks in {STUDY / 'notebooks'}")
    for notebook in participant_notebooks:
        shutil.copy(notebook, notebooks / notebook.name)
    shutil.copy(HERE / "study_checks.py", notebooks / "study_checks.py")
    if not DATA.is_file():
        raise SystemExit(f"missing dataset file {DATA}")
    shutil.copy(DATA, notebooks / "mnist_small.json")
    copy_tree(STUDY / "notebooks" / "solutions", facilitator / "notebooks" / "solutions")
    build_task_sheet(master)
    script = out / "new_session.py"
    script.write_text(NEW_SESSION, encoding="utf-8")
    script.chmod(0o755)
    print(f"Prepared {out}: master/ (copy per participant), facilitator/ (solutions), new_session.py")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="allow a non-empty OUT (rebuilds master/ and facilitator/)")
    parser.add_argument("out", metavar="OUT")
    args = parser.parse_args(argv)
    prepare(Path(args.out).expanduser(), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
