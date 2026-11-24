#!/usr/bin/env python3
"""Prepare the expert-study USB stick (docs/knowledge/testing/expert-study.md).

Usage: prepare_usb.py [--force] OUT

Writes OUT/master (start projects, start notebooks and README.txt with the Colab links,
task sheet), OUT/facilitator (solutions), OUT/new_session.py ID, which copies
master/ into a fresh participant folder <id>/, and OUT/collect.sh (activity log
as <id>.log). ID is the numeric participant ID. The PyTorch tasks run in Google Colab;
their submission files go into the participant's notebooks/ folder.
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
sys.path.insert(0, str(HERE))
from build_notebooks import DEFAULT_REF, REPO, TASKS as NOTEBOOKS  # noqa: E402

COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{DEFAULT_REF}/study/experts/notebooks/"
NOTEBOOKS_README = """PyTorch tasks run in Google Colab (browser, the expert's Google account):

""" + "".join(f"  {task}: {COLAB}{spec['file']}\n" for task, spec in NOTEBOOKS.items()) + """
The .ipynb files here are the same start notebooks, as a fallback: in Colab use
File > Upload notebook (the first cell still needs internet to reach GitHub).

Submitting a task downloads nnm-study-<task>-<participant>-<time>.json into the
expert's Downloads folder. Move that file into this folder, then run grade.py.
"""
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")

NEW_SESSION = '''#!/usr/bin/env python3
"""Create a fresh participant folder from master/ (standard library only).

Usage: new_session.py [--dest DIR] ID   (ID = numeric participant ID, digits only)
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
    parser.add_argument("participant", metavar="ID")
    args = parser.parse_args()
    text = args.participant.strip()
    if not re.fullmatch(r"[0-9]{1,9}", text) or int(text) == 0:
        sys.exit("ID must be the numeric participant ID (digits only, not 0)")
    pid = str(int(text))  # canonical: 007 -> 7
    here = Path(__file__).resolve().parent
    master = here / "master"
    if not master.is_dir():
        sys.exit(f"missing {master}")
    target = Path(args.dest).expanduser() / pid
    if target.exists():
        sys.exit(f"refusing to overwrite existing participant folder: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(master, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Participant ID: {pid} (write it on the task sheet; the expert types it in Colab)")
    print(f"Participant folder: {target}")
    for task in TASKS:
        print(f"NNModelling project: {target / 'projects' / task}")
    print(f"Colab submissions go into: {target / 'notebooks'} (links in README.txt there)")
    print(f"At the end: bash {here / 'collect.sh'} --reset {target} {pid}")


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
    (notebooks / "README.txt").write_text(NOTEBOOKS_README, encoding="utf-8")
    for spec in NOTEBOOKS.values():  # start notebooks only; solutions go to facilitator/
        shutil.copy(STUDY / "notebooks" / spec["file"], notebooks / spec["file"])
    copy_tree(STUDY / "notebooks" / "solutions", facilitator / "notebooks" / "solutions")
    build_task_sheet(master)
    shutil.copy(ROOT / "tools" / "activity_log" / "collect.sh", out / "collect.sh")
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
