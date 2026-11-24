#!/usr/bin/env python3
"""All study results in one PDF: DEST/report/results.pdf.

Usage: report_all.py DEST [--title T]

DEST is the folder written by import_usb.py (student/<id>/, expert/<id>/,
participants.csv, grades.csv). Reuses the existing tools: grade.py (expert
grades, if grades.csv is missing), parse_activity_log.py + paper_report.py
(students) and plot_session.py (one expert log at a time), then assembles a
single LaTeX document. Needs matplotlib and pdflatex (with pdfpages); run it
with `uv run --with matplotlib python tools/expert_study/report_all.py DEST`.
Reads the collected data only; writes under DEST/report/ (and DEST/grades.*
or DEST/analysis/ when missing).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG_TOOLS = ROOT / "tools" / "activity_log"


def by_id(paths):
    return sorted(paths, key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name))


def tex(text) -> str:
    """Escape plain text for LaTeX."""
    out = []
    for char in str(text):
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(char, char))
    return "".join(out)


def run(command, log) -> bool:
    done = subprocess.run([sys.executable, *map(str, command)], capture_output=True, text=True)
    log.append((" ".join(Path(str(c)).name if i == 0 else str(c) for i, c in enumerate(command)), done.returncode))
    if done.returncode:
        print(done.stdout[-2000:], done.stderr[-2000:], sep="\n", file=sys.stderr)
    return done.returncode == 0


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def participants_section(dest, students, experts) -> str:
    listed = {r.get("id", "").strip(): r.get("group", "").strip() for r in read_csv(dest / "participants.csv")}
    rows = [(p.name, "student", "yes" if (p / f"{p.name}.log").is_file() else "no", "", "") for p in students]
    for p in experts:
        colab = sorted((p / "colab").glob("*.json")) if (p / "colab").is_dir() else []
        projects = sorted(q.name for q in (p / "projects").iterdir()) if (p / "projects").is_dir() else []
        rows.append((p.name, "expert", "yes" if (p / f"{p.name}.log").is_file() else "no",
                     str(len(projects)), str(len(colab))))
    collected = {r[0] for r in rows}
    missing = sorted((pid for pid in listed if pid not in collected), key=lambda s: int(s) if s.isdigit() else 0)
    lines = [r"\section*{Participants}", r"\begin{tabular}{llccc}", r"\toprule",
             r"ID & Group & NNM log & NNM projects & Colab files \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if listed:
        lines.append(r"\par\smallskip " + (r"Listed in participants.csv but not collected: "
                                            + tex(", ".join(missing)) if missing else
                                            r"Every participant in participants.csv was collected."))
    return "\n".join(lines)


def grades_section(dest) -> str:
    rows = read_csv(dest / "grades.csv")
    if not rows:
        return r"\section*{Expert grades}No grades (no expert data)."
    lines = [r"\section*{Expert grades}", r"{\small", r"\begin{longtable}{lllcp{0.55\linewidth}}", r"\toprule",
             r"ID & Tool & Task & Result & Failed checks / notes \\", r"\midrule", r"\endhead"]
    for r in rows:
        note = r.get("detail") or ""
        info = r.get("info") or ""
        text = "; ".join(x for x in (note, info) if x)
        lines.append(" & ".join([tex(r.get("participant", "")), tex(r.get("tool", "")), tex(r.get("task", "")),
                                 "PASS" if r.get("passed") == "True" else r"\textbf{FAIL}", tex(text)]) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(lines)


def expert_section(pid, folder: Path) -> str:
    figures = [f for f in ("fig-timeline", "fig-time", "fig-problems", "fig-actions") if (folder / f"{f}.pdf").is_file()]
    if not figures:
        return rf"\section*{{Expert {tex(pid)}}}No NNModelling activity figures (log missing or empty)."
    body = [r"\clearpage", rf"\section*{{Expert {tex(pid)} --- NNModelling activity}}"]
    body += [rf"\begin{{center}}\includegraphics[width=0.85\linewidth,height=0.4\textheight,keepaspectratio]"
             rf"{{{(folder / f).as_posix()}.pdf}}\end{{center}}" for f in figures]
    return "\n".join(body)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", metavar="DEST")
    parser.add_argument("--title", default="NNModelling usability study --- results")
    args = parser.parse_args(argv)
    dest = Path(args.dest).resolve()
    report = dest / "report"
    report.mkdir(parents=True, exist_ok=True)
    students = by_id(p for p in (dest / "student").iterdir() if p.is_dir()) if (dest / "student").is_dir() else []
    experts = by_id(p for p in (dest / "expert").iterdir() if p.is_dir()) if (dest / "expert").is_dir() else []
    log = []

    if experts and not (dest / "grades.csv").is_file():
        run([HERE / "grade.py", "--json", dest / "grades.json", "--csv", dest / "grades.csv", *experts], log)
    student_pdf = None
    logs = [p / f"{p.name}.log" for p in students if (p / f"{p.name}.log").is_file()]
    if logs:
        analysis = dest / "analysis" / "students"
        if not (analysis / "tasks.csv").is_file():
            run([LOG_TOOLS / "parse_activity_log.py", *logs, "--out", analysis], log)
        if run([LOG_TOOLS / "paper_report.py", analysis, report / "students"], log):
            student_pdf = report / "students" / "report.pdf"
    for p in experts:
        if (p / f"{p.name}.log").is_file():
            run([HERE / "plot_session.py", p / f"{p.name}.log", "--out", report / "experts" / p.name], log)

    parts = [participants_section(dest, students, experts), grades_section(dest)]
    parts += [expert_section(p.name, report / "experts" / p.name) for p in experts]
    if student_pdf and student_pdf.is_file():
        parts.append(r"\clearpage\section*{Students}See the following pages (student activity report).")
        parts.append(rf"\includepdf[pages=-]{{{student_pdf.as_posix()}}}")
    elif students:
        parts.append(r"\clearpage\section*{Students}The student activity report could not be built (see the console).")
    failed = [name for name, code in log if code]
    if failed:
        parts.append(r"\section*{Steps that failed}" + tex("; ".join(failed)))
    document = "\n".join([
        r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=1.8cm]{geometry}",
        r"\usepackage{booktabs,longtable,graphicx,pdfpages}", r"\pagestyle{plain}", r"\begin{document}",
        rf"{{\LARGE\bfseries {tex(args.title)}}}\par\medskip",
        *parts, r"\end{document}", ""])
    (report / "results.tex").write_text(document, encoding="utf-8")
    for _ in range(2):
        done = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "results.tex"],
                              cwd=report, capture_output=True, text=True)
    if done.returncode or not (report / "results.pdf").is_file():
        print(done.stdout[-3000:], file=sys.stderr)
        print(f"pdflatex failed: see {report / 'results.log'}", file=sys.stderr)
        return 1
    print(f"Results: {report / 'results.pdf'}" + (f" (failed steps: {', '.join(failed)})" if failed else ""))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
