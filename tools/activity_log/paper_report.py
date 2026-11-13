#!/usr/bin/env python3
"""Behavioural analysis of parsed activity logs -> figures, LaTeX tables, PDF.

Usage: paper_report.py RESULTS_DIR OUT_DIR [--title T] [--synthetic] [--no-pdf]

RESULTS_DIR is the output of parse_activity_log.py (events/sessions/tasks.csv).
Needs matplotlib and a LaTeX install (pdflatex). Writes OUT_DIR/figures/*.pdf,
OUT_DIR/tables/*.tex (to \\input in the paper), measures.csv and report.pdf.
"""
import argparse
import csv
import json
import re
import statistics as st
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

TASKS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8a", "T8b", "T8c", "T9", "T10"]
CLUSTERS = [("Input setup", ["T1", "T2"]), ("Data reshaping", ["T3", "T4"]),
            ("Hidden layer", ["T5", "T6", "T7"]),
            ("Classification", ["T8a", "T8b", "T8c"]),
            ("Loss and output", ["T9", "T10"])]
CLUSTER_OF = {t: c for c, ts in CLUSTERS for t in ts}
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKS = ["o", "s", "^", "D", "v", "P"]
AWAY_MIN = 20.0      # app.blur longer than this counts as a consultation (s)


def rd(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def dt(s):
    return datetime.fromisoformat(s)


def tex(s):
    return (str(s).replace("\\", " ").replace("_", r"\_").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#"))


def short(pid):
    m = re.match(r"P-(sim)?(\w+?)(?:-(\w+))?$", pid)
    return pid if not m else f"P{m.group(2)}" if m.group(2).isdigit() else pid[:8]


# ------------------------------------------------------------- measures
def load(results):
    ev, tk, se = (rd(results / f) for f in ("events.csv", "tasks.csv", "sessions.csv"))
    by = defaultdict(list)
    for r in ev:
        r["_dt"] = dt(r["ts"])
        r["_d"] = json.loads(r["data"]) if r["data"] else {}
        by[r["participant"]].append(r)
    for rows in by.values():            # session order, then seq
        rows.sort(key=lambda r: (int(r["session"]), int(r["seq"])))
    return by, tk, se


def away_intervals(rows):
    out, start = [], None
    for r in rows:
        if r["event"] == "app.blur":
            start = r["_dt"]
        elif r["event"] == "app.focus" and start is not None:
            out.append((start, r["_dt"]))
            start = None
    return out


def overlap(a0, a1, ivals):
    return sum(max(0.0, (min(a1, e) - max(a0, s)).total_seconds()) for s, e in ivals)


def match(query, package):
    name = package.split(".")[-1].replace("-", " ")
    q = query.lower().strip()
    return bool(q) and (q in name or name.startswith(q) or name in q)


def per_participant(by, tk):
    comp = {(r["participant"], r["task"]): r for r in tk}
    M = {}
    for pid, rows in by.items():
        away = away_intervals(rows)
        t0 = rows[0]["_dt"]
        m = {"consult_s": sum((e - s).total_seconds() for s, e in away
                              if (e - s).total_seconds() >= AWAY_MIN),
             "consults": sum(1 for s, e in away if (e - s).total_seconds() >= AWAY_MIN)}
        # completion times, wall and net of consultations
        wall, net, prev_w, prev_n, prev_dt = {}, {}, 0.0, 0.0, t0
        for t in TASKS:
            c = comp.get((pid, t))
            if c and c["completed"] == "True":
                end = dt(c["completed_at"])
                wall[t] = (end - t0).total_seconds()
                net[t] = wall[t] - overlap(t0, end, [x for x in away
                                                    if (x[1] - x[0]).total_seconds() >= AWAY_MIN])
        m["wall"], m["net"] = wall, net
        m["total_s"] = max(wall.values()) if wall else float("nan")
        m["net_s"] = max(net.values()) if net else float("nan")
        m["completed"] = len(wall)
        m["sessions"] = len({r["session"] for r in rows})
        m["actions"] = sum(int(c["actions"]) for k, c in comp.items() if k[0] == pid)
        m["undo"] = sum(1 for r in rows if r["event"] == "edit.undo")
        # palette: queries before each add
        queries, adds, cur = [], [], []
        for r in rows:
            if r["event"] == "palette.search" and r["_d"].get("query"):
                cur.append(r["_d"]["query"])
            elif r["event"] == "palette.add":
                pkg = r["_d"].get("package", "")
                hit = [match(q, pkg) for q in cur]
                adds.append({"package": pkg, "n_queries": len(cur),
                             "first_hit": bool(hit and hit[0]),
                             "queries": list(cur), "via": r["_d"].get("via")})
                cur = []
        m["adds"] = adds
        # connections
        starts = sum(1 for r in rows if r["event"] == "connect.start")
        ok = sum(1 for r in rows if r["event"] == "edge.connect" and r["ok"] == "True")
        bad = [r for r in rows if r["event"] == "edge.connect" and r["ok"] == "False"]
        canc = Counter(r["_d"].get("reason") for r in rows if r["event"] == "connect.cancel")
        m.update(conn_starts=starts, conn_ok=ok, conn_fail=len(bad),
                 conn_cancel=sum(canc.values()), cancel_reasons=canc,
                 fail_reasons=Counter(r["error"] for r in bad))
        # parameters
        seq = defaultdict(list)
        for r in rows:
            if r["event"] == "param.set" and r["ok"] == "True":
                seq[(r["_d"].get("node"), r["_d"].get("key"))].append(r["_d"].get("new"))
        m["param_sets"] = sum(len(v) for v in seq.values())
        m["param_fields"] = len(seq)
        m["param_corrections"] = m["param_sets"] - len(seq)
        m["problems_opened"] = sum(1 for r in rows if r["event"] == "problems.toggle"
                                   and r["_d"].get("expanded"))
        m["problem_clicks"] = sum(1 for r in rows if r["event"] == "problem.click")
        m["drag_starts"] = sum(1 for r in rows if r["event"] == "drag.start")
        m["removed"] = sum(1 for r in rows if r["event"] == "node.remove")
        m["ooo"] = {t for t in TASKS if comp.get((pid, t), {}).get("out_of_order") == "True"}
        m["out_of_order"] = len(m["ooo"])
        # friction per task: failures + cancels + undo + extra queries
        fr = {}
        for t in TASKS:
            c = comp.get((pid, t))
            fr[t] = 0 if not c else (int(c["failed_ops"]) + int(c["undos"])
                                     + int(c["connect_cancels"])
                                     + max(0, len([q for q in c["palette_queries"].split("|") if q]) - 1))
        seen, bounds = set(), [(t, dt(comp[(pid, t)]["completed_at"])) for t in TASKS
                               if (pid, t) in comp and comp[(pid, t)]["completed"] == "True"]
        for r in rows:       # repeated edits of one field count in the task being worked on
            if r["event"] == "param.set" and r["ok"] == "True":
                key = (r["_d"].get("node"), r["_d"].get("key"))
                if key in seen:
                    task = next((t for t, e in bounds if r["_dt"] <= e), None)
                    if task:
                        fr[task] += 1
                seen.add(key)
        m["friction"] = fr
        M[pid] = m
    return M


def write_measures(M, path):
    keys = ["total_s", "net_s", "consult_s", "consults", "completed", "sessions", "actions",
            "undo", "conn_starts", "conn_ok", "conn_fail", "conn_cancel", "param_sets",
            "param_corrections", "problems_opened", "problem_clicks", "removed", "out_of_order"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant"] + keys)
        for pid, m in M.items():
            w.writerow([pid] + [round(m[k], 1) if isinstance(m[k], float) else m[k] for k in keys])


# ------------------------------------------------------------- tables
def fmt_t(s):
    return "--" if s != s else f"{int(s // 60)}:{int(s % 60):02d}"


def table_overview(M, path):
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Participant & Tasks & Total & Net & Consult. & Sess. & Actions & Undo & Conn. fail\\",
         r"& done & mm:ss & mm:ss & n & n & n & n & n\\", r"\midrule"]
    for pid, m in M.items():
        L.append(f"{tex(short(pid))} & {m['completed']}/12 & {fmt_t(m['total_s'])} & "
                 f"{fmt_t(m['net_s'])} & {m['consults']} & {m['sessions']} & {m['actions']} & "
                 f"{m['undo']} & {m['conn_fail']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def table_clusters(M, path):
    L = [r"\begin{tabular}{l" + "r" * len(M) + r"rr}", r"\toprule",
         "Cluster & " + " & ".join(tex(short(p)) for p in M) + r" & Median & IQR\\", r"\midrule"]
    for cname, ts in CLUSTERS:
        vals = []
        for m in M.values():
            first, last = ts[0], ts[-1]
            idx = TASKS.index(first)
            prev = m["net"].get(TASKS[idx - 1], 0.0) if idx else 0.0
            vals.append(m["net"][last] - prev if last in m["net"] else float("nan"))
        ok = sorted(v for v in vals if v == v)
        med = st.median(ok) if ok else float("nan")
        iqr = (ok[(3 * (len(ok) - 1)) // 4] - ok[(len(ok) - 1) // 4]) if len(ok) > 1 else 0
        L.append(f"{cname} & " + " & ".join(fmt_t(v) for v in vals)
                 + f" & {fmt_t(med)} & {fmt_t(iqr)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def table_search(M, path):
    L = [r"\begin{tabular}{lrrrl}", r"\toprule",
         r"Participant & Adds & First-query hit & Queries / add & Non-matching queries\\",
         r"\midrule"]
    for pid, m in M.items():
        a = m["adds"]
        hits = sum(1 for x in a if x["first_hit"])
        nq = sum(x["n_queries"] for x in a)
        miss = sorted({q for x in a for q in x["queries"] if not match(q, x["package"])})
        L.append(f"{tex(short(pid))} & {len(a)} & {hits}/{len(a)} & {nq / len(a):.1f} & "
                 f"{tex(', '.join(miss[:6]) or '--')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def table_connect(M, path):
    L = [r"\begin{tabular}{lrrrrl}", r"\toprule",
         r"Participant & Drafts & Created & Rejected & Cancelled & Rejection causes\\",
         r"\midrule"]
    for pid, m in M.items():
        causes = "; ".join(f"{k} ({v})" for k, v in m["fail_reasons"].most_common())
        L.append(f"{tex(short(pid))} & {m['conn_starts']} & {m['conn_ok']} & {m['conn_fail']} & "
                 f"{m['conn_cancel']} & {tex(causes or '--')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def table_params(M, path):
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Participant & Fields set & Edits & Corrections & Undo & Problems panel\\", r"\midrule"]
    for pid, m in M.items():
        L.append(f"{tex(short(pid))} & {m['param_fields']} & {m['param_sets']} & "
                 f"{m['param_corrections']} & {m['undo']} & {m['problems_opened']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})


def fig_progress(M, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (pid, m) in enumerate(M.items()):
        ts = [t for t in TASKS if t in m["net"]]
        ax.step([0] + [m["net"][t] / 60 for t in ts], range(len(ts) + 1), where="post",
                color=OKABE[k % 6], marker=MARKS[k % 6], ms=3.5, lw=1.4, label=short(pid))
    ax.set_xlabel("Net working time (min, consultations excluded)")
    ax.set_ylabel("Tasks completed")
    ax.set_yticks(range(0, 13, 2))
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_task_times(M, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    data, xs = [], range(len(TASKS))
    for t in TASKS:
        i, v = TASKS.index(t), []
        for m in M.values():
            if t in m["net"] and t not in m["ooo"]:
                v.append(m["net"][t] - (m["net"].get(TASKS[i - 1], 0.0) if i else 0.0))
        data.append([max(x, 1) for x in v])
    ax.boxplot(data, positions=list(xs), widths=0.55, showfliers=False,
               medianprops={"color": "black"}, boxprops={"color": "#555"},
               whiskerprops={"color": "#555"}, capprops={"color": "#555"})
    for k, m in enumerate(M.values()):
        y = [m["net"][t] - (m["net"].get(TASKS[i - 1], 0.0) if i else 0.0)
             for i, t in enumerate(TASKS) if t in m["net"] and t not in m["ooo"]]
        x = [i for i, t in enumerate(TASKS) if t in m["net"] and t not in m["ooo"]]
        ax.scatter([v + (k - 2) * 0.07 for v in x], [max(v, 1) for v in y], s=14,
                   color=OKABE[k % 6], marker=MARKS[k % 6], zorder=3, label=short(list(M)[k]))
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Time since previous completion (s, log)")
    pos = 0
    for k, (cname, ts) in enumerate(CLUSTERS):
        if k % 2 == 0:
            ax.axvspan(pos - 0.5, pos + len(ts) - 0.5, color="#000", alpha=0.045, lw=0)
        pos += len(ts)
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_friction(M, path):
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    grid = [[m["friction"][t] for t in TASKS] for m in M.values()]
    im = ax.imshow(grid, cmap="Oranges", aspect="auto", vmin=0)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([short(p) for p in M])
    ax.grid(False)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, v if v else "", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Friction events")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_timeline(by, M, path):
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for k, (pid, rows) in enumerate(by.items()):
        t0 = rows[0]["_dt"]
        away = [x for x in away_intervals(rows) if (x[1] - x[0]).total_seconds() >= AWAY_MIN]
        for s, e in away:
            ax.barh(k, (e - s).total_seconds() / 60, left=(s - t0).total_seconds() / 60,
                    color="#bbbbbb", height=0.5, hatch="///", edgecolor="#777", lw=0)
        for r in rows:
            x = (r["_dt"] - t0).total_seconds() / 60
            if r["event"] == "edge.connect" and r["ok"] == "False":
                ax.plot(x, k, "x", color=OKABE[1], ms=6)
            elif r["event"] == "connect.cancel":
                ax.plot(x, k, "|", color=OKABE[4], ms=8)
            elif r["event"] == "edit.undo":
                ax.plot(x, k, "v", color=OKABE[3], ms=5)
            elif r["event"] == "node.remove":
                ax.plot(x, k, "s", color="k", ms=3.5)
            elif r["event"] == "session.start" and r["session"] != "1":
                ax.plot([x, x], [k - 0.3, k + 0.3], color="k", lw=1.5)
        m = M[pid]
        for t in TASKS:
            if t in m["wall"]:
                ax.plot(m["wall"][t] / 60, k - 0.36, ".", color=OKABE[0], ms=3)
    ax.set_yticks(range(len(by)))
    ax.set_yticklabels([short(p) for p in by])
    ax.invert_yaxis()
    ax.set_xlabel("Elapsed time (min)")
    handles = [plt.Line2D([], [], marker="x", ls="", color=OKABE[1], label="rejected edge"),
               plt.Line2D([], [], marker="|", ls="", color=OKABE[4], label="cancelled draft"),
               plt.Line2D([], [], marker="v", ls="", color=OKABE[3], label="undo"),
               plt.Line2D([], [], marker="s", ls="", color="k", ms=4, label="node removed"),
               plt.Line2D([], [], marker=".", ls="", color=OKABE[0], label="task completed"),
               matplotlib.patches.Patch(facecolor="#bbb", hatch="///", edgecolor="#777",
                                        label="away (consultation)")]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.32))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_errors(M, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    names = [short(p) for p in M]
    ax = axes[0]
    ok = [m["conn_ok"] for m in M.values()]
    fail = [m["conn_fail"] for m in M.values()]
    canc = [m["conn_cancel"] for m in M.values()]
    ax.bar(names, ok, color=OKABE[2], label="created")
    ax.bar(names, fail, bottom=ok, color=OKABE[1], label="rejected")
    ax.bar(names, canc, bottom=[a + b for a, b in zip(ok, fail)], color=OKABE[4],
           label="cancelled")
    ax.set_ylabel("Edge drafts")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ax.bar(names, [m["param_fields"] for m in M.values()], color=OKABE[0], label="fields")
    ax.bar(names, [m["param_corrections"] for m in M.values()],
           bottom=[m["param_fields"] for m in M.values()], color=OKABE[3], label="corrections")
    ax.set_ylabel("Parameter edits")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=7)
    for a in axes:
        a.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_search(M, path):
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    pk = ["core.input", "core.flatten", "core.linear", "core.relu", "core.cross-entropy"]
    w = 0.16
    for k, (pid, m) in enumerate(M.items()):
        vals = []
        for p in pk:
            xs = [a["n_queries"] for a in m["adds"] if a["package"] == p]
            vals.append(statistics_mean(xs))
        ax.bar([i + (k - 2) * w for i in range(len(pk))], vals, w, color=OKABE[k % 6],
               label=short(pid))
    ax.set_xticks(range(len(pk)))
    ax.set_xticklabels([p.split(".")[1] for p in pk])
    ax.set_ylabel("Search queries before add")
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def statistics_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------- document
DOC = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[table]{xcolor}\usepackage{booktabs,graphicx,float,hyperref,microtype,enumitem}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{lmodern}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\setlength{\parskip}{3pt}\setlength{\parindent}{0pt}
\newcommand{\figw}{\linewidth}
\title{%(title)s}
\author{Interaction-log analysis pipeline --- rehearsal report}
\date{\today}
\begin{document}\maketitle
%(banner)s
\tableofcontents
\bigskip
\input{body.tex}
\end{document}
"""


def build_pdf(out, title, banner, body):
    (out / "body.tex").write_text(body, encoding="utf-8")
    (out / "report.tex").write_text(DOC % {"title": title, "banner": banner},
                                    encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "report.tex"], cwd=out, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:])
            raise SystemExit("pdflatex failed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--title", default="Student interaction patterns in NNModelling")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--body", help="LaTeX body (default: built-in tables+figures)")
    a = ap.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    style()
    by, tk, se = load(res)
    M = per_participant(by, tk)
    write_measures(M, out / "measures.csv")
    T, F = out / "tables", out / "figures"
    table_overview(M, T / "overview.tex")
    table_clusters(M, T / "clusters.tex")
    table_search(M, T / "search.tex")
    table_connect(M, T / "connect.tex")
    table_params(M, T / "params.tex")
    fig_progress(M, F / "progress.pdf")
    fig_task_times(M, F / "task_times.pdf")
    fig_friction(M, F / "friction.pdf")
    fig_timeline(by, M, F / "timeline.pdf")
    fig_errors(M, F / "errors.pdf")
    fig_search(M, F / "search.pdf")
    if a.no_pdf:
        return 0
    body = Path(a.body).read_text(encoding="utf-8") if a.body else ""
    banner = (r"\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{\textbf{Synthetic data.} "
              r"Every number below comes from five scripted participants; "
              r"the pipeline is real, the behaviour is not.}}" if a.synthetic else "")
    build_pdf(out, a.title, banner, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
