#!/usr/bin/env python3
"""Plot one expert participant's NNModelling activity log, per task.

Usage: plot_session.py LOG --out DIR [--rename OLD=NEW]... [--idle-seconds 30]

Splits LOG at task project.open boundaries (split_activity_log.py), then writes
into DIR: fig-timeline, fig-time, fig-problems, fig-actions (.pdf and .png) and
metrics.json. --rename maps metavariable names before sub-goal matching (the
task files expect `d-model`/`num-heads`), e.g. --rename D=d-model.
Needs matplotlib; reads only, never modifies LOG.
"""
import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "activity_log"))
import parse_activity_log as pal  # noqa: E402
import split_activity_log as split  # noqa: E402

TASK_LABEL = {"task1-mlp": "T1 MLP", "task2-multihead": "T2 Multi-head"}
TASK_FILE = {t: HERE.parent / "activity_log" / f"tasks_expert_t{n}.json" for n, t in enumerate(split.TASKS, 1)}

# Reference palette (dataviz skill): categorical slots 1-3, blue ordinal steps, status critical.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
ACTIVE, IDLE, AWAY = "#2a78d6", "#86b6ef", "#d6d5cf"
CRITICAL, INK, INK2, MUTED, GRID, AXIS = "#d03b3b", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

ACTIONS = [  # (label, predicate on record)
    ("Nodes added", lambda r: r["event"] == "node.add"),
    ("Edges connected", lambda r: r["event"] == "edge.connect" and r.get("ok")),
    ("Connections abandoned", lambda r: r["event"] == "connect.cancel"),
    ("Parameter edits", lambda r: r["event"] == "param.set" and r.get("ok")),
    ("Rejected edits / errors", lambda r: r["event"] == "error.shown"),
    ("Package searches", lambda r: r["event"] == "palette.search"),
    ("Zoom in/out", lambda r: r["event"] == "ui.action" and r["data"].get("action") in ("zoomIn", "zoomOut")),
    ("Arrange", lambda r: r["event"] == "layout.arrange"),
    ("Problem clicked", lambda r: r["event"] == "problem.click"),
    ("Undo / delete", lambda r: r["event"] in ("edit.undo", "node.remove", "edge.disconnect")),
]


def ts(rec):
    return datetime.fromisoformat(rec["ts"]).timestamp()


def rename_refs(obj, ren):
    if isinstance(obj, dict):
        if set(obj) == {"$ref"} and obj["$ref"] in ren:
            return {"$ref": ren[obj["$ref"]]}
        return {k: rename_refs(v, ren) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename_refs(v, ren) for v in obj]
    if isinstance(obj, str):
        for old, new in ren.items():
            obj = obj.replace(f'meta["{old}"]', f'meta["{new}"]')
    return obj


def task_metrics(records, task, ren, idle_s):
    t0 = next((ts(r) for r in records if r["event"] == "project.open"), ts(records[0]))
    t1 = next((ts(r) for r in reversed(records) if r["event"] == "project.close"), ts(records[-1]))
    rel = lambda r: (ts(r) - t0) / 60.0  # noqa: E731

    away, start = [], None  # app out of focus intervals (minutes)
    for r in records:
        if r["event"] == "app.blur" and start is None:
            start = rel(r)
        elif r["event"] == "app.focus" and start is not None:
            away.append((max(start, 0.0), rel(r)))
            start = None
    if start is not None:
        away.append((start, (t1 - t0) / 60))

    def away_overlap(a, b):
        return sum(max(0.0, min(b, y) - max(a, x)) for x, y in away)

    idle, prev = [], 0.0  # gaps between user actions longer than idle_s, while in focus
    for r in [r for r in records if pal.is_action(r)] + [{"ts": records[-1]["ts"], "_end": True}]:
        now = rel(r) if "_end" not in r else (t1 - t0) / 60
        if (now - prev) * 60 > idle_s and (now - prev) - away_overlap(prev, now) > idle_s / 60:
            idle.append((prev, now))
        prev = max(prev, now)
    total = (t1 - t0) / 60
    away_total = sum(b - a for a, b in away)
    idle_total = sum((b - a) - away_overlap(a, b) for a, b in idle)

    tasks = pal.load_tasks(TASK_FILE[task])
    norm = [dict(r, data=rename_refs(r.get("data"), ren)) if ren else r for r in records]
    done = pal.find_completions(norm, tasks)
    subgoals = [{"id": t["id"], "title": t["title"],
                 "minute": round(rel(records[done[t["id"]]]), 3) if t["id"] in done else None} for t in tasks]

    problems = [(rel(r), r["data"].get("problems", 0)) for r in records if r["event"] == "analysis.result"]
    errors = [{"minute": round(rel(r), 3), "message": r["data"].get("message", "")}
              for r in records if r["event"] == "error.shown"]
    metavars = []
    for r in records:
        if r["event"] == "metavariables.set":
            try:
                items = json.loads(r["data"].get("json", "[]"))
            except ValueError:
                items = []
            text = ", ".join(f"{i.get('name')}={i.get('value'):g}" if isinstance(i.get("value"), (int, float))
                             else f"{i.get('name')}={i.get('value')}" for i in items)
            metavars.append({"minute": round(rel(r), 3), "values": text})
    return {
        "task": task, "minutes_total": round(total, 2), "minutes_away": round(away_total, 2),
        "minutes_idle": round(idle_total, 2), "minutes_active": round(total - away_total - idle_total, 2),
        "away": away, "idle": idle, "subgoals": subgoals, "errors": errors, "metavariables": metavars, "problems": problems,
        "actions": {label: sum(1 for r in records if pred(r)) for label, pred in ACTIONS},
    }


def style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def plot_timeline(ms, out):
    fig, axes = plt.subplots(len(ms), 1, figsize=(9, 1.55 * len(ms) + 0.5), sharex=True)
    xmax = max(m["minutes_total"] for m in ms)
    for ax, m in zip(axes, ms):
        style(ax)
        ax.broken_barh([(0, m["minutes_total"])], (0, 1), color=ACTIVE, linewidth=0)
        ax.broken_barh([(a, b - a) for a, b in m["idle"]], (0, 1), color=IDLE, linewidth=0)
        ax.broken_barh([(a, b - a) for a, b in m["away"]], (0, 1), color=AWAY, linewidth=0)
        done = sorted((s["minute"], s["id"]) for s in m["subgoals"] if s["minute"] is not None)
        last = -9
        for i, (x, sid) in enumerate(done):
            ax.plot([x, x], [1.05, 1.35], color=INK, linewidth=1)
            level = 1.45 if x - last > 0.35 else 1.85
            ax.text(x, level, sid, ha="center", va="bottom", fontsize=6.5, color=INK)
            last = x if level == 1.45 else last
        for e in m["errors"]:
            ax.plot(e["minute"], -0.35, marker="x", color=CRITICAL, markersize=6, markeredgewidth=1.6)
        missing = [s["id"] for s in m["subgoals"] if s["minute"] is None]
        title = f"{TASK_LABEL[m['task']]}  ·  {m['minutes_total']:.1f} min"
        if missing:
            title += f"  ·  not reached: {', '.join(missing)}"
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=14)
        ax.set_ylim(-0.7, 2.3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, xmax * 1.01)
    axes[-1].set_xlabel("minutes since the task project was opened", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ACTIVE, IDLE, AWAY)]
    handles += [plt.Line2D([], [], color=INK, linewidth=1),
                plt.Line2D([], [], marker="x", color=CRITICAL, linestyle="", markeredgewidth=1.6)]
    fig.legend(handles, ["working in the app", "in the app, no action > 30 s", "app out of focus",
                         "sub-goal reached", "error message shown"],
               loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig-timeline")


def plot_time(ms, out):
    fig, ax = plt.subplots(figsize=(7, 2.3))
    style(ax)
    ys = list(range(len(ms)))[::-1]
    for y, m in zip(ys, ms):
        left = 0
        for key, color, ink in (("minutes_active", ACTIVE, "#ffffff"), ("minutes_idle", IDLE, INK),
                                ("minutes_away", AWAY, INK)):
            w = m[key]
            ax.barh(y, w - 0.06, left=left, height=0.55, color=color, linewidth=0)
            if w > 1.2:
                ax.text(left + w / 2, y, f"{w:.1f}", ha="center", va="center", fontsize=7.5, color=ink)
            left += w
        ax.text(left + 0.25, y, f"{m['minutes_total']:.1f} min", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [TASK_LABEL[m["task"]] for m in ms], fontsize=8.5, color=INK)
    ax.set_xlabel("minutes", fontsize=8, color=INK2)
    ax.set_xlim(0, max(m["minutes_total"] for m in ms) * 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ACTIVE, IDLE, AWAY)]
    ax.legend(handles, ["working in the app", "in the app, no action > 30 s", "app out of focus"],
              fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.45, -0.55))
    save(fig, out, "fig-time")


def plot_problems(ms, out):
    fig, axes = plt.subplots(1, len(ms), figsize=(9, 2.4), sharey=True)
    ymax = max((p for m in ms for _, p in m["problems"]), default=1)
    for ax, m in zip(axes, ms):
        style(ax, "y")
        xs = [x for x, _ in m["problems"]] + [m["minutes_total"]]
        ys = [p for _, p in m["problems"]]
        ys = ys + ys[-1:]
        ax.step(xs, ys, where="post", color=ACTIVE, linewidth=2)
        for k, mv in enumerate(m["metavariables"]):
            ax.axvline(mv["minute"], color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
            ax.text(mv["minute"] - 0.15, ymax + 0.6 - 1.1 * (k % 3), mv["values"], ha="right", va="top",
                    fontsize=6, color=INK2, bbox={"facecolor": "#ffffff", "edgecolor": "none", "pad": 1})
        ax.set_title(TASK_LABEL[m["task"]], loc="left", fontsize=9, color=INK)
        ax.set_xlim(0, m["minutes_total"])
        ax.set_ylim(0, ymax + 1)
        ax.set_xlabel("minutes", fontsize=8, color=INK2)
    axes[0].set_ylabel("Model Problems", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig-problems")


def plot_actions(ms, out):
    labels = [label for label, _ in ACTIONS]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style(ax)
    h = 0.26
    for i, m in enumerate(ms):
        ys = [len(labels) - 1 - k + ((len(ms) - 1) / 2 - i) * h for k in range(len(labels))]
        vals = [m["actions"][label] for label in labels]
        ax.barh(ys, vals, height=h - 0.04, color=SERIES[i], linewidth=0, label=TASK_LABEL[m["task"]])
        for y, v in zip(ys, vals):
            ax.text(v + 0.4, y, str(v), va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(labels)), labels[::-1], fontsize=8, color=INK)
    ax.set_xlabel("count", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, out, "fig-actions")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--idle-seconds", type=float, default=30)
    args = ap.parse_args(argv)
    ren = dict(item.split("=", 1) for item in args.rename)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        split.split_log(args.log, tmp, lambda msg: print(msg, file=sys.stderr))
        ms = []
        for task in split.TASKS:
            for f in sorted((Path(tmp) / task).glob("*.jsonl")):
                records, _, _ = pal.load_log(f)
                if records:
                    ms.append(task_metrics(records, task, ren, args.idle_seconds))
    plot_timeline(ms, out)
    plot_time(ms, out)
    plot_problems(ms, out)
    plot_actions(ms, out)
    (out / "metrics.json").write_text(json.dumps(ms, indent=1), encoding="utf-8")
    for m in ms:
        reached = sum(s["minute"] is not None for s in m["subgoals"])
        print(f"{TASK_LABEL[m['task']]:14} {m['minutes_total']:5.1f} min  active {m['minutes_active']:5.1f}"
              f"  idle {m['minutes_idle']:4.1f}  away {m['minutes_away']:4.1f}"
              f"  sub-goals {reached}/{len(m['subgoals'])}  errors {len(m['errors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
