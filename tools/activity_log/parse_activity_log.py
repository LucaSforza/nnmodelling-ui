#!/usr/bin/env python3
"""Parse NNModelling activity logs (schema v1) into CSV tables and a summary.

Usage: parse_activity_log.py LOG... --out DIR [--tasks FILE] [--idle-seconds 30]

One input file is one participant (id = file stem). Sessions inside a file are
ordered by their first ts and concatenated. Malformed lines, unsupported schema
versions and seq gaps are skipped/reported, unknown events and keys ignored.

Task predicate language (JSON), evaluated after every record against the
latest model.snapshot of the current session (state resets at a new session):
  {"all": [P...]}  {"any": [P...]}  {"not": P}
  {"node": M}                  some node matches M
  {"edge": {"from": M, "to": M}}  some edge joins a node matching each
  {"active_dataset": "id"}     snapshot active_dataset.id equals id
  {"event": "name", "data": {k: v}}  the current record is a successful event
                               with that name whose data contains k == v
Node matcher M (all given keys must hold):
  "package": "core.linear"
  "params": {key: value | {"nonempty": true}}   numbers compare numerically
  "has_input_from": M          some incoming edge from a node matching M
  "has_output_to": M           some outgoing edge to a node matching M
A task completes at the first record where its predicate holds.
"""
import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ACTION_UI = {"ui.action", "palette.add", "connect.start", "join.slots",
             "problem.click", "layout.arrange", "view.fit", "drag.start"}
NON_ACTION_CORE = {"model.snapshot", "analysis.result"}
FLAT = ["node", "package", "source", "target", "key", "new", "query",
        "action", "via", "dialog"]
EVENT_COLS = ["participant", "session", "seq", "ts", "t", "src", "event",
              "ok", "error"] + FLAT + ["data"]


# ---------------------------------------------------------------- loading
def load_log(path, warn=None):
    """Return (records, bad_lines, seq_gaps) for one file; records are dicts
    in session order (sessions sorted by first ts, file order inside)."""
    bad, gaps, sessions = [], [], {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad.append((n, "malformed JSON"))
                continue
            if not isinstance(rec, dict) or rec.get("v") != 1:
                bad.append((n, "unsupported schema version"))
                continue
            if not all(k in rec for k in ("ts", "session", "seq", "event")):
                bad.append((n, "missing required key"))
                continue
            try:
                rec["_dt"] = parse_ts(rec["ts"])
            except (ValueError, TypeError):
                bad.append((n, "bad ts"))
                continue
            if not isinstance(rec.get("data"), dict):
                rec["data"] = {}
            sessions.setdefault(rec["session"], []).append(rec)
    for sid, recs in sessions.items():
        prev = None
        for r in recs:
            if prev is not None and r["seq"] != prev + 1:
                gaps.append((sid, prev, r["seq"]))
            prev = r["seq"]
    ordered = sorted(sessions.items(), key=lambda kv: kv[1][0]["_dt"])
    records = []
    for idx, (sid, recs) in enumerate(ordered, 1):
        for r in recs:
            r["_session_no"] = idx
            records.append(r)
    if warn:
        for n, why in bad:
            warn(f"{path}:{n}: skipped line ({why})")
        for sid, a, b in gaps:
            warn(f"{path}: session {sid[:8]} seq gap {a} -> {b}")
    return records, bad, gaps


def parse_ts(text):
    return datetime.fromisoformat(text)


def load_tasks(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["tasks"]


# ------------------------------------------------------- state & predicates
def empty_state():
    return {"nodes": [], "edges": [], "active_dataset": None}


def apply_record(state, rec):
    if rec["event"] == "model.snapshot":
        d = rec["data"]
        state["nodes"] = d.get("nodes") or []
        state["edges"] = d.get("edges") or []
        state["active_dataset"] = d.get("active_dataset")
    return state


def _num(v):
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return v.strip()
    return v


def value_matches(actual, want):
    if isinstance(want, dict) and "nonempty" in want:
        a = actual["value"] if isinstance(actual, dict) and "value" in actual else actual
        return bool(want["nonempty"]) == (a not in (None, "", [], {}))
    return actual is not None and _num(actual) == _num(want)


def node_matches(state, node, m):
    if "package" in m and node.get("package") != m["package"]:
        return False
    params = node.get("params") or {}
    for k, want in (m.get("params") or {}).items():
        if not value_matches(params.get(k), want):
            return False
    by_id = {n.get("id"): n for n in state["nodes"]}
    if "has_input_from" in m and not any(
            e.get("target") == node.get("id") and e.get("source") in by_id
            and node_matches(state, by_id[e["source"]], m["has_input_from"])
            for e in state["edges"]):
        return False
    if "has_output_to" in m and not any(
            e.get("source") == node.get("id") and e.get("target") in by_id
            and node_matches(state, by_id[e["target"]], m["has_output_to"])
            for e in state["edges"]):
        return False
    return True


def eval_predicate(pred, state, rec):
    if "all" in pred:
        return all(eval_predicate(p, state, rec) for p in pred["all"])
    if "any" in pred:
        return any(eval_predicate(p, state, rec) for p in pred["any"])
    if "not" in pred:
        return not eval_predicate(pred["not"], state, rec)
    if "node" in pred:
        return any(node_matches(state, n, pred["node"]) for n in state["nodes"])
    if "edge" in pred:
        by_id = {n.get("id"): n for n in state["nodes"]}
        for e in state["edges"]:
            s, t = by_id.get(e.get("source")), by_id.get(e.get("target"))
            if s and t and node_matches(state, s, pred["edge"].get("from", {})) \
                    and node_matches(state, t, pred["edge"].get("to", {})):
                return True
        return False
    if "active_dataset" in pred:
        ds = state["active_dataset"]
        return isinstance(ds, dict) and ds.get("id") == pred["active_dataset"]
    if "event" in pred:
        if rec["event"] != pred["event"] or rec.get("ok") is not True:
            return False
        return all(rec["data"].get(k) == v
                   for k, v in (pred.get("data") or {}).items())
    return False


def find_completions(records, tasks):
    """Return {task_id: record index} of first satisfying record."""
    done, state, session = {}, empty_state(), None
    for i, rec in enumerate(records):
        if rec["session"] != session:
            session, state = rec["session"], empty_state()
        apply_record(state, rec)
        for task in tasks:
            if task["id"] not in done and eval_predicate(task["predicate"], state, rec):
                done[task["id"]] = i
    return done


# ---------------------------------------------------------------- metrics
def is_action(rec):
    ev = rec["event"]
    if rec.get("src") in ("core", "ipc"):
        return ev not in NON_ACTION_CORE
    return ev in ACTION_UI


def window_metrics(records, lo, hi):
    """Metrics over records[lo+1 .. hi] (inclusive)."""
    m = {"actions": 0, "failed_ops": 0, "undos": 0, "errors_shown": 0,
         "palette_queries": [], "connect_cancels": 0, "first_action": None}
    for i in range(lo + 1, hi + 1):
        r = records[i]
        if is_action(r):
            m["actions"] += 1
            if m["first_action"] is None:
                m["first_action"] = i
        if r.get("ok") is False:
            m["failed_ops"] += 1
        ev = r["event"]
        if ev in ("edit.undo", "edit.redo"):
            m["undos"] += 1
        elif ev == "error.shown":
            m["errors_shown"] += 1
        elif ev == "palette.search" and r["data"].get("query"):
            m["palette_queries"].append(str(r["data"]["query"]))
        elif ev == "connect.cancel":
            m["connect_cancels"] += 1
    return m


def task_rows(participant, records, tasks, done):
    rows, ids = [], [t["id"] for t in tasks]
    for k, tid in enumerate(ids):
        idx = done.get(tid)
        base = -1
        if idx is not None:
            earlier = [done[j] for j in ids[:k] if j in done and done[j] <= idx]
        else:
            earlier = [done[j] for j in ids[:k] if j in done]
        if earlier:
            base = max(earlier)
        hi = idx if idx is not None else len(records) - 1
        m = window_metrics(records, base, hi) if records else window_metrics([], -1, -1)
        t0 = records[base]["_dt"] if base >= 0 else (records[0]["_dt"] if records else None)
        row = {"participant": participant, "task": tid,
               "completed": idx is not None, "completed_at": "",
               "elapsed_s": "", "active_s": "", "actions": m["actions"],
               "failed_ops": m["failed_ops"], "undos": m["undos"],
               "errors_shown": m["errors_shown"],
               "palette_queries": "|".join(m["palette_queries"]),
               "connect_cancels": m["connect_cancels"], "out_of_order": ""}
        if idx is not None:
            end = records[idx]["_dt"]
            row["completed_at"] = records[idx]["ts"]
            row["elapsed_s"] = round((end - t0).total_seconds(), 3)
            fa = m["first_action"]
            row["active_s"] = round((end - records[fa]["_dt"]).total_seconds(), 3) \
                if fa is not None else 0.0
            row["out_of_order"] = any(
                j not in done or done[j] > idx for j in ids[:k])
        rows.append(row)
    return rows


def session_rows(participant, records, idle_seconds):
    rows, by = [], {}
    for r in records:
        by.setdefault(r["_session_no"], []).append(r)
    for no, recs in sorted(by.items()):
        idle, prev = 0.0, None
        for r in recs:
            t = r.get("t")
            if prev is not None and isinstance(t, (int, float)):
                gap = (t - prev) / 1000.0
                if gap > idle_seconds:
                    idle += gap
            if isinstance(t, (int, float)):
                prev = t
        start, end = recs[0]["_dt"], recs[-1]["_dt"]
        cnt = lambda ev: sum(1 for r in recs if r["event"] == ev)
        rows.append({
            "participant": participant, "session": no,
            "start_ts": recs[0]["ts"], "end_ts": recs[-1]["ts"],
            "duration_s": round((end - start).total_seconds(), 3),
            "events": len(recs),
            "user_actions": sum(1 for r in recs if is_action(r)),
            "failed_ops": sum(1 for r in recs if r.get("ok") is False),
            "undo": cnt("edit.undo"), "redo": cnt("edit.redo"),
            "errors_shown": cnt("error.shown"),
            "palette_searches": cnt("palette.search"),
            "idle_s": round(idle, 3), "blur_count": cnt("app.blur"),
            "ended_cleanly": cnt("session.end") > 0})
    return rows


def event_rows(participant, records):
    for r in records:
        d = r["data"]
        node = d.get("node")
        if node is None and r["event"].startswith("node."):
            node = d.get("id")
        row = {"participant": participant, "session": r["_session_no"],
               "seq": r["seq"], "ts": r["ts"], "t": r.get("t"),
               "src": r.get("src"), "event": r["event"], "ok": r.get("ok"),
               "error": r.get("error", ""), "node": node,
               "data": json.dumps(d, separators=(",", ":"), ensure_ascii=False)}
        for k in FLAT[1:]:
            row[k] = d.get(k)
        yield row


# ---------------------------------------------------------------- writers
def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def write_summary(path, tasks, task_all, quality, participants):
    def stat(vals, fn):
        return f"{fn(vals):.1f}" if vals else "-"
    lines = ["# Activity log summary", "",
             f"Participants: {len(participants)}", "",
             "## Tasks", "",
             "| Task | Completed | Rate | Mean elapsed s | Median elapsed s |"
             " Mean active s | Median active s | Mean failed ops |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in tasks:
        rows = [r for r in task_all if r["task"] == t["id"]]
        ok = [r for r in rows if r["completed"]]
        el = [r["elapsed_s"] for r in ok]
        ac = [r["active_s"] for r in ok]
        fo = [r["failed_ops"] for r in rows]
        rate = f"{100 * len(ok) / len(rows):.0f}%" if rows else "-"
        lines.append(f"| {t['id']} | {len(ok)}/{len(rows)} | {rate} | "
                     f"{stat(el, statistics.mean)} | {stat(el, statistics.median)} | "
                     f"{stat(ac, statistics.mean)} | {stat(ac, statistics.median)} | "
                     f"{stat(fo, statistics.mean)} |")
    lines += ["", "## Data quality", "",
              f"- Malformed/skipped lines: {sum(len(q['bad']) for q in quality)}",
              f"- Seq gaps: {sum(len(q['gaps']) for q in quality)}",
              f"- Sessions without session.end: {sum(q['no_end'] for q in quality)}",
              ""]
    for q in quality:
        if q["bad"] or q["gaps"] or q["no_end"]:
            lines.append(f"- {q['participant']}: {len(q['bad'])} bad lines, "
                         f"{len(q['gaps'])} seq gaps, {q['no_end']} sessions without end")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- main
def process(paths, tasks, idle_seconds=30, warn=None):
    ev_rows, ses_rows, task_all, quality, parts = [], [], [], [], []
    for p in paths:
        pid = Path(p).stem
        parts.append(pid)
        records, bad, gaps = load_log(p, warn)
        done = find_completions(records, tasks)
        ev_rows.extend(event_rows(pid, records))
        sr = session_rows(pid, records, idle_seconds)
        ses_rows.extend(sr)
        task_all.extend(task_rows(pid, records, tasks, done))
        quality.append({"participant": pid, "bad": bad, "gaps": gaps,
                        "no_end": sum(1 for s in sr if not s["ended_cleanly"])})
    return ev_rows, ses_rows, task_all, quality, parts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=str(Path(__file__).with_name("tasks_bachelor.json")))
    ap.add_argument("--idle-seconds", type=float, default=30)
    a = ap.parse_args(argv)
    tasks = load_tasks(a.tasks)
    warn = lambda msg: print("warning: " + msg, file=sys.stderr)
    ev, ses, tk, quality, parts = process(a.logs, tasks, a.idle_seconds, warn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "events.csv", EVENT_COLS, ev)
    write_csv(out / "sessions.csv",
              ["participant", "session", "start_ts", "end_ts", "duration_s", "events",
               "user_actions", "failed_ops", "undo", "redo", "errors_shown",
               "palette_searches", "idle_s", "blur_count", "ended_cleanly"], ses)
    write_csv(out / "tasks.csv",
              ["participant", "task", "completed", "completed_at", "elapsed_s",
               "active_s", "actions", "failed_ops", "undos", "errors_shown",
               "palette_queries", "connect_cancels", "out_of_order"], tk)
    write_summary(out / "summary.md", tasks, tk, quality, parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
