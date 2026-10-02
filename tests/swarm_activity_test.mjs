import assert from "node:assert/strict";
import { test } from "node:test";
import { createMailbox } from "../.opencode/swarm-mailbox/mailbox.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function clock() {
  let now = 1000;
  const timers = new Map();
  return {
    timers, now: () => now,
    setTimeout(callback, milliseconds) {
      const id = {};
      timers.set(id, { callback, deadline: now + milliseconds });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
    advance(milliseconds) {
      now += milliseconds;
      for (const [id, timer] of [...timers]) {
        if (timer.deadline <= now) {
          timers.delete(id);
          timer.callback();
        }
      }
    },
  };
}

async function fixture({ connectedMarker = true } = {}) {
  const sessions = new Map(["root", "a", "b", "other"].map((name) => [
    `ses_${name}`, { id: `ses_${name}`, projectID: "p", location: { directory: "/repo" },
      parentID: ["a", "b"].includes(name) ? "ses_root" : undefined,
      agent: "test-worker", title: name, time: { updated: 100 }, outcome: "succeeded" },
  ]));
  const storage = new Map();
  const observations = new Map();
  const inbox = [];
  const ready = deferred();
  const subscribers = new Set();
  const emit = (event) => {
    for (const subscriber of subscribers) { subscriber.queue.push(event); subscriber.wake?.(); }
  };
  const active = { ses_a: { type: "running" } };
  const contexts = new Map(["ses_a", "ses_b"].map((id) => [id, [{ type: "assistant", content: [
    { type: "reasoning", text: "PRIVATE_REASONING" },
    { type: "tool", name: "read", state: { status: "running", input: { secret: "PRIVATE_INPUT" } } },
    { type: "tool", name: "shell", state: { status: "completed", output: "PRIVATE_OUTPUT" } },
  ] }, { type: "assistant", content: [{ type: "text", text: "Final report" }] }]]));
  const ctx = {
    event: { async *subscribe({ signal }) {
      const subscriber = { queue: connectedMarker ? [{ type: "server.connected" }] : [] };
      subscribers.add(subscriber);
      const abort = () => subscriber.wake?.();
      signal.addEventListener("abort", abort, { once: true });
      try {
        while (!signal.aborted) {
          if (subscriber.queue.length) yield subscriber.queue.shift();
          else await new Promise((resolve) => { subscriber.wake = resolve; });
        }
      } finally {
        signal.removeEventListener("abort", abort);
        subscribers.delete(subscriber);
      }
    } },
    storage: { async get(key) { return storage.get(key); }, async set(key, value) { storage.set(key, value); } },
    session: {
      async get({ sessionID }) {
        if (!sessions.has(sessionID)) throw new Error("Session missing");
        return sessions.get(sessionID);
      },
      async wait({ sessionID }, { signal }) {
        const done = deferred();
        const observation = { ...done, signal, aborted: false };
        observations.set(sessionID, observation);
        const abort = () => { observation.aborted = true; done.reject(signal.reason); };
        signal.addEventListener("abort", abort, { once: true });
        try { await done.promise; } finally { signal.removeEventListener("abort", abort); }
      },
      async synthetic(input) {
        const item = { id: `msg_${inbox.length + 1}`, type: "synthetic", payload: input };
        if (input.sessionID === "ses_root") inbox.push(item);
        emit({ type: "session.inbox.enqueued", data: { sessionID: input.sessionID,
          inboxID: item.id, item: { type: "synthetic", payload: { metadata: input.metadata } } } });
        return item;
      },
      async active() { return active; },
      async context({ sessionID }) { return contexts.get(sessionID) ?? []; },
    },
  };
  const time = clock();
  const mailbox = createMailbox(ctx, time);
  await mailbox.init({}, { sessionID: "ses_root" });
  const principal = { sessionID: "ses_root", progress() { ready.resolve(); } };
  return { ctx, mailbox, time, sessions, observations, inbox, ready, principal, storage, contexts, subscribers, emit };
}

const caller = (name = "root", signal) => ({ sessionID: `ses_${name}`, signal });
const data = (result) => JSON.parse(result.content);

test("wait wakes on any watched child's native idleness and cancels only other observations", async () => {
  const f = await fixture();
  const result = f.mailbox.wait({ children: ["ses_a", "ses_b"] }, f.principal);
  await f.ready.promise;
  f.observations.get("ses_b").resolve();
  const event = data(await result);
  assert.equal(event.reason, "idle");
  assert.equal(event.sessionID, "ses_b");
  assert.deepEqual(event.children, ["ses_a", "ses_b"]);
  assert.equal(f.observations.get("ses_a").aborted, true);
  assert.equal(f.time.timers.size, 0);
  assert.equal(f.sessions.get("ses_a").outcome, "succeeded");
});

test("already-idle native child returns immediately without interpreting success", async () => {
  const f = await fixture();
  f.sessions.get("ses_a").outcome = "failed";
  f.ctx.session.wait = async () => {};
  const event = data(await f.mailbox.wait({ children: ["ses_a"] }, caller()));
  assert.equal(event.reason, "idle");
  assert.equal(event.elapsed_ms, 0);
});

test("plugin streams without server.connected do not block preparation or child completion", async () => {
  const f = await fixture({ connectedMarker: false });
  const result = f.mailbox.wait({ children: ["ses_a"] }, f.principal);
  await f.ready.promise;
  f.observations.get("ses_a").resolve();
  assert.equal(data(await result).reason, "idle");
  assert.equal(f.time.timers.size, 0);
});

test("successful incoming admission wakes a waiter across mailbox instances", async () => {
  const f = await fixture({ connectedMarker: false });
  const result = f.mailbox.wait({ children: ["ses_a"] }, f.principal);
  await f.ready.promise;
  // Native child plugins can have different context/storage objects backed by the
  // same durable store. This is the case a local WeakMap-only listener misses.
  const childContext = { ...f.ctx, storage: { ...f.ctx.storage } };
  await createMailbox(childContext, f.time).send({ to: "parent", message: "Need contract clarification" }, caller("b"));
  const event = data(await result);
  assert.equal(event.reason, "message");
  assert.equal(event.sender, "ses_b");
  assert.equal(event.inboxID, "msg_1");
  assert.equal(f.inbox.length, 1); // Observation never consumes durable input.
  assert.equal(f.inbox[0].payload.resume, true);
  assert.equal(f.observations.get("ses_a").aborted, true);
  assert.equal(f.time.timers.size, 0);
  assert.equal(f.subscribers.size, 0);
});

test("a prior eligible admission wakes a later wait without a new send or native inbox read", async () => {
  const f = await fixture();
  await f.mailbox.send({ to: "parent", message: "Pending question" }, caller("a"));
  const reloaded = createMailbox({ ...f.ctx, storage: { ...f.ctx.storage } }, f.time);
  const event = data(await reloaded.wait({ children: [] }, caller()));
  assert.equal(event.reason, "message");
  assert.equal(event.sender, "ses_a");
  assert.equal(f.inbox.length, 1);
  assert.equal(f.observations.size, 0);
});

test("messages arriving during preparation are not lost", async () => {
  const f = await fixture();
  const blocked = deferred();
  const release = deferred();
  const get = f.ctx.session.get;
  f.ctx.session.get = async (input, options) => {
    if (input.sessionID === "ses_a") { blocked.resolve(); await release.promise; }
    return get(input, options);
  };
  const result = f.mailbox.wait({ children: ["ses_a"] }, caller());
  await blocked.promise;
  await f.mailbox.send({ to: "parent", message: "Preparation race" }, caller("b"));
  release.resolve();
  assert.equal(data(await result).reason, "message");
  assert.equal(f.time.timers.size, 0);
});

test("failed synthetic admission does not wake a waiter", async () => {
  const f = await fixture();
  const result = f.mailbox.wait({ children: ["ses_a"] }, f.principal);
  await f.ready.promise;
  f.ctx.session.synthetic = async () => { throw new Error("Admission failed"); };
  await assert.rejects(f.mailbox.send({ to: "parent", message: "No admission" }, caller("b")), /Admission failed/);
  f.observations.get("ses_a").resolve();
  assert.equal(data(await result).reason, "idle");
});

test("default deadline is 270 seconds and timeouts do not poll", async () => {
  const f = await fixture();
  let reads = 0;
  const wait = f.ctx.session.wait;
  f.ctx.session.wait = (...args) => { reads++; return wait(...args); };
  const result = f.mailbox.wait({ children: ["ses_a"] }, f.principal);
  await f.ready.promise;
  f.time.advance(270000);
  const event = data(await result);
  assert.equal(event.reason, "timeout");
  assert.equal(event.elapsed_ms, 270000);
  assert.equal(reads, 1);
  assert.equal(f.observations.get("ses_a").aborted, true);
  assert.equal(f.time.timers.size, 0);
});

test("message-only waiting honors a shorter deadline", async () => {
  const f = await fixture();
  const result = f.mailbox.wait({ children: [], timeout_seconds: 2 }, f.principal);
  await f.ready.promise;
  f.time.advance(2000);
  const event = data(await result);
  assert.equal(event.reason, "timeout");
  assert.equal(event.elapsed_ms, 2000);
  assert.equal(f.observations.size, 0);
});

test("the deadline also bounds slow preparation", async () => {
  const f = await fixture();
  const started = deferred();
  f.ctx.session.get = async (_input, { signal }) => {
    started.resolve();
    return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
  };
  const result = f.mailbox.wait({ timeout_seconds: 1 }, caller());
  await started.promise;
  f.time.advance(1000);
  assert.equal(data(await result).reason, "timeout");
  assert.equal(f.time.timers.size, 0);
});

test("invalid deadlines and invalid/foreign/excluded child selections are rejected", async () => {
  const f = await fixture();
  for (const timeout_seconds of [0, -1, 271, 1.5, NaN, null, "10"]) {
    await assert.rejects(f.mailbox.wait({ timeout_seconds }, caller()), /integer from 1 to 270/);
  }
  for (const children of [null, ["ses_root"], ["ses_other"]]) {
    await assert.rejects(f.mailbox.wait({ children }, caller()));
  }
  await f.mailbox.members({}, caller("a"));
  await f.mailbox.register({ children: [] }, caller());
  await assert.rejects(f.mailbox.wait({ children: ["ses_a"] }, caller()), /explicitly excluded/);
  assert.equal(f.time.timers.size, 0);
});

test("wait/status require initialization and principal ownership", async () => {
  const f = await fixture();
  await assert.rejects(f.mailbox.wait({}, caller("a")), /Only the principal/);
  await assert.rejects(f.mailbox.status({}, caller("a")), /Only the principal/);
  f.storage.clear();
  await assert.rejects(f.mailbox.wait({}, caller()), /swarm.init/);
  await assert.rejects(f.mailbox.status({}, caller()), /swarm.init/);
  assert.equal(f.time.timers.size, 0);
});

test("missing native wait support fails explicitly with no timer leak", async () => {
  const f = await fixture();
  delete f.ctx.session.wait;
  await assert.rejects(f.mailbox.wait({}, caller()), /Runtime must expose/);
  assert.equal(f.time.timers.size, 0);
});

test("missing native event support fails explicitly rather than polling", async () => {
  const f = await fixture();
  delete f.ctx.event;
  await assert.rejects(f.mailbox.wait({}, caller()), /Runtime must expose/);
  assert.equal(f.time.timers.size, 0);
});

test("native event stream failure/end wakes unavailable and cleans up", async () => {
  for (const fail of [true, false]) {
    const f = await fixture();
    f.ctx.event.subscribe = async function* () {
      if (fail) throw new Error("Stream disconnected");
    };
    const event = data(await f.mailbox.wait({ children: [] }, caller()));
    assert.equal(event.reason, "unavailable");
    assert.equal(event.observation, "events");
    assert.equal(f.time.timers.size, 0);
  }
});

test("native wait failure returns unavailable rather than successful completion", async () => {
  const f = await fixture();
  f.ctx.session.wait = async () => { throw new Error("Session moved"); };
  const event = data(await f.mailbox.wait({ children: ["ses_a"] }, caller()));
  assert.equal(event.reason, "unavailable");
  assert.equal(event.sessionID, "ses_a");
  assert.match(event.error, /Session moved/);
  assert.equal(f.time.timers.size, 0);
});

test("cancellation clears observations/listeners and permits another wait", async () => {
  const f = await fixture();
  const controller = new AbortController();
  const result = f.mailbox.wait({ children: ["ses_a", "ses_b"] }, { ...f.principal, signal: controller.signal });
  await f.ready.promise;
  controller.abort();
  await assert.rejects(result, { name: "AbortError" });
  assert.ok([...f.observations.values()].every((item) => item.aborted));
  assert.equal(f.time.timers.size, 0);
  const next = f.mailbox.wait({ children: [] }, caller());
  // Admission wakes this new call even while its preparation is still running.
  await f.mailbox.send({ to: "parent", message: "New useful update" }, caller("a"));
  assert.equal(data(await next).reason, "message");
});

test("concurrent waits for one principal are rejected", async () => {
  const f = await fixture();
  const result = f.mailbox.wait({ children: [] }, f.principal);
  await f.ready.promise;
  await assert.rejects(createMailbox(f.ctx, f.time).wait({}, caller()), /already active/);
  f.time.advance(270000);
  assert.equal(data(await result).reason, "timeout");
});

test("status reports observable activity/outcome/tools without exposing private content", async () => {
  const f = await fixture();
  const result = await f.mailbox.status({ children: ["ses_a", "ses_b"] }, caller());
  const snapshot = data(result);
  assert.deepEqual(snapshot.children.map((item) => item.activity), ["running", "idle"]);
  assert.equal(snapshot.children[0].outcome, "succeeded"); // Last recorded outcome, not current run success.
  assert.deepEqual(snapshot.children[0].tools, [{ name: "read", status: "running" }, { name: "shell", status: "completed" }]);
  assert.equal(result.content.includes("PRIVATE_"), false);
});

test("status activity failures report unknown and individual context failures remain visible", async () => {
  const f = await fixture();
  f.ctx.session.active = async () => { throw new Error("Active snapshot unavailable"); };
  f.ctx.session.context = async () => { throw new Error("Context unavailable"); };
  const snapshot = data(await f.mailbox.status({ children: ["ses_a"] }, caller()));
  assert.equal(snapshot.children[0].activity, "unknown");
  assert.match(snapshot.activity_error, /Active snapshot/);
  assert.match(snapshot.children[0].context_error, /Context unavailable/);
});

test("a runtime without active listing still exposes metadata and latest tool names honestly", async () => {
  const f = await fixture();
  delete f.ctx.session.active;
  const snapshot = data(await f.mailbox.status({ children: ["ses_a"] }, caller()));
  assert.equal(snapshot.children[0].activity, "unknown");
  assert.match(snapshot.activity_error, /not exposed/);
  assert.equal(snapshot.children[0].tools[0].name, "read");
});

test("status reports enrolled but missing/moved children as unavailable without cross-checkout content", async () => {
  const f = await fixture();
  await f.mailbox.members({}, caller("a"));
  await f.mailbox.members({}, caller("b"));
  f.sessions.delete("ses_a");
  f.sessions.get("ses_b").location = { directory: "/other" };
  const snapshot = data(await f.mailbox.status({}, caller()));
  assert.ok(snapshot.children.every((item) => item.activity === "unavailable"));
  assert.ok(snapshot.children.every((item) => !item.tools));
});

test("excluded prior admissions do not wake or consume native input", async () => {
  const f = await fixture();
  await f.mailbox.send({ to: "parent", message: "Earlier admission" }, caller("a"));
  await f.mailbox.register({ children: [] }, caller());
  const result = f.mailbox.wait({ children: [] }, f.principal);
  await f.ready.promise;
  f.time.advance(270000);
  assert.equal(data(await result).reason, "timeout");
  assert.equal(f.inbox.length, 1);
});

test("an observed admission notification is not replayed by subsequent waits", async () => {
  const f = await fixture();
  await f.mailbox.send({ to: "parent", message: "One-time wake" }, caller("a"));
  assert.equal(data(await f.mailbox.wait({ children: [] }, caller())).reason, "message");
  const second = createMailbox({ ...f.ctx, storage: { ...f.ctx.storage } }, f.time)
    .wait({ children: [], timeout_seconds: 1 }, f.principal);
  await f.ready.promise;
  f.time.advance(1000);
  assert.equal(data(await second).reason, "timeout");
  assert.equal(f.inbox.length, 1);
});

test("notification persistence failure does not misreport admitted native input", async () => {
  const f = await fixture();
  const set = f.ctx.storage.set;
  f.ctx.storage.set = async (key, value) => {
    if (key.startsWith("wake/")) throw new Error("Notification persistence failed");
    return set(key, value);
  };
  const result = data(await f.mailbox.send({ to: "parent", message: "Admitted despite observation failure" }, caller("a")));
  assert.equal(result.admitted, true);
  assert.match(result.observation_error, /Notification persistence failed/);
  assert.equal(f.inbox.length, 1);
});

test("foreign event metadata is ignored after native family validation", async () => {
  const f = await fixture();
  const result = f.mailbox.wait({ children: [], timeout_seconds: 1 }, f.principal);
  await f.ready.promise;
  f.emit({ type: "session.inbox.enqueued", data: { sessionID: "ses_root", inboxID: "msg_foreign",
    item: { type: "synthetic", payload: { metadata: { source: "nnmodelling.swarm-mailbox", sender: "ses_other", swarm: "ses_root" } } } } });
  f.time.advance(1000);
  assert.equal(data(await result).reason, "timeout");
});
