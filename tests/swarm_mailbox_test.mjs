import assert from "node:assert/strict";
import { test } from "node:test";
import { createMailbox } from "../.opencode/swarm-mailbox/mailbox.js";
import plugin from "../.opencode/swarm-mailbox/index.js";

function fixture() {
  const sessions = new Map(["root", "a", "b", "other"].map((name) => [
    `ses_${name}`, {
      id: `ses_${name}`, parentID: ["a", "b"].includes(name) ? "ses_root" : undefined,
      projectID: "project", location: { directory: "/repo" },
      agent: name === "root" ? "orchestrator" : "core-worker",
      title: name, time: {},
    },
  ]));
  const storage = new Map();
  const admitted = [];
  const ctx = {
    session: {
      async get({ sessionID }) {
        if (!sessions.has(sessionID)) throw new Error("Session missing");
        return sessions.get(sessionID);
      },
      async synthetic(input) {
        admitted.push(input);
        return { id: `msg_${admitted.length}` };
      },
    },
    storage: {
      async get(key) { return storage.get(key); },
      async set(key, value) { storage.set(key, value); },
    },
  };
  return { ctx, sessions, storage, admitted, mailbox: createMailbox(ctx) };
}

const caller = (name) => ({ sessionID: `ses_${name}`, signal: new AbortController().signal });
const data = (result) => JSON.parse(result.content);
async function register(f) {
  await f.mailbox.init({}, caller("root"));
  return f.mailbox.register({ children: ["ses_a", "ses_b"] }, caller("root"));
}

test("initialization is explicit and principal-only, even with an empty registration", async () => {
  const f = fixture();
  await assert.rejects(f.mailbox.register({ children: [] }, caller("root")), /swarm.init/);
  await assert.rejects(f.mailbox.members({}, caller("root")), /swarm.init/);
  await assert.rejects(f.mailbox.init({}, caller("a")), /Only the principal/);
  assert.equal(f.storage.size, 0);
  const result = data(await f.mailbox.init({}, caller("root")));
  assert.equal(result.initialized, true);
  assert.equal(result.principal, "ses_root");
});

test("registration is persistent, deduplicated and discoverable by a fresh mailbox", async () => {
  const f = fixture();
  await f.mailbox.init({}, caller("root"));
  await f.mailbox.register({ children: ["ses_a", "ses_b", "ses_a"] }, caller("root"));
  const result = data(await createMailbox(f.ctx).members({}, caller("a")));
  assert.equal(result.principal, "ses_root");
  assert.deepEqual(result.members.map((m) => m.sessionID), ["ses_root", "ses_a", "ses_b"]);
});

test("registration rejects unrelated sessions without overwriting valid membership", async () => {
  const f = fixture();
  await register(f);
  await assert.rejects(f.mailbox.register({ children: ["ses_a", "ses_other"] }, caller("root")), /direct member/);
  assert.deepEqual(f.storage.get("members/ses_root").children, ["ses_a", "ses_b"]);
  await assert.rejects(f.mailbox.register({ children: [] }, caller("a")), /Only the principal/);
});

test("children cannot message before initialization, then can immediately enroll and send", async () => {
  const f = fixture();
  await assert.rejects(f.mailbox.send({ to: "parent", message: "Question" }, caller("a")), /swarm.init/);
  assert.equal(f.admitted.length, 0);
  await f.mailbox.init({}, caller("root"));
  const result = data(await f.mailbox.send({ to: "parent", message: "Early question" }, caller("a")));
  assert.equal(result.admitted, true);
  assert.deepEqual(f.storage.get("members/ses_root").children, ["ses_a"]);
});

test("addressing a new child enrolls it, and peer discovery is progressive", async () => {
  const f = fixture();
  await f.mailbox.init({}, caller("root"));
  const initial = data(await f.mailbox.members({}, caller("root")));
  assert.deepEqual(initial.members.map((m) => m.sessionID), ["ses_root"]);
  await f.mailbox.send({ to: "ses_a", message: "Accepted clarification" }, caller("root"));
  await f.mailbox.send({ to: "ses_b", message: "Interface ready" }, caller("a"));
  const discovered = data(await f.mailbox.members({}, caller("b")));
  assert.deepEqual(discovered.members.map((m) => m.sessionID), ["ses_root", "ses_a", "ses_b"]);
});

test("initialization migrates a legacy registry without allowing implicit migration", async () => {
  const f = fixture();
  f.storage.set("members/ses_root", ["ses_a", "ses_a"]);
  await assert.rejects(f.mailbox.members({}, caller("a")), /swarm.init/);
  await f.mailbox.init({}, caller("root"));
  assert.deepEqual(f.storage.get("members/ses_root"), {
    version: 1, children: ["ses_a"], excluded: [],
  });
  await f.mailbox.send({ to: "parent", message: "Migrated" }, caller("a"));
});

test("concurrent enrollment and initialization preserve more than three workers", async () => {
  const f = fixture();
  await f.mailbox.init({}, caller("root"));
  const ids = Array.from({ length: 8 }, (_, i) => `ses_worker${i}`);
  for (const id of ids) f.sessions.set(id, {
    ...f.sessions.get("ses_a"), id, title: id,
  });
  await Promise.all([
    ...ids.map((sessionID) => createMailbox(f.ctx).members({}, { sessionID })),
    f.mailbox.init({}, caller("root")),
    f.mailbox.members({}, { sessionID: ids[0] }),
  ]);
  const state = f.storage.get("members/ses_root");
  assert.equal(state.children.length, ids.length);
  assert.deepEqual(new Set(state.children), new Set(ids));
  const result = data(await f.mailbox.broadcast({ message: "Independent assignments" }, caller("root")));
  assert.equal(result.results.length, ids.length);
  assert.ok(result.results.every((r) => r.admitted));
});

test("invalid stored state is not reset by initialization", async () => {
  const f = fixture();
  f.storage.set("members/ses_root", { version: 1, children: [], excluded: null });
  const previous = f.storage.get("members/ses_root");
  await assert.rejects(f.mailbox.init({}, caller("root")), /Invalid persisted/);
  assert.deepEqual(f.storage.get("members/ses_root"), previous);
});

test("child-to-parent admission attributes the sender and explicitly wakes an idle recipient", async () => {
  const f = fixture();
  await register(f);
  const result = data(await f.mailbox.send({ to: "parent", message: "Contract question" }, caller("a")));
  assert.equal(result.inboxID, "msg_1");
  assert.equal(f.admitted[0].sessionID, "ses_root");
  assert.equal(f.admitted[0].resume, true);
  assert.equal(f.admitted[0].delivery, "steer");
  assert.equal(f.admitted[0].metadata.sender, "ses_a");
  assert.match(f.admitted[0].text, /Contract question/);
});

test("sibling and principal-to-child messages support queued delivery", async () => {
  const f = fixture();
  await register(f);
  await f.mailbox.send({ to: "ses_b", message: "Interface ready", delivery: "queue" }, caller("a"));
  await f.mailbox.send({ to: "ses_a", message: "Accepted clarification" }, caller("root"));
  assert.deepEqual(f.admitted.map((m) => [m.sessionID, m.delivery, m.resume]),
    [["ses_b", "queue", true], ["ses_a", "steer", true]]);
});

test("foreign, self, archived and moved recipients cannot receive messages", async () => {
  const f = fixture();
  await register(f);
  await assert.rejects(f.mailbox.send({ to: "ses_other", message: "No" }, caller("a")), /direct member/);
  await assert.rejects(f.mailbox.send({ to: "ses_a", message: "No" }, caller("a")), /yourself/);
  f.sessions.get("ses_b").time.archived = 123;
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: "No" }, caller("a")), /archived/);
  delete f.sessions.get("ses_b").time.archived;
  f.sessions.get("ses_b").location.directory = "/other-worktree";
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: "No" }, caller("a")), /direct member/);
  assert.equal(f.admitted.length, 0);
});

test("automatic enrollment rejects unrelated, nested, moved and archived senders", async () => {
  const f = fixture();
  await f.mailbox.init({}, caller("root"));
  await f.mailbox.init({}, caller("other"));
  await assert.rejects(f.mailbox.send({ to: "ses_root", message: "No" }, caller("other")), /direct member/);
  f.sessions.get("ses_a").location.directory = "/other";
  await assert.rejects(f.mailbox.members({}, caller("a")), /one checkout/);
  f.sessions.get("ses_a").location.directory = "/repo";
  f.sessions.get("ses_a").projectID = "other";
  await assert.rejects(f.mailbox.members({}, caller("a")), /one checkout/);
  f.sessions.get("ses_a").projectID = "project";
  f.sessions.get("ses_a").time.archived = 1;
  await assert.rejects(f.mailbox.members({}, caller("a")), /archived/);
  f.sessions.get("ses_b").parentID = "ses_a";
  await assert.rejects(f.mailbox.init({}, caller("b")), /direct children/);
  assert.deepEqual(f.storage.get("members/ses_root").children, []);
  assert.equal(f.admitted.length, 0);
});

test("invalid first messages do not enroll a sender or recipient", async () => {
  const f = fixture();
  await f.mailbox.init({}, caller("root"));
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: " " }, caller("a")), /1 to 16000/);
  await assert.rejects(f.mailbox.broadcast({ message: "Hi", delivery: "bad" }, caller("a")), /Delivery/);
  assert.deepEqual(f.storage.get("members/ses_root").children, []);
  assert.equal(f.admitted.length, 0);
});

test("nested and cross-project children cannot join the swarm", async () => {
  const f = fixture();
  f.sessions.get("ses_b").projectID = "other-project";
  await assert.rejects(register(f), /direct member/);
  f.sessions.get("ses_a").parentID = "ses_b";
  await assert.rejects(f.mailbox.members({}, caller("a")), /direct children/);
});

test("broadcast excludes sender and reports partial admission failure", async () => {
  const f = fixture();
  await register(f);
  const original = f.ctx.session.synthetic;
  f.ctx.session.synthetic = async (input) => {
    if (input.sessionID === "ses_b") throw new Error("Admission failed");
    return original(input);
  };
  const result = data(await f.mailbox.broadcast({ message: "Interface update" }, caller("a")));
  assert.deepEqual(result.results.map((r) => [r.to, r.admitted]), [["ses_root", true], ["ses_b", false]]);
  assert.match(result.results[1].error, /Admission failed/);
  assert.equal(f.admitted.length, 1);
});

test("invalid message, delivery and cancelled calls admit nothing", async () => {
  const f = fixture();
  await register(f);
  for (const message of ["", "  ", "x".repeat(16001), null]) {
    await assert.rejects(f.mailbox.send({ to: "ses_b", message }, caller("a")), /1 to 16000/);
  }
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: "Hi", delivery: "invalid" }, caller("a")), /Delivery/);
  const context = { sessionID: "ses_a", signal: AbortSignal.abort() };
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: "Hi" }, context), { name: "AbortError" });
  assert.equal(f.admitted.length, 0);
});

test("explicit removal survives initialization and reload, and registration restores access", async () => {
  const f = fixture();
  await register(f);
  await f.mailbox.register({ children: ["ses_a"] }, caller("root"));
  await f.mailbox.init({}, caller("root"));
  const fresh = createMailbox(f.ctx);
  await assert.rejects(fresh.send({ to: "parent", message: "Hi" }, caller("b")), /explicitly excluded/);
  await assert.rejects(fresh.members({}, caller("b")), /explicitly excluded/);
  await assert.rejects(fresh.send({ to: "ses_b", message: "Hi" }, caller("root")), /explicitly excluded/);
  assert.deepEqual(f.storage.get("members/ses_root").excluded, ["ses_b"]);
  await fresh.register({ children: ["ses_a", "ses_b"] }, caller("root"));
  assert.deepEqual(f.storage.get("members/ses_root").excluded, []);
  await fresh.send({ to: "parent", message: "Restored" }, caller("b"));
});

test("cancelled initialization, enrollment and replacement preserve stored state", async () => {
  const f = fixture();
  const signal = AbortSignal.abort();
  await assert.rejects(f.mailbox.init({}, { sessionID: "ses_root", signal }), { name: "AbortError" });
  assert.equal(f.storage.size, 0);
  await f.mailbox.init({}, caller("root"));
  await assert.rejects(f.mailbox.members({}, { sessionID: "ses_a", signal }), { name: "AbortError" });
  await assert.rejects(f.mailbox.register({ children: ["ses_a"] }, { sessionID: "ses_root", signal }), { name: "AbortError" });
  assert.deepEqual(f.storage.get("members/ses_root").children, []);
});

test("concurrent explicit replacement and enrollment retain membership and exclusions", async () => {
  const f = fixture();
  await register(f);
  f.sessions.set("ses_c", { ...f.sessions.get("ses_a"), id: "ses_c" });
  await Promise.all([
    f.mailbox.register({ children: ["ses_a"] }, caller("root")),
    createMailbox(f.ctx).members({}, { sessionID: "ses_c" }),
  ]);
  assert.deepEqual(new Set(f.storage.get("members/ses_root").children), new Set(["ses_a", "ses_c"]));
  assert.deepEqual(f.storage.get("members/ses_root").excluded, ["ses_b"]);
});

test("plugin registers executable Code Mode tools against the V2 context", async () => {
  const f = fixture();
  const tools = new Map();
  f.ctx.tool = { async transform(callback) {
    callback({ namespace(value) { assert.equal(value.name, "swarm"); },
      add(tool) { tools.set(tool.name, tool); } });
  } };
  await plugin.setup(f.ctx);
  assert.deepEqual([...tools.keys()], ["init", "register", "members", "wait", "status", "send", "broadcast"]);
  for (const tool of tools.values()) assert.equal(tool.options.codemode, true);
  await tools.get("init").execute({}, caller("root"));
  await tools.get("send").execute({ to: "parent", message: "Question" }, caller("a"));
  assert.equal(f.admitted[0].metadata.sender, "ses_a");
});
