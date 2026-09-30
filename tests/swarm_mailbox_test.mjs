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
const register = (f) => f.mailbox.register({ children: ["ses_a", "ses_b"] }, caller("root"));

test("registration is persistent, deduplicated and discoverable by a fresh mailbox", async () => {
  const f = fixture();
  await f.mailbox.register({ children: ["ses_a", "ses_b", "ses_a"] }, caller("root"));
  const result = data(await createMailbox(f.ctx).members({}, caller("a")));
  assert.equal(result.principal, "ses_root");
  assert.deepEqual(result.members.map((m) => m.sessionID), ["ses_root", "ses_a", "ses_b"]);
});

test("registration rejects unrelated sessions without overwriting valid membership", async () => {
  const f = fixture();
  await register(f);
  await assert.rejects(f.mailbox.register({ children: ["ses_a", "ses_other"] }, caller("root")), /direct member/);
  assert.deepEqual(f.storage.get("members/ses_root"), ["ses_a", "ses_b"]);
  await assert.rejects(f.mailbox.register({ children: [] }, caller("a")), /Only the principal/);
});

test("unregistered children cannot message the principal", async () => {
  const f = fixture();
  await assert.rejects(f.mailbox.send({ to: "parent", message: "Question" }, caller("a")), /register this child/);
  assert.equal(f.admitted.length, 0);
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
  await assert.rejects(f.mailbox.send({ to: "ses_other", message: "No" }, caller("a")), /not registered/);
  await assert.rejects(f.mailbox.send({ to: "ses_a", message: "No" }, caller("a")), /yourself/);
  f.sessions.get("ses_b").time.archived = 123;
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: "No" }, caller("a")), /archived/);
  delete f.sessions.get("ses_b").time.archived;
  f.sessions.get("ses_b").location.directory = "/other-worktree";
  await assert.rejects(f.mailbox.send({ to: "ses_b", message: "No" }, caller("a")), /direct member/);
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

test("replacing membership removes a former child's messaging access", async () => {
  const f = fixture();
  await register(f);
  await f.mailbox.register({ children: ["ses_a"] }, caller("root"));
  await assert.rejects(f.mailbox.send({ to: "parent", message: "Hi" }, caller("b")), /register this child/);
});

test("plugin registers executable Code Mode tools against the V2 context", async () => {
  const f = fixture();
  const tools = new Map();
  f.ctx.tool = { async transform(callback) {
    callback({ namespace(value) { assert.equal(value.name, "swarm"); },
      add(tool) { tools.set(tool.name, tool); } });
  } };
  await plugin.setup(f.ctx);
  assert.deepEqual([...tools.keys()], ["register", "members", "send", "broadcast"]);
  for (const tool of tools.values()) assert.equal(tool.options.codemode, true);
  await tools.get("register").execute({ children: ["ses_a"] }, caller("root"));
  await tools.get("send").execute({ to: "parent", message: "Question" }, caller("a"));
  assert.equal(f.admitted[0].metadata.sender, "ses_a");
});
