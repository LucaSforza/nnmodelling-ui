// Session IDs and parent relationships come from OpenCode, not tool arguments.
import { createActivity } from "./activity.js";

// Share locks with fresh mailbox instances using the same plugin storage context.
const storageLocks = new WeakMap();

export function createMailbox(ctx, clock) {
  const key = (rootID) => `members/${rootID}`;
  const reply = (data) => ({ content: JSON.stringify(data) });
  if (!storageLocks.has(ctx.storage)) storageLocks.set(ctx.storage, new Map());
  const locks = storageLocks.get(ctx.storage);
  const activity = createActivity(ctx, clock);

  async function locked(rootID, action) {
    const previous = locks.get(rootID) ?? Promise.resolve();
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    locks.set(rootID, gate);
    await previous;
    try {
      return await action();
    } finally {
      release();
      if (locks.get(rootID) === gate) locks.delete(rootID);
    }
  }

  function validIDs(ids) {
    return Array.isArray(ids) && ids.every((id) => typeof id === "string" && id.length > 0);
  }

  function registry(value) {
    if (value === undefined || validIDs(value)) {
      throw new Error("Principal must call swarm.init before launching children or using the mailbox");
    }
    if (!value || value.version !== 1 || !validIDs(value.children) ||
        !validIDs(value.excluded)) throw new Error("Invalid persisted swarm registry");
    return value;
  }

  function sameCheckout(a, b) {
    return a.projectID === b.projectID &&
      a.location.directory === b.location.directory;
  }

  async function sessions(sessionID, signal) {
    const self = await ctx.session.get({ sessionID }, { signal });
    const root = self.parentID
      ? await ctx.session.get({ sessionID: self.parentID }, { signal }) : self;
    if (root.parentID || !sameCheckout(self, root)) {
      throw new Error("Mailbox requires a principal and direct children in one checkout");
    }
    if (self.time.archived || root.time.archived) throw new Error("Session is archived");
    return { self, root };
  }

  async function enroll(root, sessionID, signal) {
    await member(root, sessionID, signal);
    return locked(root.id, async () => {
      const state = registry(await ctx.storage.get(key(root.id)));
      if (state.excluded.includes(sessionID)) {
        throw new Error("Child is explicitly excluded; the principal must register it again to restore access");
      }
      signal?.throwIfAborted();
      if (sessionID !== root.id && !state.children.includes(sessionID)) {
        const next = { ...state, children: [...state.children, sessionID] };
        await ctx.storage.set(key(root.id), next);
        return next.children;
      }
      return state.children;
    });
  }

  async function family(context) {
    const group = await sessions(context.sessionID, context.signal);
    const ids = await enroll(group.root, group.self.id, context.signal);
    return { ...group, ids };
  }

  async function member(root, sessionID, signal) {
    if (typeof sessionID !== "string" || !sessionID) throw new Error("Recipient must be a session ID");
    const session = sessionID === root.id ? root
      : await ctx.session.get({ sessionID }, { signal });
    if (!sameCheckout(session, root) ||
        (session.id !== root.id && session.parentID !== root.id)) {
      throw new Error("Recipient is not a direct member of this checkout's swarm");
    }
    if (session.time.archived) throw new Error("Session is archived");
    return session;
  }

  async function watched(input, context, inspect = false) {
    const group = await family(context);
    if (group.self.id !== group.root.id) throw new Error("Only the principal may wait or inspect child status");
    const ids = input.children === undefined ? group.ids : input.children;
    if (!validIDs(ids)) throw new Error("children must be an array of session IDs");
    const children = [...new Set(ids)];
    for (const id of children) {
      if (id === group.root.id) throw new Error("Principal is not a child");
      if (inspect && group.ids.includes(id)) continue;
      await enroll(group.root, id, context.signal);
    }
    return { ...group, children };
  }

  function message(input) {
    if (typeof input.message !== "string" || !input.message.trim() ||
        input.message.length > 16000) {
      throw new Error("Message must contain 1 to 16000 characters");
    }
    const delivery = input.delivery ?? "steer";
    if (delivery !== "steer" && delivery !== "queue") {
      throw new Error("Delivery must be steer or queue");
    }
    return delivery;
  }

  async function admit(group, to, input, signal) {
    if (to === group.self.id) throw new Error("Cannot send to yourself");
    await enroll(group.root, to, signal);
    const recipient = await member(group.root, to);
    const delivery = message(input);
    signal?.throwIfAborted();
    const admitted = await ctx.session.synthetic({
      sessionID: recipient.id,
      text: `[Swarm message from ${group.self.id} (${group.self.agent ?? "agent"})]\n${input.message}`,
      description: "Swarm coordination message",
      metadata: { source: "nnmodelling.swarm-mailbox", sender: group.self.id,
        swarm: group.root.id },
      delivery,
      resume: true,
    }, { signal });
    const observationError = await activity.admitted(recipient.id, group.self.id, admitted.id, group.root.id);
    return { to: recipient.id, admitted: true, inboxID: admitted.id, delivery,
      observation_error: observationError };
  }

  return {
    async wait(input, context) {
      return reply(await activity.wait(input, context, async (signal) => {
        const group = await watched(input, { ...context, signal });
        return { children: group.children, async accepts(sender) {
          if (sender === group.root.id) return false;
          try {
            await enroll(group.root, sender, signal);
            return true;
          } catch (error) {
            signal.throwIfAborted();
            return false;
          }
        } };
      }));
    },

    async status(input, context) {
      const group = await watched(input, context, true);
      let active;
      let activityError;
      try {
        if (typeof ctx.session.active !== "function") {
          throw new Error("Native active-session listing is not exposed by this runtime's plugin context");
        }
        active = await ctx.session.active(undefined, { signal: context.signal });
      } catch (error) {
        context.signal?.throwIfAborted();
        activityError = error.message;
      }
      const children = await Promise.all(group.children.map(async (sessionID) => {
        try {
          const session = await member(group.root, sessionID, context.signal);
          const result = { sessionID, agent: session.agent, title: session.title,
            activity: active ? (active[sessionID] ? "running" : "idle") : "unknown",
            outcome: session.outcome, updated: session.time.updated, idle: session.time.idle };
          try {
            const messages = await ctx.session.context({ sessionID }, { signal: context.signal });
            const latest = [...messages].reverse().find((item) => item.type === "assistant" &&
              item.content.some((part) => part.type === "tool"));
            result.tools = (latest?.content ?? []).filter((part) => part.type === "tool")
              .map((part) => ({ name: part.name, status: part.state.status }));
          } catch (error) {
            context.signal?.throwIfAborted();
            result.context_error = error.message;
          }
          return result;
        } catch (error) {
          context.signal?.throwIfAborted();
          return { sessionID, activity: "unavailable", error: error.message };
        }
      }));
      context.signal?.throwIfAborted();
      return reply({ principal: group.root.id, observed_at: Date.now(), activity_error: activityError, children });
    },

    async init(_input, context) {
      const { self, root } = await sessions(context.sessionID);
      if (self.id !== root.id) throw new Error("Only the principal may initialize the swarm");
      return locked(root.id, async () => {
        const value = await ctx.storage.get(key(root.id));
        const state = value === undefined || validIDs(value)
          ? { version: 1, children: [...new Set(value ?? [])], excluded: [] }
          : registry(value);
        context.signal?.throwIfAborted();
        await ctx.storage.set(key(root.id), state);
        return reply({ principal: root.id, initialized: true, children: state.children });
      });
    },

    async register(input, context) {
      const { self, root } = await sessions(context.sessionID);
      if (self.id !== root.id) throw new Error("Only the principal may register children");
      if (!validIDs(input.children)) {
        throw new Error("children must be an array of session IDs");
      }
      const ids = [...new Set(input.children)];
      return locked(root.id, async () => {
        const state = registry(await ctx.storage.get(key(root.id)));
        for (const id of ids) {
          if (id === root.id) throw new Error("Principal is not a child");
          await member(root, id);
        }
        const excluded = [...new Set([...state.excluded, ...state.children])]
          .filter((id) => !ids.includes(id));
        context.signal?.throwIfAborted();
        await ctx.storage.set(key(root.id), { version: 1, children: ids, excluded });
        return reply({ principal: root.id, children: ids });
      });
    },

    async members(_input, context) {
      const group = await family(context);
      const sessions = await Promise.all([group.root.id, ...group.ids].map(async (id) => {
        try {
          const session = await member(group.root, id);
          return { sessionID: id, agent: session.agent, title: session.title,
            role: id === group.root.id ? "principal" : "worker" };
        } catch (error) {
          return { sessionID: id, unavailable: error.message };
        }
      }));
      return reply({ self: group.self.id, principal: group.root.id, members: sessions });
    },

    async send(input, context) {
      message(input);
      const group = await family(context);
      const to = input.to === "parent" ? group.root.id : input.to;
      return reply(await admit(group, to, input, context.signal));
    },

    async broadcast(input, context) {
      message(input);
      const group = await family(context);
      const recipients = [group.root.id, ...group.ids].filter((id) => id !== group.self.id);
      const results = await Promise.all(recipients.map(async (to) => {
        try {
          return await admit(group, to, input, context.signal);
        } catch (error) {
          return { to, admitted: false, error: error.message };
        }
      }));
      return reply({ results });
    },
  };
}
