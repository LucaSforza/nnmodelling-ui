// Session IDs and parent relationships come from OpenCode, not tool arguments.
// Share locks with fresh mailbox instances using the same plugin storage context.
const storageLocks = new WeakMap();

export function createMailbox(ctx) {
  const key = (rootID) => `members/${rootID}`;
  const reply = (data) => ({ content: JSON.stringify(data) });
  if (!storageLocks.has(ctx.storage)) storageLocks.set(ctx.storage, new Map());
  const locks = storageLocks.get(ctx.storage);

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

  async function sessions(sessionID) {
    const self = await ctx.session.get({ sessionID });
    const root = self.parentID
      ? await ctx.session.get({ sessionID: self.parentID }) : self;
    if (root.parentID || !sameCheckout(self, root)) {
      throw new Error("Mailbox requires a principal and direct children in one checkout");
    }
    if (self.time.archived || root.time.archived) throw new Error("Session is archived");
    return { self, root };
  }

  async function enroll(root, sessionID, signal) {
    await member(root, sessionID);
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
    const group = await sessions(context.sessionID);
    const ids = await enroll(group.root, group.self.id, context.signal);
    return { ...group, ids };
  }

  async function member(root, sessionID) {
    if (typeof sessionID !== "string" || !sessionID) throw new Error("Recipient must be a session ID");
    const session = sessionID === root.id ? root
      : await ctx.session.get({ sessionID });
    if (!sameCheckout(session, root) ||
        (session.id !== root.id && session.parentID !== root.id)) {
      throw new Error("Recipient is not a direct member of this checkout's swarm");
    }
    if (session.time.archived) throw new Error("Session is archived");
    return session;
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
    return { to: recipient.id, admitted: true, inboxID: admitted.id, delivery };
  }

  return {
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
