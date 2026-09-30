// Session IDs and parent relationships come from OpenCode, not tool arguments.
export function createMailbox(ctx) {
  const key = (rootID) => `members/${rootID}`;
  const reply = (data) => ({ content: JSON.stringify(data) });

  function sameCheckout(a, b) {
    return a.projectID === b.projectID &&
      a.location.directory === b.location.directory;
  }

  async function family(sessionID) {
    const self = await ctx.session.get({ sessionID });
    const root = self.parentID
      ? await ctx.session.get({ sessionID: self.parentID }) : self;
    if (root.parentID || !sameCheckout(self, root)) {
      throw new Error("Mailbox requires a principal and direct children in one checkout");
    }
    const ids = await ctx.storage.get(key(root.id)) ?? [];
    if (!Array.isArray(ids) || (self.id !== root.id && !ids.includes(self.id))) {
      throw new Error("Principal must register this child before mailbox use");
    }
    return { self, root, ids };
  }

  async function member(root, sessionID) {
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
    if (to !== group.root.id && !group.ids.includes(to)) {
      throw new Error("Recipient is not registered in this swarm");
    }
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
    async register(input, context) {
      const { self, root } = await family(context.sessionID);
      if (self.id !== root.id) throw new Error("Only the principal may register children");
      if (!Array.isArray(input.children) ||
          input.children.some((id) => typeof id !== "string")) {
        throw new Error("children must be an array of session IDs");
      }
      const ids = [...new Set(input.children)];
      for (const id of ids) {
        if (id === root.id) throw new Error("Principal is not a child");
        await member(root, id);
      }
      context.signal?.throwIfAborted();
      await ctx.storage.set(key(root.id), ids);
      return reply({ principal: root.id, children: ids });
    },

    async members(_input, context) {
      const group = await family(context.sessionID);
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
      const group = await family(context.sessionID);
      const to = input.to === "parent" ? group.root.id : input.to;
      return reply(await admit(group, to, input, context.signal));
    },

    async broadcast(input, context) {
      message(input);
      const group = await family(context.sessionID);
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
