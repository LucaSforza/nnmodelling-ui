// Only the duplicate-wait guard is local; messages cross plugin instances via
// native events and plugin storage, not JavaScript object identity.
const contexts = new WeakMap();
const defaultClock = { now: Date.now, setTimeout, clearTimeout };
const notificationKey = (id) => `wake/${id}`;
const observedKey = (id) => `wake-seen/${id}`;

export function createActivity(ctx, clock = defaultClock) {
  if (!contexts.has(ctx.storage)) contexts.set(ctx.storage, new Set());
  const waiting = contexts.get(ctx.storage);

  return {
    async admitted(to, sender, inboxID, principal) {
      try {
        await ctx.storage.set(notificationKey(to), { reason: "message", sender, inboxID, principal });
      } catch (error) {
        // Synthetic admission already succeeded; do not misreport it as failure.
        return error.message;
      }
    },

    async wait(input, context, prepare) {
      const seconds = input.timeout_seconds === undefined ? 270 : input.timeout_seconds;
      if (!Number.isInteger(seconds) || seconds < 1 || seconds > 270) {
        throw new Error("timeout_seconds must be an integer from 1 to 270");
      }
      if (typeof ctx.session.wait !== "function" || typeof ctx.event?.subscribe !== "function") {
        throw new Error("Runtime must expose native session.wait and event.subscribe for swarm.wait");
      }
      context.signal?.throwIfAborted();
      if (waiting.has(context.sessionID)) throw new Error("A swarm.wait is already active for this principal");

      const started = clock.now();
      const controller = new AbortController();
      const { signal } = controller;
      const queue = [];
      let children = [];
      let prepared;
      let processing = false;
      let settled = false;
      let resolve;
      let reject;
      const done = new Promise((yes, no) => { resolve = yes; reject = no; });
      const finish = (event) => {
        if (settled) return;
        settled = true;
        const { principal, ...result } = event;
        resolve({ ...result, children, elapsed_ms: clock.now() - started });
      };
      const fail = (error) => {
        if (settled) return;
        settled = true;
        reject(error);
      };
      const drain = async () => {
        if (!prepared || processing || settled) return;
        processing = true;
        try {
          while (queue.length && !settled) {
            const event = queue.shift();
            if (event.reason !== "message" || event.principal !== context.sessionID ||
                typeof event.inboxID !== "string") continue;
            const observed = await ctx.storage.get(observedKey(context.sessionID));
            signal.throwIfAborted();
            if (observed === event.inboxID || !await prepared.accepts(event.sender, signal)) continue;
            await ctx.storage.set(observedKey(context.sessionID), event.inboxID);
            signal.throwIfAborted();
            finish(event);
          }
        } finally { processing = false; }
      };
      const process = () => { void drain().catch((error) => { if (!signal.aborted) fail(error); }); };
      const snapshot = async () => {
        const notification = await ctx.storage.get(notificationKey(context.sessionID));
        signal.throwIfAborted();
        if (notification) queue.push(notification);
        process();
      };
      const abort = () => fail(context.signal.reason);
      waiting.add(context.sessionID);
      context.signal?.addEventListener("abort", abort, { once: true });
      const timer = clock.setTimeout(() => finish({ reason: "timeout" }), seconds * 1000);

      // Subscribe before the persisted snapshot. Consume/filter quickly, then
      // validate only relevant inbox admissions outside the shared event iterator.
      void (async () => {
        let first = true;
        try {
          for await (const event of ctx.event.subscribe({ signal })) {
            if (first) {
              first = false;
              void snapshot().catch((error) => { if (!signal.aborted) fail(error); });
            }
            const data = event.data;
            const metadata = data?.item?.payload?.metadata;
            if (event.type !== "session.inbox.enqueued" || data.sessionID !== context.sessionID ||
                data.item?.type !== "synthetic" || metadata?.source !== "nnmodelling.swarm-mailbox") continue;
            queue.push({ reason: "message", sender: metadata.sender,
              inboxID: data.inboxID, principal: metadata.swarm });
            process();
          }
          if (!signal.aborted) finish({ reason: "unavailable", observation: "events", error: "Native event stream ended" });
        } catch (error) {
          if (!signal.aborted) finish({ reason: "unavailable", observation: "events", error: error.message });
        }
      })();

      // All branches are observed, including requests aborted after another wins.
      void (async () => {
        prepared = await prepare(signal);
        signal.throwIfAborted();
        children = prepared.children;
        await snapshot();
        process();
        if (settled) return;
        for (const sessionID of children) {
          void Promise.resolve().then(() => ctx.session.wait({ sessionID }, { signal }))
            .then(() => finish({ reason: "idle", sessionID }), (error) => {
              if (!signal.aborted) finish({ reason: "unavailable", sessionID, error: error.message });
            });
        }
        await context.progress?.({ status: "Waiting for swarm activity" });
      })().catch((error) => { if (!signal.aborted) fail(error); });

      try {
        return await done;
      } finally {
        waiting.delete(context.sessionID);
        clock.clearTimeout(timer);
        context.signal?.removeEventListener("abort", abort);
        controller.abort(); // Cancels observers/subscriber, not the child agents.
      }
    },
  };
}
