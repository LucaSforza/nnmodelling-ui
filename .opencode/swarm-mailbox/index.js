import { createMailbox } from "./mailbox.js";

const delivery = { type: "string", enum: ["steer", "queue"] };
const message = { type: "string", minLength: 1, maxLength: 16000 };
const schema = (properties, required) => ({
  type: "object", properties, required, additionalProperties: false,
});

export default {
  id: "nnmodelling.swarm-mailbox",
  async setup(ctx) {
    const mailbox = createMailbox(ctx);
    await ctx.tool.transform((editor) => {
      editor.namespace({ name: "swarm", description: "Initialize before child launch; direct children automatically join the local swarm mailbox" });
      for (const [name, description, input] of [
        ["init", "Principal: initialize communication BEFORE launching children; idempotent and preserves membership/exclusions", schema({}, [])],
        ["register", "Principal: explicitly replace membership; omitted enrolled children are excluded, listed children regain access. Not needed after launch",
          schema({ children: { type: "array", items: { type: "string" } } }, ["children"])],
        ["members", "Automatically enroll and discover your principal and currently enrolled peer session IDs", schema({}, [])],
        ["send", "Admit an actionable message to a sessionID or parent and schedule the recipient to resume",
          schema({ to: { type: "string" }, message, delivery }, ["to", "message"])],
        ["broadcast", "Message all other swarm members; report admission or failure for each recipient",
          schema({ message, delivery }, ["message"])],
      ]) {
        editor.add({ name, description, input,
          options: { namespace: "swarm", codemode: true },
          execute: mailbox[name] });
      }
    });
  },
};
