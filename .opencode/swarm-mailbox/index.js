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
      editor.namespace({ name: "swarm", description: "Messages within a principal's registered native child swarm" });
      for (const [name, description, input] of [
        ["register", "Principal: replace the registry with the complete native direct-child sessionID list",
          schema({ children: { type: "array", items: { type: "string" } } }, ["children"])],
        ["members", "Discover your principal and registered peer session IDs", schema({}, [])],
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
