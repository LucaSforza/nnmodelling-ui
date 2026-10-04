import { readdir, readFile } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

let mermaid;
try {
  const { JSDOM } = await import("jsdom");
  const dom = new JSDOM("");
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  ({ default: mermaid } = await import("mermaid"));
} catch (error) {
  console.error("Unable to load Mermaid check dependencies. Run `just check-mermaid` to install them.");
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
mermaid.initialize({ startOnLoad: false });

const repositoryRoot = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const knowledgeRoot = resolve(repositoryRoot, "docs/knowledge");
const markdownFiles = await findMarkdown(knowledgeRoot);
const mermaidFence = /^[ \t]{0,3}```mermaid[ \t]*\r?\n([\s\S]*?)^[ \t]{0,3}```[ \t]*$/gm;
const failures = [];
let diagramCount = 0;

for (const file of markdownFiles) {
  const markdown = await readFile(file, "utf8");
  for (const match of markdown.matchAll(mermaidFence)) {
    diagramCount += 1;
    const documentLine = markdown.slice(0, match.index).split(/\r?\n/).length;
    try {
      await mermaid.parse(match[1]);
    } catch (error) {
      const diagramLine = error?.hash?.loc?.first_line;
      const location = diagramLine ? ` (diagram line ${diagramLine})` : "";
      const message = String(error?.message ?? error).split("\n", 1)[0];
      const sourceLine = diagramLine ? match[1].split(/\r?\n/)[diagramLine - 1]?.trim() : "";
      const excerpt = sourceLine ? ` | ${sourceLine}` : "";
      failures.push(`${relative(repositoryRoot, file).split(sep).join("/")}:${documentLine}${location}: ${message}${excerpt}`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  console.error(`\n${failures.length} of ${diagramCount} Mermaid diagrams failed to parse.`);
  process.exit(1);
}

console.log(`Parsed ${diagramCount} Mermaid diagrams in ${markdownFiles.length} Knowledge Base Markdown files.`);

async function findMarkdown(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) return findMarkdown(path);
    return entry.isFile() && entry.name.endsWith(".md") ? [path] : [];
  }));
  return nested.flat().sort();
}
