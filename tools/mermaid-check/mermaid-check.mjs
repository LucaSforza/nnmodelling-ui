#!/usr/bin/env node
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('');
globalThis.window = window;
globalThis.document = window.document;
const { default: mermaid } = await import('mermaid');

mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });

const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../docs/knowledge');

export function formatMermaid(content) {
  const lines = content.split(/\r?\n/);
  const indents = lines.filter((line) => line.trim()).map((line) => line.match(/^\s*/)[0].length);
  const minimumIndent = indents.length ? Math.min(...indents) : 0;
  return lines
    .map((line) => line.trim() ? `  ${line.slice(minimumIndent).trimEnd()}` : '')
    .join('\n')
    .replace(/\n+$/, '');
}

export function mermaidBlocks(markdown) {
  const blocks = [];
  const lines = markdown.split(/\r?\n/);
  let open = null;
  let body = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (open === null) {
      if (/^\s*```mermaid\s*$/i.test(line)) {
        open = index + 1;
        body = [];
      }
    } else if (/^\s*```\s*$/.test(line)) {
      blocks.push({ line: open, body: body.join('\n') });
      open = null;
    } else {
      body.push(line);
    }
  }
  if (open !== null) blocks.push({ line: open, body: null });
  return blocks;
}

export function formatMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  let inside = false;
  let start = -1;
  for (let index = 0; index < lines.length; index += 1) {
    if (!inside && /^\s*```mermaid\s*$/i.test(lines[index])) {
      inside = true;
      start = index + 1;
    } else if (inside && /^\s*```\s*$/.test(lines[index])) {
      const block = formatMermaid(lines.slice(start, index).join('\n'));
      lines.splice(start, index - start, ...block.split('\n'));
      index = start + block.split('\n').length;
      inside = false;
    }
  }
  return `${lines.join('\n').replace(/\n*$/, '')}\n`;
}

export async function validateMarkdown(file, markdown) {
  const errors = [];
  for (const block of mermaidBlocks(markdown)) {
    if (block.body === null) {
      errors.push(`${file}:${block.line}: unclosed Mermaid fence`);
      continue;
    }
    try {
      await mermaid.parse(block.body);
    } catch (error) {
      const message = String(error?.message ?? error).split('\n').slice(0, 3).join(' ');
      errors.push(`${file}:${block.line}: ${message}`);
    }
  }
  return errors;
}

async function markdownFiles(root) {
  const result = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const candidate = path.join(root, entry.name);
    if (entry.isDirectory()) result.push(...await markdownFiles(candidate));
    else if (entry.isFile() && entry.name.endsWith('.md')) result.push(candidate);
  }
  return result.sort();
}

async function main(args) {
  const mode = args[0];
  if (!['--check', '--write'].includes(mode)) {
    throw new Error('usage: node mermaid-check.mjs --check|--write [markdown files...]');
  }
  const files = args.slice(1).length
    ? args.slice(1).map((file) => path.resolve(file))
    : await markdownFiles(defaultRoot);
  const errors = [];
  let diagrams = 0;
  for (const file of files) {
    let content = await readFile(file, 'utf8');
    if (mode === '--write') {
      const formatted = formatMarkdown(content);
      if (formatted !== content) await writeFile(file, formatted, 'utf8');
      content = formatted;
    }
    diagrams += mermaidBlocks(content).filter((block) => block.body !== null).length;
    errors.push(...await validateMarkdown(file, content));
  }
  if (errors.length) {
    for (const error of errors) console.error(error);
    console.error(`Mermaid validation failed: ${errors.length} error(s) in ${files.length} file(s).`);
    process.exitCode = 1;
    return;
  }
  console.log(`Mermaid validation passed: ${diagrams} diagram(s) in ${files.length} file(s).`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.message ?? error);
    process.exitCode = 2;
  });
}
