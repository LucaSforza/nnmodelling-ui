import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { formatMarkdown, validateMarkdown } from '../tools/mermaid-check/mermaid-check.mjs';

test('formatter changes only Mermaid fence indentation and is stable', () => {
  const source = '# Note\n\n```mermaid\nsequenceDiagram\n  A->>B: hello\n```\n';
  const formatted = formatMarkdown(source);
  assert.equal(formatted, '# Note\n\n```mermaid\n  sequenceDiagram\n    A->>B: hello\n```\n');
  assert.equal(formatMarkdown(formatted), formatted);
});

test('validator reports malformed fences at their source line', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'mermaid-check-'));
  const file = path.join(directory, 'bad.md');
  try {
    const source = '# Header\n\n```mermaid\nsequenceDiagram\n A->>B hello\n```\n';
    await writeFile(file, source);
    const errors = await validateMarkdown(file, source);
    assert.equal(errors.length, 1);
    assert.match(errors[0], /bad\.md:3:/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
