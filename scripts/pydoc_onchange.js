#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');

function deriveModuleNameFromFile(filePath) {
  const absolutePath = path.resolve(filePath);
  const projectRoot = path.resolve(__dirname, '..');
  const srcRoot = path.join(projectRoot, 'src') + path.sep;
  if (!absolutePath.startsWith(srcRoot)) return '';

  let relativePath = absolutePath.slice(srcRoot.length);
  if (relativePath.endsWith('.py')) relativePath = relativePath.slice(0, -3);
  const parts = relativePath.split(path.sep).filter(Boolean);
  if (parts.length === 0) return '';
  if (parts[parts.length - 1] === '__init__') parts.pop();
  return parts.join('.');
}

function runPydocMarkdownForModule(moduleName) {
  const args = ['run', 'pydoc-markdown', 'pydoc-markdown.yaml', '--module', moduleName];
  const proc = spawn('uv', args, { stdio: 'inherit' });
  proc.on('exit', code => process.exit(code ?? 0));
}

const changedFilePath = process.argv[2];
if (!changedFilePath) {
  process.stderr.write('Usage: node scripts/pydoc_onchange.js <file>\n');
  process.exit(2);
}

const moduleName = deriveModuleNameFromFile(changedFilePath);
if (!moduleName) {
  process.stdout.write(`[pydoc-onchange] Ignored: ${changedFilePath}\n`);
  process.exit(0);
}

runPydocMarkdownForModule(moduleName);

