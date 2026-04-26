#!/usr/bin/env node

/**
 * Smoke tests for CLI flags and non-interactive project generation.
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import chalk from 'chalk';

console.log(chalk.blue.bold('\n🧪 Testing CLI Flags\n'));

const distCliPath = path.join(process.cwd(), 'dist', 'index.js');
const packageJsonPath = path.join(process.cwd(), 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8')) as { version: string };
const smokeProjectName = 'test-cli-flags-project';
const smokeProjectPath = path.join(process.cwd(), smokeProjectName);

let passed = 0;
let failed = 0;

function runCli(args: string[]) {
  return spawnSync(process.execPath, [distCliPath, ...args], {
    cwd: process.cwd(),
    encoding: 'utf-8'
  });
}

function assertCheck(condition: boolean, description: string, details?: string): void {
  if (condition) {
    console.log(chalk.green(`✓ ${description}`));
    passed++;
    return;
  }

  console.log(chalk.red(`✗ ${description}`));
  if (details) {
    console.log(chalk.gray(`  ${details}`));
  }
  failed++;
}

try {
  if (fs.existsSync(smokeProjectPath)) {
    fs.rmSync(smokeProjectPath, { recursive: true, force: true });
  }

  console.log(chalk.cyan('Test 1: --version'));
  const versionResult = runCli(['--version']);
  assertCheck(
    versionResult.status === 0 && versionResult.stdout.trim() === packageJson.version,
    'CLI version matches package.json',
    versionResult.stdout || versionResult.stderr
  );

  console.log(chalk.cyan('\nTest 2: Template list flag and subcommand'));
  const listFlagResult = runCli(['--list']);
  const listCommandResult = runCli(['list']);
  assertCheck(
    listFlagResult.status === 0 &&
      listFlagResult.stdout.includes('NANO') &&
      listFlagResult.stdout.includes('0.5M params'),
    '--list prints current template catalog',
    listFlagResult.stdout || listFlagResult.stderr
  );
  assertCheck(
    listCommandResult.status === 0 &&
      listCommandResult.stdout.includes('TINY') &&
      listCommandResult.stdout.includes('5M params'),
    '`list` subcommand prints template catalog',
    listCommandResult.stdout || listCommandResult.stderr
  );

  console.log(chalk.cyan('\nTest 3: --yes generates without prompts'));
  const createResult = runCli([
    smokeProjectName,
    '-y',
    '--template',
    'nano',
    '--tokenizer',
    'unigram',
    '--skip-install'
  ]);
  assertCheck(
    createResult.status === 0 && fs.existsSync(smokeProjectPath),
    'Non-interactive create succeeds',
    createResult.stdout || createResult.stderr
  );

  console.log(chalk.cyan('\nTest 4: --force overwrites existing project'));
  const forceResult = runCli([
    smokeProjectName,
    '-y',
    '--template',
    'tiny',
    '--tokenizer',
    'bpe',
    '--skip-install',
    '--force'
  ]);
  assertCheck(
    forceResult.status === 0,
    'Forced overwrite succeeds',
    forceResult.stdout || forceResult.stderr
  );

  const configPath = path.join(smokeProjectPath, 'llm.config.js');
  const configContent = fs.readFileSync(configPath, 'utf-8');
  assertCheck(
    configContent.includes("size: 'tiny'") && configContent.includes("type: 'bpe'"),
    'Overwrite regenerated project files with new settings'
  );

  console.log(chalk.yellow(`\nResults: ${passed} passed, ${failed} failed`));

  if (failed > 0) {
    process.exit(1);
  }
} catch (error) {
  console.error(chalk.red('\n❌ CLI smoke test failed:'), error instanceof Error ? error.message : String(error));
  process.exit(1);
} finally {
  if (fs.existsSync(smokeProjectPath)) {
    fs.rmSync(smokeProjectPath, { recursive: true, force: true });
  }
}
