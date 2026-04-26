#!/usr/bin/env node

import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';

function getCompiledTests(distDir: string): string[] {
  return fs
    .readdirSync(distDir)
    .filter(file => /^test-.*\.js$/.test(file))
    .sort();
}

function run(): void {
  const distDir = path.join(__dirname);
  const tests = getCompiledTests(distDir);

  for (const test of tests) {
    const result = spawnSync(process.execPath, [path.join(distDir, test)], {
      stdio: 'inherit'
    });

    if (result.status !== 0) {
      process.exit(result.status ?? 1);
    }
  }
}

run();
