#!/usr/bin/env node

/**
 * Smoke tests for generated BERT and T5 custom scaffolds.
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import chalk from 'chalk';
import { ScaffolderEngine } from './scaffolder';
import { TemplateManager } from './template-manager';
import { ProjectConfig } from './prompts';
import { Template } from './types/template';

console.log(chalk.blue.bold('\n🧪 Testing Multi-Architecture Scaffolds\n'));

const projects = {
  bert: path.join(process.cwd(), 'test-bert-architecture'),
  t5: path.join(process.cwd(), 'test-t5-architecture')
};

let passed = 0;
let failed = 0;

function check(condition: boolean, description: string, details?: string): void {
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

async function scaffoldProject(projectName: string, projectPath: string, modelType: 'bert' | 't5', tokenizer: ProjectConfig['tokenizer']): Promise<void> {
  const manager = new TemplateManager();
  const template = JSON.parse(JSON.stringify(manager.getTemplate('custom'))) as Template;
  template.config.model.type = modelType;

  const config: ProjectConfig = {
    projectName,
    template: 'custom',
    tokenizer,
    plugins: [],
    skipInstall: true
  };

  const scaffolder = new ScaffolderEngine(projectPath);
  await scaffolder.createProjectStructure(config, template);
  await scaffolder.copyTemplateFiles(config, template);
}

async function run(): Promise<number> {
  try {
    for (const projectPath of Object.values(projects)) {
      if (fs.existsSync(projectPath)) {
        fs.rmSync(projectPath, { recursive: true, force: true });
      }
    }

    console.log(chalk.cyan('Test 1: Generate BERT and T5 custom projects'));
    await scaffoldProject('test-bert-architecture', projects.bert, 'bert', 'wordpiece');
    await scaffoldProject('test-t5-architecture', projects.t5, 't5', 'unigram');

    check(fs.existsSync(path.join(projects.bert, 'models/architectures/bert.py')), 'BERT scaffold includes bert.py');
    check(fs.existsSync(path.join(projects.t5, 'models/architectures/t5.py')), 'T5 scaffold includes t5.py');

    console.log(chalk.cyan('\nTest 2: Generated Python compiles'));
    const compileResult = spawnSync('python', ['-m', 'compileall', '-q', projects.bert, projects.t5], {
      cwd: process.cwd(),
      encoding: 'utf-8'
    });

    check(
      compileResult.status === 0,
      'compileall passes for generated BERT/T5 scaffolds',
      compileResult.stderr || compileResult.stdout
    );

    console.log(chalk.yellow(`\nResults: ${passed} passed, ${failed} failed`));
    return failed === 0 ? 0 : 1;
  } catch (error) {
    console.error(chalk.red('\n❌ Architecture smoke test failed:'), error instanceof Error ? error.message : String(error));
    return 1;
  } finally {
    for (const projectPath of Object.values(projects)) {
      if (fs.existsSync(projectPath)) {
        fs.rmSync(projectPath, { recursive: true, force: true });
      }
    }
  }
}

run().then(code => process.exit(code));
