# Update Log - Forked Changes

> **Fork Author**: @shubhampardule  
> **Original Repo**: [theaniketgiri/create-llm](https://github.com/theaniketgiri/create-llm)  
> **Date**: April 2026

This document tracks the functional changes made in the fork. It includes the original feature/fix work and the later verification pass that fixed issues discovered while running the repo end to end.

---

## Table of Contents

1. [Fix #1 - Dynamic Version](#fix-1---dynamic-version)
2. [Fix #2 - `--yes` Flag Skips All Prompts](#fix-2---yes-flag-skips-all-prompts)
3. [Fix #3 - SynthexAI Plugin Missing from Prompt](#fix-3---synthexai-plugin-missing-from-prompt)
4. [Fix #4 - BERT/T5 Architecture Implementation](#fix-4---bertt5-architecture-implementation)
5. [Fix #5 - `--list` Flag / `list` Subcommand](#fix-5---list-flag--list-subcommand)
6. [Fix #6 - `--force` Flag for Overwrite](#fix-6---force-flag-for-overwrite)
7. [Fix #7 - `npm test` No Longer Hangs Interactively](#fix-7---npm-test-no-longer-hangs-interactively)
8. [Fix #8 - Windows-safe Overwrite and File Write Retries](#fix-8---windows-safe-overwrite-and-file-write-retries)
9. [Fix #9 - BERT Template Newline Escaping](#fix-9---bert-template-newline-escaping)
10. [Improvement #1 - T5 Causal Mask Performance](#improvement-1---t5-causal-mask-performance)
11. [Improvement #2 - BERT `generate()` Guard](#improvement-2---bert-generate-guard)
12. [Improvement #3 - Operator Precedence Fix](#improvement-3---operator-precedence-fix)
13. [Improvement #4 - Conditional Architecture File Generation](#improvement-4---conditional-architecture-file-generation)
14. [Improvement #5 - Model-Type-Aware Trainer](#improvement-5---model-type-aware-trainer)
15. [Improvement #6 - Consistent Parameter Count Formatting](#improvement-6---consistent-parameter-count-formatting)
16. [Improvement #7 - CLI Smoke Coverage](#improvement-7---cli-smoke-coverage)
17. [Improvement #8 - BERT/T5 Compile Smoke Coverage](#improvement-8---bertt5-compile-smoke-coverage)
18. [Improvement #9 - Test Suite Synchronization](#improvement-9---test-suite-synchronization)
19. [Improvement #10 - Windows UTF-8 Console Support](#improvement-10---windows-utf-8-console-support)
20. [Improvement #11 - Optional Architecture Imports for GPT-only Projects](#improvement-11---optional-architecture-imports-for-gpt-only-projects)
21. [Improvement #12 - Dataloader Robustness for Small and Variable Batches](#improvement-12---dataloader-robustness-for-small-and-variable-batches)
22. [Improvement #13 - Comparison Script Evaluation Accuracy](#improvement-13---comparison-script-evaluation-accuracy)
23. [Verification Addendum - Full Runtime Smoke Pass](#verification-addendum---full-runtime-smoke-pass)

---

## Fix #1 - Dynamic Version

**File**: `src/index.ts`

**Problem**: The CLI version was hardcoded as `0.1.0` while `package.json` was at `2.2.3`. `create-llm --version` returned the wrong value, and version bumps required manual sync.

**Before**:
```typescript
program.version('0.1.0')
```

**After**:
```typescript
const { version } = require('../package.json') as { version: string };

program.version(version)
```

**Why**: `package.json` is now the single source of truth for CLI versioning.

---

## Fix #2 - `--yes` Flag Skips All Prompts

**Files**: `src/index.ts`, `src/prompts.ts`

**Problem**: `-y, --yes` only skipped the final confirmation prompt. It still asked for template, tokenizer, and plugins, so it was not actually automation-friendly.

**What changed**:

- `runInteractiveFlow()` now accepts a `skipAll` flag.
- `--yes` uses defaults without prompting.
- CLI flags passed alongside `--yes` still override the defaults.
- The selected values are printed so the user can still see what was chosen.

**Example behavior**:

| Command | Result |
|---------|--------|
| `create-llm` | Fully interactive |
| `create-llm my-app -y` | No prompts, defaults used |
| `create-llm -y -t nano` | No prompts, `nano` template used |
| `create-llm -y -t nano --tokenizer wordpiece` | No prompts, all explicit flags respected |

**Why**: This makes the CLI usable in CI/CD and scripted setups.

---

## Fix #3 - SynthexAI Plugin Missing from Prompt

**File**: `src/prompts.ts`

**Problem**: The project already had SynthexAI post-install guidance, but the interactive plugin selection never offered `synthex` as a choice.

**What changed**:

- Added `synthex` to the interactive plugin checklist.

**Why**: The prompt, generated config, and next-step guidance are now consistent.

---

## Fix #4 - BERT/T5 Architecture Implementation

**Files**: `src/python-templates.ts`, `src/scaffolder.ts`, `src/template-manager.ts`, `src/config-generator.ts`

**Problem**: The config and validators already accepted `gpt`, `bert`, and `t5`, but only GPT architecture code actually existed. Custom templates using `bert` or `t5` would validate successfully and then fail at runtime.

**What was added**:

### BERT support

- `BERTConfig`
- `BERTEmbeddings`
- Bidirectional self-attention
- Encoder stack and `[CLS]` pooling
- `BERTForMaskedLM`
- `create_bert_model()`

### T5 support

- `T5Config`
- RMSNorm
- Relative position bias
- Encoder-decoder attention blocks
- Gated feed-forward network
- `T5Model`
- `create_t5_model()`

### Runtime routing

`load_model_from_config()` now dispatches by `model.type`:

```python
if model_type == 'bert':
    from .architectures.bert import create_bert_model
    return create_bert_model(model_config)
elif model_type == 't5':
    from .architectures.t5 import create_t5_model
    return create_t5_model(model_config)
```

### Scaffolding/export updates

- `scaffolder.ts` now generates the relevant architecture files.
- `__init__.py` exports were updated for BERT and T5 classes/factories.

**Why**: The accepted config values now map to real, generated, runnable code.

---

## Fix #5 - `--list` Flag / `list` Subcommand

**File**: `src/index.ts`

**Problem**: There was no fast way to discover templates from the CLI without entering interactive mode or reading the source.

**What changed**:

- Added `create-llm --list` / `-l`
- Added `create-llm list`
- Added `displayTemplateList()` to print template summary rows

**Why**: This makes the CLI more discoverable and scriptable.

---

## Fix #6 - `--force` Flag for Overwrite

**File**: `src/index.ts`

**Problem**: If the target directory already existed, project generation failed immediately unless the directory was removed manually first.

**Before**:
```typescript
if (fs.existsSync(projectPath)) {
  console.error(chalk.red(`\n❌ Directory "${config.projectName}" already exists`));
  process.exit(1);
}
```

**After**:
```typescript
.option('-f, --force', 'Overwrite the target directory if it already exists')

if (fs.existsSync(projectPath)) {
  if (options.force) {
    fs.rmSync(projectPath, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100
    });
  } else {
    process.exit(1);
  }
}
```

**Why**: The CLI now supports fully headless overwrite flows such as `create-llm my-app -y --force`.

---

## Fix #7 - `npm test` No Longer Hangs Interactively

**Files**: `package.json`, `src/run-tests.ts`

**Problem**: The old `npm test` script launched the main CLI:

```json
"test": "node dist/index.js test-project --template nano"
```

That path still prompted for tokenizer/plugin input, so `npm test` hung waiting for terminal interaction instead of running the test suite.

**What changed**:

- `npm test` now builds the project and runs a dedicated test runner:

```json
"test": "npm run build && node dist/run-tests.js"
```

- `src/run-tests.ts` discovers all compiled `dist/test-*.js` files and runs them sequentially.

**Why**: `npm test` is now a real, non-interactive verification command.

---

## Fix #8 - Windows-safe Overwrite and File Write Retries

**Files**: `src/index.ts`, `src/scaffolder.ts`

**Problem**: During a real overwrite smoke test on Windows, `--force` sometimes failed with transient filesystem errors such as `EBUSY` after removing the existing directory. The delete completed, but very short-lived locks could still cause immediate rewrites to fail.

**What changed**:

- `fs.rmSync()` now uses `maxRetries` and `retryDelay` during forced overwrite.
- `ScaffolderEngine` now writes files through `writeFileWithRetry()`.
- File writes retry on transient `EBUSY` / `EPERM` errors before failing.

**Why**: This makes overwrite behavior more reliable on Windows, especially during fast delete-and-recreate flows.

---

## Fix #9 - BERT Template Newline Escaping

**File**: `src/python-templates.ts`

**Problem**: The generated `bert.py` template used literal newlines inside a Python string in the `NotImplementedError` message for `generate()`. That produced an unterminated string literal in scaffolded BERT projects.

**Broken generated output**:
```python
raise NotImplementedError(
    "BERT is an encoder-only ... generation.
"
)
```

**What changed**:

- The TypeScript template now uses escaped `\\n` inside the generated Python string:

```python
raise NotImplementedError(
    "BERT is an encoder-only ... generation.\n"
    "Use model(input_ids, labels=masked_labels) ...\n"
    "For text generation, use a GPT or T5 model."
)
```

**Why**: Scaffolded BERT and T5 custom projects now compile cleanly with `python -m compileall`.

---

## Improvement #1 - T5 Causal Mask Performance

**File**: `src/python-templates.ts`

**Problem**: The T5 decoder created a fresh causal mask tensor on every forward pass.

**What changed**:

- The mask is now registered once as a buffer and sliced per sequence length.

**Why**: Less allocation overhead and behavior consistent with the GPT implementation.

---

## Improvement #2 - BERT `generate()` Guard

**File**: `src/python-templates.ts`

**Problem**: BERT is encoder-only, but surrounding code paths could still call `model.generate()`, which would previously fail with an unhelpful `AttributeError`.

**What changed**:

- `BERTForMaskedLM.generate()` now raises a clear `NotImplementedError` explaining that BERT is not an autoregressive text-generation model.

**Why**: Users now get an actionable explanation instead of a confusing missing-method failure.

---

## Improvement #3 - Operator Precedence Fix

**File**: `src/prompts.ts`

**Problem**: This expression used logical OR:

```typescript
const tokenizer = initialTokenizer as 'bpe' | 'wordpiece' | 'unigram' || await this.promptTokenizer();
```

That is too broad because `||` falls through on any falsy value.

**What changed**:

- Switched to nullish coalescing and explicit grouping:

```typescript
const tokenizer =
  (initialTokenizer as 'bpe' | 'wordpiece' | 'unigram') ?? await this.promptTokenizer();
```

- The same nullish behavior is now used in the `skipAll` / `--yes` path too.

**Why**: Only `null` / `undefined` trigger the fallback, which is the intended behavior.

---

## Improvement #4 - Conditional Architecture File Generation

**File**: `src/scaffolder.ts`

**Problem**: `bert.py` and `t5.py` were being generated even for GPT-only projects.

**What changed**:

- GPT projects only get GPT architecture files.
- Custom projects include all architectures because they may be repurposed.
- BERT/T5 template types generate their matching architecture files.

**Why**: Generated projects are cleaner and less cluttered.

---

## Improvement #5 - Model-Type-Aware Trainer

**File**: `src/python-trainer-templates.ts`

**Problem**: The generic trainer path assumed GPT-style inputs, but T5 requires `decoder_input_ids`.

**What changed**:

- Trainer now detects model family.
- T5 batches are adapted by right-shifting labels into `decoder_input_ids`.
- The same logic is used in training and evaluation.

**Why**: T5 training now works with the generated trainer instead of failing on missing decoder inputs.

---

## Improvement #6 - Consistent Parameter Count Formatting

**Files**: `src/formatters.ts`, `src/index.ts`, `src/prompts.ts`, `src/template-manager.ts`, `src/config-generator.ts`, `src/scaffolder.ts`

**Problem**: Parameter counts were formatted differently in different places. In practice this caused the `nano` preset (500K parameters) to be rounded up and shown as `1M` in some CLI output and docs.

**What changed**:

- Added a shared `formatParameterCount()` helper in `src/formatters.ts`.
- CLI lists, prompt summaries, template summaries, config tips, and generated docs now use the same formatter.
- `nano` is now displayed correctly as `0.5M`.

**Why**: Output is now consistent and accurate across the CLI and generated artifacts.

---

## Improvement #7 - CLI Smoke Coverage

**File**: `src/test-cli-flags.ts`

**What changed**:

- Added smoke coverage for:
  - `--version`
  - `--list`
  - `list`
  - `-y`
  - `--force`

- The test verifies both command success and generated config values after overwrite.

**Why**: The most important non-interactive CLI paths now have automated coverage.

---

## Improvement #8 - BERT/T5 Compile Smoke Coverage

**File**: `src/test-architectures.ts`

**What changed**:

- Added a test that scaffolds custom BERT and T5 projects.
- Runs `python -m compileall -q` against the generated projects.
- Verifies that the relevant architecture files actually exist.

**Why**: This catches template-level Python syntax issues that TypeScript tests alone cannot see.

---

## Improvement #9 - Test Suite Synchronization

**Files**:

- `src/test-config-generator.ts`
- `src/test-python-files.ts`
- `src/test-scaffolder.ts`
- `src/test-chat-interface.ts`
- `src/test-e2e.ts`
- `src/test-validation.ts`

**Problem**: Several tests were stale. They still expected:

- 4 templates instead of 5
- Older `tiny` model values
- Older sample/README text
- Older checkpoint validation wording
- Validation failures caused by missing temp template files instead of actual invalid config checks

**What changed**:

- Tests now derive expectations from the real template data where appropriate.
- `nano` is included in validation temp fixtures.
- Assertions were updated to match current generated content.

**Why**: The suite now validates the current repo instead of older assumptions.

---

## Improvement #10 - Windows UTF-8 Console Support

**Files**: `src/python-tokenizer-templates.ts`, `src/scaffolder.ts`

**Problem**: During a real Windows runtime pass, several generated Python scripts completed their core work and then failed while printing Unicode characters such as checkmarks and emojis. The default console encoding could not encode those characters reliably.

**What changed**:

- Added UTF-8 `stdout` / `stderr` setup to generated tokenizer scripts.
- Added the same Windows console guard to generated data prep, evaluation, generation, chat, chat interface, deploy, and compare scripts.

**Why**: Generated projects now finish cleanly on Windows instead of failing at the final logging/output step.

---

## Improvement #11 - Optional Architecture Imports for GPT-only Projects

**File**: `src/python-templates.ts`

**Problem**: After conditional file generation was added, GPT-only scaffolds correctly stopped generating `bert.py` and `t5.py`, but package exports still imported those modules unconditionally. A generated GPT project could fail at startup with `ModuleNotFoundError`.

**What changed**:

- `models/architectures/__init__.py` now always exports GPT symbols directly.
- BERT and T5 imports are added only when those modules actually exist.
- `models/__init__.py` now re-exports the dynamically available architecture symbols instead of hardcoding all model families.

**Why**: GPT-only scaffolds no longer depend on architecture files that were intentionally omitted.

---

## Improvement #12 - Dataloader Robustness for Small and Variable Batches

**File**: `src/python-dataset-templates.ts`

**Problem**: The shared `create_dataloader()` helper always used `drop_last=True`, which silently dropped tiny validation sets such as a single-example `val.pt`. Its collate path also assumed tensors in each batch already had identical lengths, which broke generated tests using variable-length examples.

**What changed**:

- Added a `drop_last` argument and changed the default behavior to keep incomplete batches.
- Updated the collate function to truncate first and then pad dynamically with `pad_sequence`.
- Padding is now applied with the correct values for each tensor type:
  - `input_ids`: `0`
  - `attention_mask`: `0`
  - `labels`: `-100`

**Why**: Small sample projects now evaluate correctly, and generated projects handle variable-length batches without crashing.

---

## Improvement #13 - Comparison Script Evaluation Accuracy

**File**: `src/scaffolder.ts`

**Problem**: The generated `compare.py` script used a hardcoded validation `max_length=512` even when the trained model used a smaller context window. It also merged checkpoint metadata into the same `loss` field used for computed evaluation results, so the summary table could show checkpoint training loss instead of actual validation loss.

**What changed**:

- `compare.py` now uses `model.config.max_length` when preparing validation batches.
- Checkpoint metadata is now stored under `checkpoint_loss` instead of overwriting the evaluated `loss`.
- The comparison table now reports the actual evaluated loss, perplexity, and token counts.

**Why**: Comparison results are now accurate and no longer emit avoidable truncation warnings on correctly processed validation data.

---

## Verification Addendum - Full Runtime Smoke Pass

After the earlier template/test fixes, the fork was also re-validated by generating and exercising a real project locally on Windows.

**Generated project used**:

- `runtime-check-gpt`
- Template: `nano`
- Tokenizer: `bpe`

**What was executed successfully**:

- Fresh scaffold generation with `-y --skip-install --force`
- `pip install -r requirements.txt`
- `python tokenizer/train.py`
- `python data/prepare.py`
- `python training/train.py --device cpu --max-steps ...`
- `python evaluation/evaluate.py`
- `python evaluation/generate.py`
- `python compare.py`
- `python chat.py`
- `python chat_interface.py --help`
- `python deploy.py --help`
- `python -m pytest -q`

**What this uncovered and fixed**:

- Windows Unicode console failures in generated scripts
- GPT-only import regressions after conditional architecture generation
- Empty validation loaders caused by unconditional `drop_last=True`
- Variable-length collate failures in generated tests
- Incorrect loss/context handling in `compare.py`

**Why this matters**: The fork has now been checked both by the repo-level TypeScript/CLI suite and by a real generated Python project running its own runtime and pytest flows.

---

## Files Changed Summary

| File | What changed |
|------|--------------|
| `package.json` | `npm test` now runs the real compiled test suite |
| `src/index.ts` | Dynamic version, template listing, force overwrite, Windows-safe delete retries, shared parameter formatting |
| `src/prompts.ts` | `--yes` skip-all flow, SynthexAI plugin prompt, nullish tokenizer fallback, shared parameter formatting |
| `src/python-templates.ts` | BERT/T5 implementations, T5 mask buffer, BERT generate guard, BERT newline escape fix, dynamic optional architecture exports |
| `src/python-tokenizer-templates.ts` | Windows UTF-8 console handling for generated tokenizer/data scripts |
| `src/python-dataset-templates.ts` | Safer dataloader defaults and variable-length batch padding |
| `src/python-trainer-templates.ts` | Model-type-aware batch preparation for T5 |
| `src/scaffolder.ts` | Conditional architecture generation, write retries for transient Windows locks, Windows UTF-8 script guards, compare-script correctness fixes, dynamic README/sample formatting |
| `src/template-manager.ts` | Shared parameter formatting in summaries |
| `src/config-generator.ts` | Shared parameter formatting in generated tips/comments |
| `src/formatters.ts` | New shared parameter count formatter |
| `src/run-tests.ts` | New sequential compiled test runner |
| `src/test-cli-flags.ts` | New CLI smoke coverage |
| `src/test-architectures.ts` | New BERT/T5 scaffold compile smoke coverage |
| `src/test-config-generator.ts` | Synced with live template values |
| `src/test-python-files.ts` | Synced with current sample data and template sizing |
| `src/test-scaffolder.ts` | Synced with current README output |
| `src/test-chat-interface.ts` | Synced with current checkpoint validation code path |
| `src/test-e2e.ts` | Updated for 5-template catalog |
| `src/test-validation.ts` | Fixed temp fixture setup so validation failures are real validation failures |

---

## Verification Status

After the verification pass and the fixes above:

- `npm run build` passes
- `npm test` passes
- CLI smoke tests for `--version`, `--list`, `list`, `-y`, and `--force` pass
- Generated custom BERT and T5 projects compile with `python -m compileall`
- A fresh generated GPT project can:
  - train a tokenizer
  - preprocess data
  - run short CPU training
  - evaluate a checkpoint
  - generate text
  - compare checkpoints
  - launch chat
- The generated project test suite passes with `python -m pytest -q` (`28/28`)

This means the fork now verifies cleanly in the current local environment for both the TypeScript CLI and a real generated Python project.
