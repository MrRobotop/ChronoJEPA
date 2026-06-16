# prompts.md

A sequenced library of prompts for building ChronoJEPA with Claude Code. Read `prompts-readme.md` first: it explains the structure, why each prompt is shaped this way, and how to run them. Run the phases in order, one phase per session, and commit between phases so each prompt starts from a clean, known state.

Every prompt below uses the same skeleton: a role, the context and motivation, the task, hard constraints, explicit success criteria, and the output format. Fill the bracketed placeholders before sending.

---

## Reusable preamble (paste at the top of any session, or rely on CLAUDE.md)

```
<role>
You are a senior ML engineer building a self-supervised representation learning
library for financial and multivariate time series. You value correctness,
small surgical changes, and verifiable results over speed.
</role>

<operating_rules>
Before writing code, state your assumptions and a short plan, and ask one
clarifying question if the task is ambiguous. Keep implementations minimal: no
speculative abstractions, no unrequested features. Change only what the task
requires. Do not declare a task done until the stated success criteria pass when
you run them. Write all comments and docs without em dashes.
</operating_rules>
```

---

## Phase 0: Plan and scaffold (use this as the first context window)

```
<context>
We are starting ChronoJEPA, a heuristics-free SSL library for time series built on
the SIGReg objective from LeJEPA (arXiv:2511.08544). The research goal is to fix the
time-axis "ID vector" collapse described in LeJEPA issue #27 by comparing three SIGReg
placements: pooled, dual (within-sequence plus between-sample), and structured. This
first session sets up the skeleton and the test harness only. We will implement
behavior in later sessions.
</context>

<task>
1. Propose a concrete repo layout and a short build plan as a numbered checklist, and
   write the checklist to PLAN.md.
2. Scaffold the project: pyproject.toml (managed by uv), ruff and pytest config,
   pre-commit, a package skeleton with empty modules matching the layout in CLAUDE.md,
   and a tests/ directory with one trivial passing test so the harness runs.
3. Add scripts/init.sh that creates the environment, installs deps, runs ruff, and runs
   pytest, so future sessions can verify the project in one command.
</task>

<constraints>
- Do not implement any model, loss, or data logic yet. Empty function stubs with type
  signatures and a one-line docstring are fine.
- Keep dependencies minimal and pinned to major versions only.
- The SIGReg core package must not import anything from the rest of the project.
</constraints>

<success_criteria>
- `bash scripts/init.sh` completes with ruff clean and pytest green.
- PLAN.md lists the phases and the three SIGReg placements as explicit milestones.
</success_criteria>

<output_format>
First give the plan and your assumptions in prose. Then create the files. End with the
exact command I should run to verify, and the result you expect.
</output_format>
```

---

## Phase 1: SIGReg core, ground-truthed against scipy

```
<context>
SIGReg lifts a univariate normality test to multivariate embeddings by projecting onto
random unit directions (slices) and averaging the per-slice statistic. The univariate
test we use is Epps-Pulley: it integrates the squared difference between the empirical
characteristic function and the standard normal characteristic function. Correctness
here is the foundation for everything else, so it must be tested against an independent
reference.
</context>

<task>
Implement in chronojepa/sigreg/:
1. EppsPulley univariate test (numerical integration over [0, t_max], exploiting symmetry
   so only t >= 0 is computed).
2. A random-slicing wrapper that projects (N, D) embeddings onto K normalized Gaussian
   directions and aggregates per-slice statistics with a configurable reduction.
3. A PooledSIGReg loss that pools a sequence embedding to (N, D) and applies the above.
</task>

<constraints>
- Pure PyTorch in this package, no project-internal imports.
- Device-agnostic: never hardcode the device.
- Fully differentiable: do not use in-place masked writes on tensors that require grad.
- Add type hints and a short docstring only where logic is not obvious.
</constraints>

<success_criteria>
Write tests in tests/test_sigreg.py that:
- Confirm the loss is near zero for large samples drawn from N(0, I) and clearly larger
  for non-Gaussian inputs (uniform, shifted, scaled).
- Cross-check the univariate statistic against a scipy or closed-form reference within a
  stated tolerance.
- Confirm gradients flow (loss.backward() populates grads on a leaf input).
- Confirm identical results on CPU and, if available, MPS.
All tests pass under `uv run pytest tests/test_sigreg.py -q`.
</success_criteria>

<output_format>
Show the test results after implementing. If a test fails, fix the code, not the test.
</output_format>
```

---

## Phase 2: Encoders and RevIN

```
<context>
We need a sequence encoder that maps a multivariate window (batch, channels, time) to a
sequence of embeddings. PatchTST-style patching with channel independence is a strong,
standard choice for time series. RevIN (reversible instance normalization) stabilizes
forecasting under distribution shift.
</context>

<task>
Implement in chronojepa/models/:
1. A PatchTST-style transformer encoder: patch the time axis, treat channels
   independently, return both per-patch embeddings and a pooled embedding.
2. A small TCN encoder as a baseline with the same input and output contract.
3. RevIN as a module with normalize and denormalize methods.
Keep a single documented tensor contract shared by both encoders.
</task>

<constraints>
Keep both encoders under a combined budget that stays readable; prefer composition over
deep class hierarchies. No pretrained weights. Device-agnostic.
</constraints>

<success_criteria>
tests/test_models.py checks output shapes for both encoders on a toy batch, confirms
RevIN normalize then denormalize is close to identity, and confirms a forward and
backward pass runs on CPU and MPS if present. Tests pass.
</success_criteria>

<output_format>
State the shared tensor contract explicitly in prose before coding, then implement.
</output_format>
```

---

## Phase 3: Data and augmentations (PEMS first)

```
<context>
PEMS is a standard public time-series benchmark, which lets us compare against published
forecasting numbers later. SSL needs two augmented views per window. Time-series
augmentations differ from image ones: jitter, scaling, time and frequency masking, and
random cropping along time.
</context>

<task>
Implement in chronojepa/data/:
1. A PEMS loader that produces train, val, and test windows with a sliding window.
2. A two-view augmentation pipeline (jitter, scaling, masking, crop) returning two views
   of each window plus the label or target.
3. A torch Dataset and a factory that builds DataLoaders from a Hydra config.
</task>

<constraints>
- No look-ahead bias: compute normalization statistics on the train split only and apply
  them forward to val and test. Splits must respect time order.
- Make augmentation strengths config-driven, not hardcoded.
</constraints>

<success_criteria>
tests/test_data.py asserts: train, val, and test windows do not overlap in time; train
statistics are reused for val and test; each item yields two views with matching shapes.
Tests pass, and a one-batch smoke load prints shapes.
</success_criteria>

<output_format>
Before coding, state in one sentence how you prevent look-ahead bias, then implement.
</output_format>
```

---

## Phase 4: SIGReg placements and training loop (the core contribution)

```
<context>
This is the heart of the project. We compare how the SIGReg objective is placed relative
to the time axis, because the naive pooled placement collapses each sequence to a constant
"ID vector" along time while still reporting low SIGReg loss. We need the dual placement
(SIGReg within each sequence across time, plus across samples in the batch) and a
structured variant, plus the LeJEPA prediction or invariance term across the two views.
</context>

<task>
Implement in chronojepa/sigreg/ and chronojepa/train/:
1. DualSIGReg and StructuredSIGReg losses alongside the existing PooledSIGReg, behind a
   common interface selectable by config name.
2. The combined objective: lejepa_loss = sigreg * lambda + invariance * (1 - lambda),
   where invariance pulls the two views of a window together. Optionally route invariance
   through a small MLP predictor for a predictive variant.
3. A minimal training loop: device-agnostic, bf16 where supported, W&B logging of each
   loss term, cosine schedule with warmup, seeded.
</task>

<constraints>
Single hyperparameter lambda for the tradeoff, matching the LeJEPA philosophy. No
stop-gradient, no teacher-student, no EMA. Keep the loop lean and readable. Surgical
changes only: do not alter Phase 1 code beyond adding the new loss classes behind the
shared interface.
</constraints>

<success_criteria>
- A 50-step smoke run on PEMS completes, and all loss terms are finite and logged.
- A unit test confirms each placement returns a scalar and backpropagates.
- Config can switch placement among pooled, dual, structured without code changes.
</success_criteria>

<output_format>
Briefly explain how dual placement is expected to prevent the time-axis collapse before
implementing. Then implement and show the smoke-run log.
</output_format>
```

---

## Phase 5: Collapse diagnostics and downstream evaluation

```
<context>
We need to measure collapse directly and evaluate representation quality. Because SIGReg
forces embeddings toward N(0, I), Mahalanobis distance becomes a principled anomaly score
for free. We also want a linear or kNN probe and a forecasting head.
</context>

<task>
Implement in chronojepa/eval/:
1. Collapse diagnostics: per-sequence variance of the embedding across time, effective
   rank of the embedding matrix, and a printed or logged collapse report.
2. A frozen-encoder linear probe and a kNN probe for any labeled dataset.
3. A forecasting evaluation that attaches a small head to frozen features and reports MAE
   and MSE on PEMS.
4. A Mahalanobis anomaly scorer fit on training embeddings.
</task>

<constraints>
Frozen encoder during probing (no gradients into the backbone). Reuse the no-look-ahead
splits from Phase 3. Surgical: do not modify training-loop internals.
</constraints>

<success_criteria>
Run pooled vs dual on a short PEMS run and produce a comparison: the diagnostic should
show low across-time variance for pooled (collapse) and higher variance for dual, and the
dual probe or forecast metric should be at least as good as pooled. Save the numbers to a
results file.
</success_criteria>

<output_format>
Present the comparison as a small table of placement vs collapse metric vs downstream
metric. State plainly which placement wins on this run.
</output_format>
```

---

## Phase 6: Label-free model selection

```
<context>
LeJEPA reports a high rank correlation between the SIGReg training loss and downstream
performance, which means we can select checkpoints and architectures without labels. We
want to verify and use this.
</context>

<task>
Implement in chronojepa/eval/model_selection.py a utility that, given a set of runs or
checkpoints, ranks them by final SIGReg loss and reports the rank correlation (Spearman)
between that ranking and the labeled downstream metric, to test whether label-free
selection holds on time series.
</task>

<constraints>
Read existing logged metrics; do not retrain inside this utility. Keep it a pure analysis
function with a thin CLI wrapper.
</constraints>

<success_criteria>
Given at least three runs with different lambda or encoder settings, the utility prints
the Spearman correlation and the label-free top pick versus the label-based top pick.
A unit test feeds synthetic correlated and uncorrelated inputs and checks the correlation
sign and magnitude.
</success_criteria>
```

---

## Phase 7: Experiment runner, configs, sweeps

```
<context>
We need reproducible, configurable experiments to run the placement comparison across
datasets and seeds, with results that can go into a writeup or back into LeJEPA issue #27.
</context>

<task>
1. Build Hydra configs for model, data, optimizer, and experiment, including named
   experiments smoke, pems_pooled, pems_dual, pems_structured.
2. Make scripts/train.py the single entry point driven entirely by config.
3. Add a sweep over lambda and the three placements, with W&B logging and the resolved
   config saved per run.
</task>

<constraints>
No hardcoded paths or hyperparameters in code; everything lives in configs. Each run is
seeded and the seed is logged.
</constraints>

<success_criteria>
`uv run python scripts/train.py +experiment=pems_dual` runs end to end and logs to W&B.
A sweep launches multiple runs that differ only by config. Resolved configs are saved.
</success_criteria>
```

---

## Phase 8: README, reproducibility, and writeup

```
<context>
The project should be reproducible by a stranger and legible to a hiring manager, and the
key finding (which placement fixes the collapse) should be presentable as a short result.
</context>

<task>
Write README.md covering the idea, the connection to LeJEPA and issue #27, the tech stack,
install and run commands, the placement comparison result with the saved numbers, and the
known limitations. Add a RESULTS.md with the comparison table and plots.
</task>

<constraints>
Prose over heavy bullet lists. No em dashes. Only claim results that are in the saved
results files; do not invent numbers.
</constraints>

<success_criteria>
A new user can clone, run init.sh, run one named experiment, and reproduce a row of the
results table. The README states clearly which placement won and by how much.
</success_criteria>
```

---

## Reusable: review and self-correct (run after a phase)

```
<task>
Review the change you just made against its success criteria as if you were a strict
reviewer who did not write it.
</task>

<checklist>
- Does it meet every success criterion, verified by running the tests?
- Is it the simplest version that works, or is there speculative complexity to remove?
- Did it touch anything outside the task scope? If so, revert that.
- Any device assumptions, NaN or Inf risks, autograd-breaking in-place ops, or
  look-ahead leakage?
- Any em dashes in new comments or docs?
</checklist>

<output_format>
List concrete issues with file and line. Then apply only the fixes that matter and rerun
the tests. Report what changed and the new test result.
</output_format>
```

---

## Reusable: debugging

```
<context>
[paste the failing command and the full error or the unexpected metric]
</context>

<task>
Find the root cause before changing anything. Read the relevant files; do not speculate
about code you have not opened. Form two or three competing hypotheses, then identify the
cheapest check that distinguishes them.
</task>

<constraints>
Do not mute the symptom (no broad try/except, no loosening a test to pass). Fix the cause.
Make the smallest change that resolves it.
</constraints>

<output_format>
State the hypotheses, the check you ran, the root cause, and the minimal fix. Then show
the passing result.
</output_format>
```
