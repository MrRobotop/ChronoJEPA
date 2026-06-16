# CLAUDE.md

This file is loaded into context at the start of every Claude Code session in this repo. Keep it short and high-signal. It is a constraints document, not documentation: its job is to interrupt default agent behavior before it runs, not to teach the codebase.

## Project

ChronoJEPA is a heuristics-free self-supervised representation learning library for multivariate and financial time series, built on the SIGReg objective from LeJEPA (arXiv:2511.08544). SIGReg regularizes embeddings toward an isotropic Gaussian with no stop-gradient, no teacher-student, and no schedulers. We extend it from images to the time domain.

## The research bet

Applying SIGReg naively to sequence embeddings causes a time-axis collapse: each sample converges to a constant "ID vector" along time while SIGReg still reports low loss (see LeJEPA issue #27). The core contribution of this repo is to diagnose that failure and compare three placements of the objective:
1. `pooled`: one SIGReg over the pooled sequence embedding (baseline, expected to collapse).
2. `dual`: SIGReg applied within each sequence across time, plus across samples in the batch.
3. `structured`: a multivariate or joint formulation (open research question).

Everything in the repo serves measuring which placement prevents collapse and yields the best downstream representations.

## Tech stack

Python 3.11+. PyTorch 2.x, device-agnostic (CUDA, Apple MPS, CPU). PatchTST-style transformer encoder plus a TCN baseline, with RevIN for forecasting. Hydra and OmegaConf for config. Weights and Biases for logging. polars and pandas for data, yfinance for financial series, scipy and scikit-learn for evaluation and ground-truth tests. uv for environments, ruff for lint and format, pytest for tests, pre-commit for gates.

## Repo map

- `chronojepa/sigreg/` core objective: univariate tests, random slicing, placement variants. Pure PyTorch, no project-internal imports.
- `chronojepa/models/` encoders (PatchTST, TCN), optional MLP predictor, RevIN.
- `chronojepa/data/` dataset loaders (PEMS first, then ETT and financial), time-series augmentations.
- `chronojepa/train/` trainer and training loop.
- `chronojepa/eval/` linear and kNN probe, forecasting, Mahalanobis anomaly scoring, label-free model selection.
- `chronojepa/utils/` seeding, device selection, logging.
- `configs/` Hydra configs. `tests/` pytest. `scripts/` runnable entry points.

## Commands

- Install: `uv sync`
- Lint and format: `uv run ruff check . --fix && uv run ruff format .`
- Test: `uv run pytest -q`
- Single test: `uv run pytest tests/test_sigreg.py::test_pooled_matches_scipy -q`
- Train (smoke): `uv run python scripts/train.py +experiment=smoke`
- Train (PEMS): `uv run python scripts/train.py +experiment=pems_dual`

## Working agreement

These four rails apply to every change.

1. Think before coding. State your assumptions and your plan before writing code. When the request has more than one reasonable interpretation, ask one clarifying question instead of silently picking one. Surface inconsistencies and tradeoffs rather than burying them.
2. Keep it simple. Prefer the smallest implementation that satisfies the success criteria. If 200 lines could be 50, write 50. Do not add speculative flexibility, abstractions for single-use code, or features nobody asked for.
3. Make surgical changes. Touch only what the task requires. Do not reformat, rename, or refactor adjacent code, and do not fix unrelated issues in the same change. Remove dead code that your own change introduces; for pre-existing dead code, mention it rather than deleting it.
4. Work toward verifiable success criteria. Each task defines what done looks like. Loop toward that target and stop when it is met. Do not declare success without running the relevant tests.

## Code style

Type hints on all public functions. Docstrings only where the logic is not self-evident; do not add docstrings or comments to code you did not change. Follow ruff defaults. Keep the SIGReg core free of dependencies beyond torch so it stays portable into other codebases.

## Domain rules

- Device-agnostic from the start. Resolve the device once through `utils.devices.get_device()` and pass it down. Never hardcode `"cuda"`. Code must run on Apple MPS for local development.
- SIGReg numerics. Validate every statistical test against scipy or a closed-form value in `tests/`. Guard against NaN and Inf. Watch for in-place masked writes on tensors that carry gradients, since they break autograd.
- No look-ahead bias. For any financial pipeline, normalization statistics and splits must be computed on training data only and applied forward. Never let future timestamps inform past windows. Call this out explicitly when writing data code.
- Reproducibility. Seed numpy and torch through `utils.seed.set_seed()`. Save the resolved config alongside every run.

## Definition of done

A change is done when: ruff passes, the targeted tests pass, new behavior has a test, and a smoke run completes without error. Report results as facts (what ran, what passed) rather than as a summary of intentions.

## Writing style

In all generated docstrings, comments, commit messages, and docs, write without em dashes. Use commas, colons, parentheses, or separate sentences instead. Prefer plain prose over heavy bullet formatting in long-form docs.

## Ask before

Confirm before any hard-to-reverse or shared-system action: deleting files or branches, `git push`, `git reset --hard`, rewriting history, or changing dependency versions. Local, reversible actions (editing files, running tests) need no confirmation.
