# ChronoJEPA

ChronoJEPA is a heuristics-free self-supervised representation learning library for
multivariate and financial time series. It is built on the SIGReg objective introduced by
LeJEPA (arXiv:2511.08544), and its purpose is to study how that objective should be placed
relative to the time axis so that it learns useful representations instead of collapsing.

## The idea

SIGReg regularizes a batch of embeddings toward an isotropic Gaussian. It does this by
lifting a univariate normality test to many dimensions: it projects the embeddings onto
random unit directions and averages a per-direction statistic. Unlike most self-supervised
methods, it needs no stop-gradient, no teacher-student pair, no exponential moving average,
and no schedulers. That simplicity is what makes it attractive to carry into the time domain.

## The research bet and the connection to LeJEPA issue #27

Applying SIGReg naively to a pooled sequence embedding causes a time-axis collapse. Each
sample converges to a constant "ID vector" along time, yet SIGReg still reports a low loss
because the pooled batch can look Gaussian even when every sequence is internally flat. This
failure is described in LeJEPA issue #27. The core contribution of this repository is to
diagnose that collapse and compare three placements of the objective:

1. `pooled`: one SIGReg over the pooled sequence embedding. This is the baseline and is
   expected to collapse.
2. `dual`: SIGReg applied within each sequence across time, plus across samples in the
   batch.
3. `structured`: a multivariate or joint formulation. This is an open research question.

Everything in the repository serves measuring which placement prevents collapse and yields
the best downstream representations.

## Tech stack

Python 3.11 or newer and PyTorch 2.x, device-agnostic across CUDA, Apple MPS, and CPU. The
planned model side is a PatchTST-style transformer encoder with a TCN baseline and RevIN for
forecasting. Configuration uses Hydra and OmegaConf, experiment logging uses Weights and
Biases, data handling uses polars and pandas with yfinance for financial series, and
evaluation uses scipy and scikit-learn. Environments are managed with uv, linting and
formatting with ruff, tests with pytest, and commit gates with pre-commit.

Dependencies are introduced phase by phase to keep each step minimal. At this stage only
PyTorch and the development tools are declared.

## Install

This project uses [uv](https://docs.astral.sh/uv/). The pinned interpreter is recorded in
`.python-version`, and uv will provision it for you.

```bash
uv sync
```

## Quickstart

The fastest way to verify a clean checkout is the one-shot script, which creates the
environment, installs dependencies, lints, and runs the tests:

```bash
bash scripts/init.sh
```

## Running the tests

```bash
uv run pytest -q
```

Lint and format checks match what continuous integration runs:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Project layout

```
chronojepa/
  sigreg/   SIGReg objective: univariate tests, random slicing, placement variants
  models/   encoders (PatchTST, TCN), optional MLP predictor, RevIN
  data/     dataset loaders and time-series augmentations
  train/    trainer and training loop
  eval/     probes, forecasting, anomaly scoring, label-free model selection
  utils/    seeding, device selection, logging
configs/    Hydra configs
scripts/    runnable entry points
tests/      pytest suite
```

## Status

This is the Phase 0 scaffold: project layout, tooling, and a passing test harness. The
build plan and the placement milestones are tracked in [PLAN.md](PLAN.md). Behavior is
implemented in later phases.

## Citation

This work builds directly on LeJEPA and its SIGReg objective.

```bibtex
@misc{lejepa,
  title = {LeJEPA},
  note  = {arXiv:2511.08544},
  url   = {https://arxiv.org/abs/2511.08544},
  year  = {2025}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
