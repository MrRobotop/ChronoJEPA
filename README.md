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
3. `structured`: a multivariate or joint formulation. This is an open research question, and
   the repository ships an initial concrete instantiation of it.

Everything in the repository serves measuring which placement prevents collapse and yields
the best downstream representations.

## Result so far

On the real PEMS08 traffic benchmark, across five seeds, the dual placement robustly prevents
the time-axis collapse: it keeps about 9.5 times the across-time variance of the pooled baseline
(0.607 against 0.064) and a clearly higher effective rank (11.5 against 8.1), with no overlap in
the seed-to-seed bands. So the central mechanistic claim holds firmly on real data. The honest
surprise is downstream: preventing the collapse does not help this task. Even with a temporally
sensitive probe (predict the full horizon trajectory from the token sequence), pooled forecasts
reliably better than dual (MAE 0.440 against 0.457), and the gap sits outside the seed noise.
Short-horizon traffic forecasting is close to persistence, so the level that a collapsed
representation retains is most of what the task needs. Collapse and forecasting quality are
decoupled here. See [RESULTS.md](RESULTS.md) for the full tables, the refuted hypothesis, and
the next experiments worth running.

## Tech stack

Python 3.11 or newer and PyTorch 2.x, device-agnostic across CUDA, Apple MPS, and CPU. The
model side is a PatchTST-style transformer encoder with a TCN baseline and RevIN. SIGReg
ground-truthing and evaluation use scipy and scikit-learn, configuration uses Hydra and
OmegaConf, and experiment logging uses Weights and Biases (optional, offline by default).
Environments are managed with uv, linting and formatting with ruff, tests with pytest, and
commit gates with pre-commit. The SIGReg core itself depends only on torch.

## Install

This project uses [uv](https://docs.astral.sh/uv/). The pinned interpreter is recorded in
`.python-version`, and uv will provision it for you.

```bash
uv sync
```

## Quickstart

Verify a clean checkout in one command (creates the environment, lints, and runs the tests):

```bash
bash scripts/init.sh
```

Run a fast end-to-end training smoke on synthetic data:

```bash
uv run python scripts/train.py +experiment=smoke
```

Reproduce the placement comparison table:

```bash
uv run python scripts/compare.py
```

Sweep over the placements and the lambda tradeoff with Hydra multirun:

```bash
uv run python scripts/train.py -m placement=pooled,dual,structured lam=0.1,0.5,0.9
```

Train on PEMS once you have downloaded the dataset:

```bash
uv run python scripts/train.py +experiment=pems_dual data.path=/path/to/pems.npz
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
  models/   encoders (PatchTST, TCN), an MLP predictor, RevIN
  data/     dataset loaders and time-series augmentations
  train/    training loop and the config-driven experiment runner
  eval/     probes, forecasting, collapse diagnostics, anomaly scoring, model selection
  utils/    seeding, device selection, logging
configs/    Hydra configs (data, model, optimizer, named experiments)
scripts/    runnable entry points (init.sh, train.py, compare.py, plot_results.py)
tests/      pytest suite
```

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
