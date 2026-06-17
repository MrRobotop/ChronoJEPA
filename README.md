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

The headline, from an architecture-by-placement factorial on PEMS08, is that the time-axis
collapse diagnostic does not measure what we assumed. Across-time variance does not track whether
a representation keeps temporal order, and in the factorial it is anticorrelated with it: the most
collapsed configuration (a positional transformer, pooled placement) recovers a content-matched
half-swap at 0.98, while the least collapsed one (a position-free bag-of-patches encoder, variance
about 15x higher) is at chance, 0.50. What determines order recovery is positional encoding, not
the collapse and not the placement. The dual placement that fixes the collapse therefore helps no
downstream order task in either architecture: a positional encoder already has the order, and a
position-free one cannot get it back by preventing the collapse. See `figures/halfswap.png` and
the factorial table in [RESULTS.md](RESULTS.md).

The investigation that led there, in detail. On the real PEMS08 traffic benchmark, across five
seeds, the dual placement robustly prevents
the time-axis collapse: it keeps about 9.5 times the across-time variance of the pooled baseline
(0.607 against 0.064) and a clearly higher effective rank (11.5 against 8.1), with no overlap in
the seed-to-seed bands. So the central mechanistic claim holds firmly on real data. The honest
surprise is downstream: preventing the collapse does not help this task. Even with a temporally
sensitive probe (predict the full horizon trajectory from the token sequence), pooled forecasts
reliably better than dual (MAE 0.440 against 0.457), and the gap sits outside the seed noise.
We then ruled out the obvious explanation. The guess that pooled wins only because short
horizons are near persistence predicts the gap should close at longer horizons; a horizon sweep
from 3 to 48 steps shows it does not, so pooled's small edge is horizon-independent, not a
persistence artifact. A third test, Mahalanobis anomaly detection on token features, points the
same way: both placements detect injected anomalies near-perfectly. A lambda sweep then supplies
the mechanism and a fourth test: SIGReg and the forecaster are in tension, since raising lambda
raises effective rank but monotonically worsens dual's forecasting, and final SIGReg loss does
not track downstream quality, so LeJEPA's label-free selection claim does not transfer to this
task. A fifth test finally finds where dual wins: on a temporal-order classification task (is the
window rising or falling) dual beats pooled, consistently across seeds, while both tie on a level
control. The margin is small but in the predicted direction. A content-matched pure-position task
(window vs the same window with halves swapped), built to be decisive, then led to the central
finding. It saturated for both placements, and a controlled follow-up, probing the time-mean
feature to simulate full collapse, still classified it at about 0.98 rather than the predicted
chance. The reason is architectural: the time-mean of PatchTST tokens is not permutation
invariant, because positional encoding and attention write order into the token values before any
pooling. So low across-time variance, which is what the collapse diagnostic measures, does not
entail loss of order information. That is why the collapse is downstream-benign on PEMS: the
order-relevant information lives in the token values, which a positional transformer computes
order-sensitively whether or not the tokens vary across time. The collapse is real and the dual
placement is a clean fix, but on a positional transformer it is not by itself a downstream
problem. See [RESULTS.md](RESULTS.md) for the full tables, the refuted hypotheses, and the
experiments behind this.

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
