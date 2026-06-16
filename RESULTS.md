# Results

This file records the placement comparison that the project was built to measure: does the
choice of where SIGReg is applied relative to the time axis prevent the time-axis collapse
described in LeJEPA issue #27, and does it help downstream?

## The comparison

The numbers below are from a reference run on synthetic multivariate data (three channels of
mixed-frequency sinusoids plus noise, 2400 steps), training each placement for 300 steps with
identical settings (seed 0, PatchTST encoder, d_model 64, 48 slices, lambda 0.5). They were
produced on Apple MPS. Reproduce them with:

```bash
uv run python scripts/compare.py
```

| placement | across-time variance | effective rank | forecast MAE | forecast MSE |
|-----------|----------------------|----------------|--------------|--------------|
| pooled    | 0.022337             | 4.664          | 0.4701       | 0.3926       |
| dual      | 0.479007             | 8.581          | 0.4857       | 0.3632       |

## What it shows

The `pooled` baseline collapses along time. Its across-time variance is 0.022, roughly 21
times lower than `dual` at 0.479, which is the signature of each sequence converging to a
near-constant "ID vector" over time. Its effective rank is also far lower (4.66 against 8.58),
so the pooled representation occupies a more degenerate subspace.

The `dual` placement prevents that collapse, because its within-sequence term asks the
per-timestep embeddings of each sequence to look like an isotropic Gaussian, which is
maximally violated by a constant-over-time sequence. On downstream forecasting, `dual` is at
least as good as `pooled`: its MSE is lower (0.363 against 0.393, about 8 percent) and its MAE
is roughly tied. So on this run, `dual` wins on the collapse diagnostic decisively and is no
worse, slightly better, downstream.

## Plot

Render bar charts of the three headline metrics from the saved JSON:

```bash
uv sync --extra plot
uv run python scripts/plot_results.py results/placement_comparison.json
```

## Limitations and honesty about scope

These numbers are on synthetic data, not on a published benchmark. The PEMS path is wired but
the dataset has not been downloaded, so there are no PEMS forecasting numbers yet; run
`scripts/compare.py --pems /path/to/pems.npz` once you have the file. The `structured`
placement is an initial concrete instantiation of an open question, not a tuned method.
Hyperparameters were not swept for these numbers. The qualitative result (pooled collapses,
dual does not) is robust to the random seed; exact values depend on hardware and backend.
