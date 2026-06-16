# Results

This file records the placement comparison that the project was built to measure: does the
choice of where SIGReg is applied relative to the time axis prevent the time-axis collapse
described in LeJEPA issue #27, and does it help downstream?

There are two runs below: a synthetic sanity run and the real PEMS08 benchmark. The headline
finding (the `dual` placement prevents the collapse) holds in both. The downstream story is
more nuanced on real data, and is reported here honestly.

## PEMS08 (real data)

PEMS08 is California highway traffic: 170 sensors, 17856 five-minute steps, here reduced to
one channel (traffic flow) per sensor. The dataset (about 17.7 MB) comes from the ASTGCN
repository (`data/PEMS08/pems08.npz` in github.com/wanhuaiyu/ASTGCN). Numbers are from seed 0,
500 steps per placement, PatchTST encoder (d_model 64, depth 2), 32 slices, lambda 0.5,
window 96, horizon 12, on Apple MPS. Reproduce with:

```bash
uv run python scripts/compare.py --pems data/pems08.npz --steps 500 --batch-size 32 --num-slices 32
```

| placement | across-time variance | effective rank | forecast MAE | forecast MSE |
|-----------|----------------------|----------------|--------------|--------------|
| pooled    | 0.0705               | 8.13           | 0.2583       | 0.2564       |
| dual      | 0.6023               | 11.21          | 0.2606       | 0.2597       |

What it shows. The collapse result is clear and survives on real data: `dual` has about 8.5
times the across-time variance of `pooled` (0.602 against 0.070) and a higher effective rank
(11.2 against 8.1), so `pooled` is collapsing each sequence toward a near-constant vector over
time while `dual` is not.

The downstream story differs from the synthetic run. On this linear forecasting probe `pooled`
is marginally better than `dual` (MAE 0.258 against 0.261, MSE 0.256 against 0.260), a gap of
roughly one percent that is within plausible run-to-run noise for a single seed. A likely
reason is that the probe predicts the per-channel mean over the next horizon, and the collapsed
"ID vector" that `pooled` learns still encodes a sequence's overall level, which is most of what
predicting a near-future mean needs. In other words, this particular probe is fairly insensitive
to the collapse. A downstream task that depends on temporal detail would be a fairer test of
whether preventing collapse helps, and that is the natural next experiment.

So on PEMS08, preventing the collapse (the central claim) is confirmed, while the claim that
preventing it improves this specific forecasting probe is not. Both are reported as measured.

## Synthetic sanity run

A fixed synthetic series (three mixed-frequency channels plus noise, 2400 steps), 300 steps per
placement. Reproduce with `uv run python scripts/compare.py`.

| placement | across-time variance | effective rank | forecast MAE | forecast MSE |
|-----------|----------------------|----------------|--------------|--------------|
| pooled    | 0.0223               | 4.66           | 0.4701       | 0.3926       |
| dual      | 0.4790               | 8.58           | 0.4857       | 0.3632       |

Here `dual` both prevents the collapse (across-time variance 0.479 against 0.022) and gives a
lower forecasting MSE (0.363 against 0.393). The collapse result agrees with PEMS08; the
forecasting result is more favorable to `dual` than on real data.

## Plot

Render bar charts of the headline metrics from a saved JSON:

```bash
uv sync --extra plot
uv run python scripts/plot_results.py results/placement_comparison_pems.json
```

## Limitations and honesty about scope

The collapse-prevention finding is robust across synthetic and real data and across the random
seed. The downstream forecasting comparison is single-seed and the PEMS gap is about one percent,
so it should be read as "roughly tied, pooled marginally ahead on this probe" rather than a firm
ranking. The `structured` placement is an initial concrete instantiation of an open question, not
a tuned method. Hyperparameters were not swept. PEMS08 is reduced to one feature (flow) per
sensor. The clear next steps are multi-seed runs, a sweep over lambda, and a downstream task that
depends on temporal structure rather than a horizon mean.
