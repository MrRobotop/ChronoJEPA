# Results

This file records the placement comparison the project was built to measure: does where
SIGReg is applied relative to the time axis prevent the time-axis collapse from LeJEPA issue
#27, and does preventing it help downstream? The honest short answer on PEMS08: it prevents
the collapse robustly, and it does not help this forecasting task. Both halves are reported as
measured.

## PEMS08, multi-seed, temporally sensitive probe (the definitive run)

PEMS08 is California highway traffic: 170 sensors, 17856 five-minute steps, reduced to one
channel (flow) per sensor. The dataset (about 17.7 MB) is `data/PEMS08/pems08.npz` from
github.com/wanhuaiyu/ASTGCN. This run uses the trajectory probe (predict the full next-horizon
trajectory from the flattened token sequence, which depends on temporal structure) and
aggregates over five seeds (0 to 4), 500 steps per placement, PatchTST d_model 64, 32 slices,
window 96, horizon 12, lambda 0.5, on Apple MPS. Reproduce with:

```bash
uv run python scripts/compare.py --pems data/pems08.npz --steps 500 \
  --batch-size 32 --num-slices 32 --forecast trajectory --seeds 5
```

| placement | across-time variance | effective rank | trajectory MAE | trajectory MSE |
|-----------|----------------------|----------------|----------------|----------------|
| pooled    | 0.064 +- 0.007       | 8.11 +- 0.64   | 0.4403 +- 0.0050 | 0.4846 +- 0.0153 |
| dual      | 0.607 +- 0.008       | 11.49 +- 0.24  | 0.4569 +- 0.0030 | 0.5074 +- 0.0091 |

What it shows, stated plainly:

1. The collapse result is strong and robust. The dual placement keeps about 9.5 times the
   across-time variance of pooled (0.607 against 0.064) and a clearly higher effective rank
   (11.5 against 8.1), and the seed-to-seed spread is tiny, so the bands do not overlap. The
   central mechanistic claim, that dual prevents the time-axis collapse, holds firmly on real
   data.

2. Preventing the collapse does not help this task, and the trajectory probe did not change
   that. We added the trajectory probe specifically to test the hypothesis that the earlier
   mean probe was hiding a dual advantage. It was not: pooled forecasts better than dual on the
   full-trajectory probe too (MAE 0.440 against 0.457, MSE 0.485 against 0.507), and across five
   seeds the gap sits outside the standard-deviation bands, so it is not noise. The hypothesis
   is refuted on PEMS08.

A plausible reason is the nature of the task. Short-horizon traffic forecasting is close to
persistence: the next hour is mostly determined by the current level, which even a collapsed
"ID vector" representation encodes well. The extra temporal spread that dual maintains is real
(higher variance and rank) but appears not to be what a linear forecasting head needs here, and
may even cost a little. So on PEMS08 the answer is that collapse and downstream forecasting
quality are decoupled: you can prevent the collapse without improving, and slightly worsening,
this particular forecast.

## PEMS08, single seed, mean probe (earlier run, for context)

The first PEMS run used the simpler mean probe (predict the next-horizon per-channel mean from
the pooled feature), single seed, same training settings. Reproduce with
`uv run python scripts/compare.py --pems data/pems08.npz --steps 500 --batch-size 32 --num-slices 32`.

| placement | across-time variance | effective rank | forecast MAE | forecast MSE |
|-----------|----------------------|----------------|--------------|--------------|
| pooled    | 0.0705               | 8.13           | 0.2583       | 0.2564       |
| dual      | 0.6023               | 11.21          | 0.2606       | 0.2597       |

The collapse result agrees with the multi-seed run. The forecasting gap there was about one
percent and within single-seed noise, which is what motivated the multi-seed trajectory run
above. The trajectory MAE and MSE are larger than the mean-probe ones simply because predicting
a full trajectory is harder than predicting its mean; the two probes are not directly comparable.

## Synthetic sanity run

A fixed synthetic series (three mixed-frequency channels plus noise, 2400 steps), 300 steps,
mean probe. Reproduce with `uv run python scripts/compare.py`.

| placement | across-time variance | effective rank | forecast MAE | forecast MSE |
|-----------|----------------------|----------------|--------------|--------------|
| pooled    | 0.0223               | 4.66           | 0.4701       | 0.3926       |
| dual      | 0.4790               | 8.58           | 0.4857       | 0.3632       |

On synthetic data dual both prevents the collapse and gives a lower forecasting MSE. The
collapse result matches PEMS08; the downstream result does not, which is part of why the real
benchmark matters.

## Conclusion and next steps

The mechanistic claim holds: the dual placement prevents the time-axis collapse, robustly and by
a large margin, on real data. The utility claim does not hold on PEMS08 short-horizon
forecasting: preventing the collapse does not improve, and slightly worsens, this task, even
under a temporally sensitive probe and across seeds. This is a genuine, non-confirmatory result
worth reporting to LeJEPA issue #27.

The open question is whether any downstream task rewards the temporal structure dual preserves.
Forecasting that is near-persistence is the wrong place to look. More promising tests: long
horizons where persistence breaks down, sequence or regime classification, and anomaly detection
through the Mahalanobis scorer, where a richer, non-collapsed representation should matter more.
A lambda sweep and longer training are also untested levers.
