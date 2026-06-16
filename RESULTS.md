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

## PEMS08 forecasting vs horizon (refutes the persistence explanation)

The explanation above was a hypothesis: pooled wins because short-horizon traffic is near
persistence, so the gap should shrink or reverse as the horizon grows and persistence breaks
down. We tested it directly with a horizon sweep (trajectory probe, three seeds, each encoder
trained once and probed at every horizon). Reproduce with:

```bash
uv run python scripts/horizon_sweep.py --pems data/pems08.npz --seeds 3 --horizons 3,6,12,24,48
```

| horizon | pooled MAE | dual MAE | gap (dual minus pooled) |
|---------|------------|----------|--------------------------|
| 3       | 0.3395 +- 0.0011 | 0.3453 +- 0.0021 | +0.0058 |
| 6       | 0.3422 +- 0.0013 | 0.3489 +- 0.0012 | +0.0067 |
| 12      | 0.3473 +- 0.0023 | 0.3545 +- 0.0001 | +0.0072 |
| 24      | 0.3617 +- 0.0044 | 0.3669 +- 0.0033 | +0.0051 |
| 48      | 0.3805 +- 0.0073 | 0.3854 +- 0.0069 | +0.0049 |

The hypothesis is refuted. Pooled stays about 0.005 to 0.007 MAE better at every horizon from 3
(fifteen minutes) to 48 (four hours), with no shrinking and no reversal, and the gap exceeds the
seed spread at the short and middle horizons. Absolute error grows with horizon for both, as
expected, but the ordering is stable. So pooled's edge is not a near-persistence artifact: the
collapsed representation is simply, modestly, and reliably better for this forecasting task
across the whole horizon range. Why preventing the collapse costs a little here, rather than
being merely neutral, is the open question; a plausible read is that the within-sequence SIGReg
term spends representational capacity making each sequence isotropic over time, which is real
structure but not what a linear forecasting head rewards.

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

## PEMS08 anomaly detection (Mahalanobis on token features)

To test whether the temporal structure dual preserves helps a task that should need it, we fit
a Mahalanobis scorer on the flattened token features of normal training windows, then measure
AUROC separating real test windows from windows with injected anomalies. Three anomaly types:
`spike` (transient amplitude bumps), `shuffle` (full time permutation, which also reshuffles
content across patches), and `block_shuffle` (permute contiguous blocks, preserving each block's
content, so only the order changes). Three seeds, 500 steps, d_model 32. Reproduce with:

```bash
uv run python scripts/anomaly.py --pems data/pems08.npz --seeds 3 --steps 500 --strength 1.5
```

| placement | spike AUROC | shuffle AUROC | block_shuffle AUROC |
|-----------|-------------|---------------|---------------------|
| pooled    | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.9981 +- 0.0016 |
| dual      | 1.000 +- 0.000 | 1.000 +- 0.000 | 0.9994 +- 0.0006 |

What it shows. Both placements detect every anomaly type near-perfectly, so the collapse does
not impair Mahalanobis anomaly detection here. The detector on high-dimensional token features
is sensitive enough that even a collapsed encoder, whose tokens are constant across time, still
responds to the changed patch contents and flags the anomaly. The one place the prediction shows
through is the order-only `block_shuffle`: it is the single anomaly where pooled drops below 1.0
while dual stays closer to it (0.9981 against 0.9994), a faint signal in the predicted direction.
But both are at ceiling, so this is not a meaningful downstream advantage. We did not keep
redesigning anomalies until one favored dual; this is reported as measured.

## Conclusion and next steps

The mechanistic claim holds: the dual placement prevents the time-axis collapse, robustly and by
a large margin, on real data. The utility claim does not hold on PEMS08. Across three honest
downstream tests, forecasting (mean and full-trajectory, multi-seed), forecasting swept over
horizons 3 to 48, and Mahalanobis anomaly detection, preventing the collapse gives no measurable
benefit: pooled is reliably a little better at forecasting at every horizon, and anomaly
detection saturates for both. We also ruled out our own leading explanation. The persistence
hypothesis (pooled wins only because short horizons are easy) predicts the gap should close as
the horizon grows; the horizon sweep shows it does not, so the explanation is wrong and pooled's
edge is horizon-independent. The accumulating picture on this benchmark is that the time-axis
collapse, though real and large, is downstream-benign and even mildly costly to prevent, because
the collapsed representation still carries the content and level these tasks rely on. This is a
genuine, non-confirmatory result worth reporting to LeJEPA issue #27.

What remains open is whether any task genuinely depends on the temporal structure dual preserves,
and why preventing the collapse costs a little rather than being neutral. Forecasting across
horizons and a saturating anomaly detector are both insensitive to the distinction. The most
discriminating untested options are limited-label sequence or regime classification, a weaker
(non-saturating) anomaly detector, and a sweep over lambda to see whether a different
SIGReg-to-invariance balance changes the downstream ordering. Longer training is also untested.
