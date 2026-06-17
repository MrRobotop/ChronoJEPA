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

## PEMS08 lambda sweep (the SIGReg vs forecasting tension, and label-free selection)

Sweeping the single lambda knob across both placements, three seeds, 500 steps, trajectory
probe at horizon 12. Reproduce with:

```bash
uv run python scripts/lambda_sweep.py --pems data/pems08.npz --seeds 3 --lambdas 0.1,0.3,0.5,0.7,0.9
```

| config        | sigreg loss | across-time var | eff rank | trajectory MAE   |
|---------------|-------------|-----------------|----------|------------------|
| pooled lam0.1 | 1.158       | 0.094           | 5.48     | 0.3469 +- 0.0026 |
| pooled lam0.3 | 1.044       | 0.106           | 6.30     | 0.3485 +- 0.0035 |
| pooled lam0.5 | 1.004       | 0.113           | 6.61     | 0.3473 +- 0.0023 |
| pooled lam0.7 | 0.973       | 0.118           | 6.80     | 0.3480 +- 0.0019 |
| pooled lam0.9 | 0.947       | 0.122           | 6.95     | 0.3494 +- 0.0015 |
| dual lam0.1   | 2.520       | 0.624           | 7.46     | 0.3495 +- 0.0032 |
| dual lam0.3   | 2.347       | 0.614           | 8.21     | 0.3537 +- 0.0019 |
| dual lam0.5   | 2.259       | 0.620           | 8.47     | 0.3545 +- 0.0001 |
| dual lam0.7   | 2.206       | 0.630           | 8.58     | 0.3555 +- 0.0028 |
| dual lam0.9   | 2.176       | 0.638           | 8.66     | 0.3559 +- 0.0034 |

Three findings.

1. Placement, not lambda, controls the collapse. Across-time variance is about 0.62 for dual
   and about 0.10 for pooled at every lambda. Lambda has only a mild within-placement effect:
   more SIGReg weight nudges pooled's variance up from 0.094 to 0.122 and its effective rank
   from 5.5 to 7.0, but never close to dual. So whether a sequence collapses is set by where
   SIGReg is applied, and lambda only tunes it slightly.

2. SIGReg and forecasting are in tension. As lambda rises, effective rank rises monotonically
   for both placements (more isotropy), but forecasting gets worse, clearly for dual: MAE climbs
   monotonically from 0.3495 at lambda 0.1 to 0.3559 at lambda 0.9, a gap of about two standard
   deviations end to end. Pooled forecasting stays flat within noise. So the very pressure that
   raises rank and prevents collapse trades against the forecasting head. This is the mechanism
   behind the earlier results: preventing the collapse is mildly costly because the isotropic
   regularization is not what a linear forecaster rewards.

3. Label-free selection by SIGReg loss does not work here, and the naive correlation is
   confounded. The cross-config Spearman between SIGReg loss and forecasting MAE is 0.552
   (p = 0.098, ten configs, not significant), and the label-free pick (lowest SIGReg loss,
   pooled lam0.9) disagrees with the label-based pick (lowest MAE, pooled lam0.1). More
   importantly, the raw cross-config ranking is not a fair test: DualSIGReg sums a within-time
   and a between-sample term, so its loss (about 2.2 to 2.5) is on a different scale from pooled
   (about 0.95 to 1.16), and the positive correlation is mostly the placement effect (pooled is
   lower on both axes). Within a single placement the relationship is flat for pooled and
   reversed for dual, where lower SIGReg loss (lambda 0.9) goes with the worst forecasting. So on
   PEMS08 forecasting, final SIGReg loss is not a reliable label-free selector, which is a
   non-confirmatory result for that LeJEPA claim on this task, with the caveat that SIGReg loss
   is only comparable within one objective formulation.

## PEMS08 temporal-structure classification (the first place dual wins)

A frozen-encoder linear probe on token features, classifying three binary labels built from the
windows: `trend` (is the second half's mean above the first half's, a subtle temporal-order
property), `level` (is the window mean above the train median, a control any representation can
do), and `halfswap` (a window against the same window with its two halves swapped, a content
matched pure-position task: the value multiset is identical, only the order changes). Three
seeds, 500 steps. Reproduce with:

```bash
uv run python scripts/classify.py --pems data/pems08.npz --seeds 3
```

| placement | trend accuracy   | level accuracy   | halfswap accuracy |
|-----------|------------------|------------------|-------------------|
| pooled    | 0.9747 +- 0.0038 | 0.9946 +- 0.0011 | 1.0000 +- 0.0000  |
| dual      | 0.9793 +- 0.0000 | 0.9946 +- 0.0022 | 0.9996 +- 0.0005  |

What it shows. On the level control both placements tie at 0.995. On the subtle trend task, which
needs temporal order, dual is the better representation: 0.9793 against 0.9747, a gap of about
half a percentage point consistent across all three seeds (dual's accuracy sits just above
pooled's mean plus one standard deviation). This is the one downstream task where dual beats
pooled, and it is exactly the temporal-order task dual's within-sequence term should suit.

The halfswap task was meant to be the decisive test: with content matched exactly, a
position-blind representation should sit near chance and only a position-aware one should classify
it. Instead both placements are at ceiling, pooled at 1.000 and dual at 0.9996.

To probe why, we reran classification on the pooled (time-mean) feature instead of the token
sequence, which simulates a fully collapsed representation. The prediction was that the time-mean
is permutation-invariant, so halfswap should drop to exactly 0.5.

| feature                       | trend            | level            | halfswap         |
|-------------------------------|------------------|------------------|------------------|
| token (pooled placement)      | 0.9747           | 0.9946           | 1.0000           |
| token (dual placement)        | 0.9793           | 0.9946           | 0.9996           |
| pooled mean (pooled placement)| 0.9571           | 0.9931           | 0.9805           |
| pooled mean (dual placement)  | 0.9663           | 0.9931           | 0.9759           |

The prediction was wrong, and the way it was wrong is the deepest result here. Averaging the
tokens over time barely dents any task: halfswap stays at about 0.98, not 0.5. The reason is
architectural. The time-mean of PatchTST tokens is not permutation-invariant, because positional
encoding and self-attention inject order into the token values before any pooling, so the averaged
feature still carries order information. So low across-time variance, which is what the collapse
diagnostic measures, does not mean the representation has lost temporal or order information. The
two are different properties. This refines the earlier partial-collapse explanation: it is not
mainly that pooled keeps a residual 6 percent of variance, it is that the order-relevant
information lives in the token values, which a positional transformer computes order-sensitively
whether or not the tokens vary across time.

To confirm that positional encoding is the carrier, we ablated it (the transformer over patches
is then permutation-equivariant) and reran the pooled-feature probe. Reproduce with:

```bash
uv run python scripts/classify.py --pems data/pems08.npz --seeds 3 --pool --no-pos-encoding
```

| pooled-feature condition | trend            | level            | halfswap         |
|--------------------------|------------------|------------------|------------------|
| with positional encoding | 0.9571           | 0.9931           | 0.9805           |
| without (ablation)       | 0.9218 / 0.9379  | 0.9900 / 0.9939  | 0.9337 / 0.8908  |

Removing positional encoding drops halfswap on the pooled feature from about 0.98 to about 0.90,
which confirms positional encoding supplies much of the order signal in the time-mean. It does not
fall to chance, for a concrete reason: the half-swap is `np.roll` by 48 over overlapping
stride-8 patches, which is not a clean patch permutation, so the wrap-around boundary leaves real
content differences that a linear probe still reads. The exact mechanism is nailed by the unit
tests instead: with non-overlapping, patch-aligned permutations, the encoder without positional
encoding is permutation-equivariant (so a mean over patches is permutation-invariant), and with
positional encoding it is not. So positional encoding is the order carrier in the clean setting,
and on the messy real-data swap it accounts for about half the gap to chance, the rest being
windowing content leakage.

## Architecture x placement factorial (the central result)

A factorial study over encoder architecture (positional PatchTST vs a position-free
bag-of-patches: non-overlapping patches, a shared per-patch MLP, no positional encoding and no
cross-patch attention) crossed with placement, three seeds, measuring the collapse diagnostics
and halfswap and trend accuracy from both the token and the pooled feature. Reproduce with:

```bash
uv run python scripts/architecture_study.py --pems data/pems08.npz --seeds 3
uv run python scripts/plot_study.py results/pems_architecture_study.json --outdir figures
```

| arch \| placement      | across-time var | eff rank      | halfswap token | halfswap pooled | trend token   | trend pooled  |
|------------------------|-----------------|---------------|----------------|-----------------|---------------|---------------|
| positional \| pooled   | 0.113 +- 0.018  | 6.615 +- 0.14 | 1.000 +- 0.000 | 0.980 +- 0.012  | 0.975 +- 0.004| 0.957 +- 0.001|
| positional \| dual     | 0.620 +- 0.006  | 8.485 +- 0.19 | 1.000 +- 0.001 | 0.976 +- 0.010  | 0.979 +- 0.000| 0.966 +- 0.002|
| bagofpatches \| pooled | 1.712 +- 0.135  | 1.449 +- 0.05 | 0.499 +- 0.008 | 0.500 +- 0.000  | 0.992 +- 0.003| 0.887 +- 0.010|
| bagofpatches \| dual   | 1.739 +- 0.091  | 1.605 +- 0.05 | 0.489 +- 0.007 | 0.500 +- 0.000  | 0.995 +- 0.001| 0.896 +- 0.002|

Figures in `figures/`: collapse.png, halfswap.png, trend.png.

This is the strongest finding of the project, and it overturns the assumption the project began
with. Read the columns against each other.

1. The collapse is set by architecture and placement together, not placement alone. On the
   positional encoder the placement controls it (across-time variance 0.11 pooled against 0.62
   dual). On the bag-of-patches encoder neither placement collapses: both sit near 1.7, because
   each token is an independent embedding of its own patch and so varies by content regardless of
   the objective.

2. Across-time variance does not measure order information, and here it is anticorrelated with it.
   The most collapsed configuration, positional pooled at variance 0.11, recovers a half-swap at
   0.98. The least collapsed configuration, bag-of-patches at variance 1.7, is at chance, 0.50.
   So the diagnostic the project is built around runs opposite to the thing it was meant to
   protect: high across-time variance coincides with order being lost, low variance with order
   being kept.

3. Order availability is determined by positional encoding, not by placement or by the collapse.
   The positional encoder recovers the half-swap at about 0.98 to 1.00 in every placement and from
   either feature, while the bag-of-patches encoder is at chance, 0.500 plus or minus 0.000 from
   the pooled feature, in every placement. The placement (pooled vs dual) moves the half-swap
   result by nothing. The bag-of-patches pooled feature hitting exactly 0.500 with zero variance
   is the clean proof: a genuinely position-free representation is exactly order-blind, and the
   probe detects it.

4. A prediction we made was refuted, usefully. We expected token features to retain order
   everywhere, since order could live in position-in-vector. The bag-of-patches token feature is
   also at chance on the half-swap. Without positional encoding the token values do not encode
   absolute position, and the half-swap is a cyclic rotation of the same set of tokens, which a
   linear probe cannot anchor. So even token-level order recovery needs positional encoding.

5. Effective rank dissociates from across-time variance too: bag-of-patches has the highest
   variance (1.7) and the lowest effective rank (about 1.5), because its per-patch embeddings vary
   across time but lie on a low-dimensional content manifold. Two geometric diagnostics, two
   different stories, neither tracking order.

The trend column is consistent: trend is a per-patch level comparison, available in token features
for both architectures (bag-of-patches is even cleanest at 0.99 because its tokens are undistorted
per-patch content), and degraded in the pooled feature that averages the halves together.

## External validity: the factorial on ETT

To check that the factorial result is not a quirk of 170-channel traffic data, we reran the same
study on ETTh1 (electricity transformer temperature: 7 channels, hourly, about 17000 steps), a
very different regime. Reproduce with:

```bash
uv run python scripts/architecture_study.py --ett data/ETTh1.csv --seeds 3
uv run python scripts/plot_study.py results/ett_architecture_study.json --outdir figures/ett
```

| arch \| placement      | across-time var | eff rank       | halfswap token | halfswap pooled | trend token   | trend pooled  |
|------------------------|-----------------|----------------|----------------|-----------------|---------------|---------------|
| positional \| pooled   | 0.045 +- 0.010  | 6.71 +- 1.28   | 0.561 +- 0.011 | 0.523 +- 0.014  | 0.877 +- 0.004| 0.700 +- 0.062|
| positional \| dual     | 0.554 +- 0.029  | 12.12 +- 0.57  | 0.608 +- 0.010 | 0.542 +- 0.023  | 0.895 +- 0.014| 0.756 +- 0.013|
| bagofpatches \| pooled | 0.525 +- 0.040  | 1.722 +- 0.059 | 0.484 +- 0.002 | 0.500 +- 0.000  | 0.973 +- 0.006| 0.546 +- 0.001|
| bagofpatches \| dual   | 0.878 +- 0.005  | 2.145 +- 0.057 | 0.513 +- 0.007 | 0.500 +- 0.000  | 0.967 +- 0.008| 0.538 +- 0.015|

Figures in `figures/ett/`. The central thesis replicates, with one probe carrying the signal
cleanly and a methodological subtlety exposed in the other.

The clean cross-dataset evidence on ETT is the trend task probed from the pooled feature. The
positional encoder retains the order needed for trend in its pooled feature (0.70 to 0.76, well
above chance), while the position-free bag-of-patches encoder does not (0.546 and 0.538, at
chance), because its pooled feature is permutation-invariant and trend asks which half is higher.
This is the same conclusion as PEMS: positional encoding, not the absence of collapse, is what
lets a pooled or collapsed representation keep order. The across-time-variance anticorrelation
also replicates: the most collapsed configuration, positional pooled at variance 0.045, keeps the
trend order at 0.70, while the higher-variance bag-of-patches loses it at 0.546. And the effective
rank dissociation replicates: bag-of-patches has low rank (about 1.7 to 2.1) regardless of its
variance.

The methodological subtlety is that the two order probes have dataset-dependent confounds, and
they trade places between the datasets. On ETT the halfswap probe is muted: ETT is hourly with
strong daily periodicity, and the half-swap rolls a 96-hour window by 48 hours, almost exactly two
daily periods, so the swapped window is nearly identical to the original and even the positional
encoder sits near chance (0.52 to 0.61). On PEMS the halfswap was clean (roll by four hours is not
a period multiple) but the trend probe was confounded: bag-of-patches reached 0.89 on trend from
the pooled feature there, because traffic level correlates with trend direction, whereas on ETT
that correlation is absent and bag-of-patches falls to the theoretically expected chance. So each
dataset has one clean order probe and one confounded or muted one, but the central thesis (order
availability is set by positional encoding and is orthogonal to, even anticorrelated with, the
across-time collapse) holds on both through the clean probe. The position-free pooled feature
hitting exactly 0.500 on halfswap on both datasets is the architecture-independent anchor.

### Forecasting on ETT: dual helps here, unlike PEMS

The forecasting comparison on ETT (positional encoder, trajectory probe, three seeds) flips the
PEMS downstream result. Reproduce with `uv run python scripts/compare.py --ett data/ETTh1.csv
--steps 500 --batch-size 32 --num-slices 32 --forecast trajectory --seeds 3`.

| placement | across-time var | eff rank      | trajectory MAE | trajectory MSE |
|-----------|-----------------|---------------|----------------|----------------|
| pooled    | 0.031 +- 0.006  | 10.9 +- 1.6   | 1.188 +- 0.059 | 2.369 +- 0.264 |
| dual      | 0.534 +- 0.026  | 16.8 +- 0.3   | 1.136 +- 0.020 | 2.117 +- 0.089 |

On ETT the dual placement forecasts better than pooled, about 4 percent lower MAE and 11 percent
lower MSE, with the MSE bands barely overlapping, the opposite of the PEMS forecasting result
where pooled was slightly ahead. The likely reason is that ETT forecasting at a twelve-step
horizon genuinely needs temporal structure (electricity load swings over the day), whereas the
PEMS short-horizon forecast was close to persistence, where the level a collapsed representation
keeps is already enough. So whether preventing the collapse helps forecasting is dataset
dependent, and it is not mainly about order recovery (positional encoding already supplies that)
but about representation richness: dual carries a higher effective rank (16.8 against 10.9) that
gives the linear head more to work with when the target is a structured trajectory rather than a
near-constant level.

## Conclusion and next steps

The central result, from the architecture factorial, is that the time-axis collapse diagnostic
(across-time variance) does not measure what the project assumed it measured. It does not track
downstream order availability, and across the factorial it is anticorrelated with it: the most
collapsed configuration keeps order and the least collapsed one loses it. What actually determines
whether order survives is positional encoding in the encoder, an axis orthogonal to the placement
that controls the collapse. So the dual placement, which robustly prevents the collapse, does not
recover order in either architecture: with a positional transformer the order is already present
whether or not the tokens collapse, and with a position-free encoder the order is absent and
preventing the collapse does not bring it back.

That said, the dual placement is not useless, and the two datasets together draw the line cleanly.
Dual carries a higher effective rank, a richer per-timestep representation, and whether that
richness helps downstream depends on the task. On PEMS short-horizon forecasting, which is close
to persistence, it does not, and pooled is slightly ahead. On ETT twelve-step forecasting, which
genuinely needs temporal structure, dual is better by about 4 percent MAE and 11 percent MSE. So
the honest synthesis is two-axis: preventing the collapse does not change order availability
(that is positional encoding), but it does enrich the representation, and that enrichment pays off
on downstream tasks whose targets are structured rather than near-constant. Preventing the
collapse is not a universal good and not the right lever for order, but it helps representation
richness where the task can use it.

The supporting investigation is consistent with this. The dual placement prevents the time-axis
collapse robustly and by a large margin on the positional encoder, and the lambda sweep shows the
placement, not the lambda, controls it. The utility claim does not hold on PEMS08. Across four honest downstream tests, forecasting
(mean and full-trajectory, multi-seed), forecasting swept over horizons 3 to 48, Mahalanobis
anomaly detection, and a lambda sweep, preventing the collapse gives no measurable benefit and is
mildly costly. We also ruled out our own leading explanation: the persistence hypothesis predicts
the forecasting gap should close at long horizons, and the horizon sweep shows it does not, so
pooled's edge is horizon-independent.

The lambda sweep adds the mechanism. SIGReg and the linear forecaster are in tension: raising
lambda raises effective rank but monotonically worsens dual's forecasting, so the isotropic
regularization that prevents the collapse is not what the downstream head rewards. That is why
preventing the collapse costs a little rather than being neutral. The sweep also tested LeJEPA's
label-free selection claim on real time series and did not confirm it: final SIGReg loss does not
track forecasting quality here (flat within pooled, reversed within dual), and the only positive
correlation across configs is a confound from the two placements having different SIGReg loss
scales. The overall picture on this benchmark: the time-axis collapse is real and large but
downstream-benign, the dual placement that fixes it trades slightly against forecasting, and
SIGReg loss is not a reliable label-free selector for this task.

The classification experiment supplied the first positive evidence for dual, on the subtle trend
task, and then a controlled follow-up reframed the whole study. The plan was that the content
matched halfswap task would be decisive, and that probing the pooled time-mean feature would
simulate full collapse and drop halfswap to chance. Neither happened: both placements classify
halfswap at ceiling, and the time-mean feature still reaches about 0.98. The reason is
architectural. The time-mean of PatchTST tokens is not permutation-invariant, because positional
encoding and attention write order into the token values before pooling. So the diagnostic the
project is built around, low across-time variance, does not entail loss of temporal or order
information: a representation can collapse along time and still carry the order the task needs.

That is the central finding, and it is sharper than the earlier partial-collapse story. Across six
tests the time-axis collapse is real and large in the across-time variance sense, the dual
placement robustly produces it, but it is downstream-benign on PEMS because the order-relevant
information lives in the token values, which a positional transformer computes order-sensitively
whether or not the tokens vary across time. Pooled therefore matches dual on forecasting at any
horizon, on anomaly detection, on level classification, and even on a gross position swap, and
dual edges ahead only on the one subtle order task. The SIGReg pressure that prevents the collapse
also trades slightly against the linear forecaster, and SIGReg loss is not a reliable label-free
selector here.

A positional-encoding ablation confirms this. Unit tests show the encoder without positional
encoding is exactly permutation-equivariant over patches, and removing it on PEMS drops the
pooled-feature halfswap accuracy from about 0.98 toward 0.90 (not all the way to chance, because
overlapping non-aligned patching leaks content at the swap boundary). So positional encoding is
the order carrier, and the across-time collapse is downstream-benign here precisely because that
order information lives in the token values rather than in how the tokens vary across time.

The honest takeaway for LeJEPA issue #27 is two-sided. The time-axis collapse is genuine and the
dual placement is a clean fix for it, but on time series encoded by a positional transformer the
across-time variance collapse is not by itself a downstream problem, because that architecture
retains order information independently of it. The collapse diagnostic is most likely to matter
for encoders without strong positional structure, or for tasks where order information must survive
in the geometry of the embedding rather than in token values. The cleanest remaining test is a
genuinely position-free encoder (non-overlapping patches, no positional encoding, no cross-patch
attention) where mean-pooling really would erase order and the collapse should finally bite;
longer training and a non-saturating anomaly detector are the other untested levers.
