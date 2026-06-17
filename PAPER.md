# When Does the Time-Axis Collapse Matter? Disentangling Collapse, Positional Structure, and Representation Richness in SIGReg-Regularised Time-Series Representations

Rishabh Patil. University of St Andrews.

Draft working paper. All numbers in this document are produced by the scripts in this
repository and are reproducible from the commands quoted in each section. Standard deviations
are over random seeds and confidence intervals are 95 percent bootstrap intervals on paired
seed-aligned differences.

## Abstract

LeJEPA (arXiv:2511.08544) replaces the heuristic machinery of self-supervised learning with a
single isotropic-Gaussian regulariser, SIGReg, applied with no stop-gradient, no teacher, and no
schedulers. Carrying SIGReg to sequence models raises a documented hazard (LeJEPA issue #27): a
naive pooled application can drive each sequence to a constant vector along time, a time-axis
collapse that the objective does not penalise. We build a self-supervised time-series library
around this question and compare three placements of SIGReg relative to the time axis: pooled (the
baseline), dual (SIGReg within each sequence across time plus across the batch), and structured (a
joint variant). We confirm that the collapse is real and that the placement controls it: dual
robustly raises across-time variance by roughly an order of magnitude relative to pooled.

Our central finding is negative and, we argue, clarifying. Through a factorial study that crosses
encoder architecture with placement and probe feature, we show that the across-time variance
collapse does not measure downstream order availability, and in our experiments it is
anticorrelated with it. The determinant of whether a representation retains temporal order is the
encoder's positional structure (positional encoding and attention, or convolution), an axis
orthogonal to the placement that controls the collapse. A position-free encoder yields a
permutation-invariant pooled feature whose order-recovery accuracy is exactly 0.500 on a
content-matched probe on two datasets, while a positional transformer, even when maximally
collapsed, recovers order at 0.97 to 1.00. The result holds across three architectures
(positional transformer, temporal convolution, position-free bag of patches) and two datasets.

The dual placement is nonetheless useful where the task needs the richer representation it
produces. On a real labeled task, UCI HAR activity recognition, dual outperforms pooled by 8 to 11
accuracy points under both linear and nonlinear probes. On forecasting, with eight seeds and paired
significance tests, dual is significantly better on ETTh2 and ETTm1, significantly worse on PEMS,
and not different on ETTh1. We therefore propose a two-axis account: preventing the time-axis
collapse does not change order availability (set by positional structure) but increases
representation richness (higher effective rank), which helps downstream tasks whose targets are
structured rather than near-constant. The practical recommendation for SIGReg on time series is to
treat the across-time collapse diagnostic with caution on positional encoders, and to motivate the
dual placement by representation richness rather than by collapse prevention per se.

## 1. Introduction

Self-supervised representation learning has converged on a small number of recurring tricks:
asymmetric encoders, stop-gradients, momentum teachers, predictor heads, and carefully tuned
schedules. LeJEPA argues that most of this can be replaced by a single, well-motivated objective.
SIGReg regularises a batch of embeddings toward an isotropic Gaussian by lifting a univariate
normality test to many dimensions through random one-dimensional projections, and pairs it with a
plain invariance term across two augmented views. There is no stop-gradient, no teacher-student
pair, no exponential moving average, and no scheduler. This minimalism is attractive, and it is
natural to ask whether it transfers to time series, where the dominant encoders are patching
transformers and temporal convolutions and where the relevant structure lives along an explicit
time axis.

A specific hazard motivates this work. When SIGReg is applied to a single pooled embedding per
sequence, nothing in the objective prevents each sequence from converging to a constant vector
along time while the pooled batch still looks Gaussian. The objective reports a low loss, yet the
per-timestep representation has degenerated. This time-axis collapse is described in LeJEPA issue
#27. The obvious fix is to apply SIGReg within each sequence across time, so that the per-timestep
embeddings of a single window are themselves pushed toward an isotropic Gaussian and cannot all
coincide. We call this the dual placement, against the pooled baseline, and we add a structured
joint variant.

We set out to measure which placement best prevents the collapse and yields the best downstream
representations. We confirm the first half quickly: the collapse is real and the dual placement
prevents it. The second half is where the work becomes interesting, because a sequence of
falsifiable predictions about downstream benefit are refuted by the data, and the refutations
compose into a cleaner account than the one we began with.

Contributions.

1. A heuristics-free SIGReg time-series library with three placements behind a common interface,
   univariate normality tests ground-truthed against scipy, and a battery of collapse diagnostics
   and frozen-encoder probes, all reproducible and seeded.
2. The empirical finding that the across-time variance collapse diagnostic does not track
   downstream order availability and is anticorrelated with it, established by a factorial over
   three encoder architectures, two placements, and two probe features, on two datasets, with a
   permutation-invariance proof and an exact chance-level anchor.
3. The identification of positional encoding as the actual determinant of order availability, with
   a positional-encoding ablation and a temporal-convolution control that places the effect on a
   graded architectural axis.
4. A two-axis account separating order availability (architecture) from representation richness
   (placement), supported by a real labeled classification task (UCI HAR) and by multi-dataset
   forecasting with paired significance tests, including a corrected overclaim where a three-seed
   advantage did not survive eight seeds.
5. A negative result on label-free model selection by SIGReg loss for time-series forecasting, and
   an analysis of why the naive cross-configuration correlation is confounded.

## 2. Background and related work

Joint-embedding predictive architectures and non-contrastive SSL. Modern non-contrastive methods
prevent representational collapse through architectural asymmetries (a momentum teacher, a
predictor, a stop-gradient) or through explicit variance and covariance penalties. LeJEPA belongs
to the latter family but replaces the bespoke penalties with a single objective, SIGReg, that
targets an isotropic Gaussian, with a theoretical argument that the isotropic Gaussian is the
optimal prior for downstream-agnostic embeddings.

SIGReg and sliced normality testing. Testing multivariate normality directly is hard, so SIGReg
projects embeddings onto random unit directions and applies a univariate test per slice, averaging
the result. We use the Epps-Pulley test, which integrates the squared difference between the
empirical characteristic function of the samples and the characteristic function of the standard
normal. We deliberately compare against the standard normal rather than standardising per slice,
because the regulariser targets N(0, I) and must therefore see deviations of location and scale.

Time-series encoders. PatchTST patches the time axis into overlapping windows, embeds each patch,
adds positional encoding, and processes the patch sequence with a transformer, treating channels
independently. Temporal convolutional networks stack dilated causal convolutions. Both are
positional in the sense that the representation of a position depends on order, the transformer
through positional encoding and attention and the TCN through the convolutional receptive field.
As a deliberate contrast we introduce a position-free bag-of-patches encoder that embeds
non-overlapping patches independently with a shared multilayer perceptron, with no positional
encoding and no cross-patch interaction, so that its pooled feature is permutation-invariant by
construction.

The time-axis collapse. The collapse this paper studies is specific to sequence models under a
pooled objective and is the time-series analogue of dimensional collapse. Our contribution is not
to rediscover that it exists but to ask whether the diagnostic for it, low variance of the
embedding along time, is the quantity that downstream performance actually depends on.

## 3. Method

### 3.1 The SIGReg objective

Given a batch of embeddings `Z` of shape `(N, D)`, SIGReg samples `K` unit-norm directions, forms
the projections `P = Z W` of shape `(N, K)`, and for each slice computes the Epps-Pulley statistic
of the `N` projected values against the standard normal characteristic function. The per-slice
statistic integrates `(C(t) - exp(-t^2 / 2))^2 + S(t)^2` over `t` in `[0, t_max]`, where `C` and
`S` are the empirical cosine and sine transforms; the integrand is even in `t`, so we integrate
over the half line and double. The univariate test is validated against scipy adaptive quadrature
to a stated tolerance, gradients are confirmed to flow, and CPU and Apple MPS results agree. The
slice statistics are averaged to a scalar.

### 3.2 The three placements

Let an encoder map a window to per-timestep token embeddings `T` of shape `(B, L, D)` and a pooled
embedding by averaging over time.

Pooled. SIGReg is applied to the pooled embedding, `(B, D)`. This is the baseline and the source of
the collapse: nothing constrains the `L` tokens of a sequence to differ.

Dual. SIGReg is applied within each sequence across time, treating the `L` tokens of a sequence as
`N = L` samples and asking them to look like N(0, I), plus across the batch on the pooled
embeddings. The within-sequence term is large precisely when a sequence collapses to a constant
vector, which is the mechanism that prevents the collapse.

Structured. A joint variant that applies SIGReg to all per-timestep embeddings stacked as
`(B * L, D)`. We include it for completeness; it is an initial instantiation of an open question
rather than a tuned method.

### 3.3 The combined objective

The training loss is `lambda * sigreg + (1 - lambda) * invariance`, a single tradeoff knob, where
the invariance term pulls the pooled embeddings of two augmented views together, optionally through
a small predictor. There is no stop-gradient, no teacher, and no EMA. The loop is device-agnostic
and seeded, with a cosine schedule and warmup.

### 3.4 Encoders

We use three encoders that share a single tensor contract, mapping `(B, C, T)` to tokens
`(B, L, D)` and a pooled embedding `(B, D)`. PatchTST is the positional transformer. The TCN is a
stack of dilated causal convolutions, positional through its receptive field. The bag-of-patches
encoder embeds non-overlapping patches independently with a shared MLP and is position-free: we
prove and unit-test that it is permutation-equivariant over patches, so that permuting input
patches permutes output tokens identically and the pooled mean is permutation-invariant. We also
expose a switch to remove positional encoding from PatchTST, after which its transformer over
patches is permutation-equivariant, which we likewise unit-test.

### 3.5 Collapse diagnostics

Across-time variance is the mean over batch and feature dimensions of the variance of the token
embedding along time. It is near zero when each sequence is constant over time, the signature of
the collapse. Effective rank is `exp(entropy(normalised singular values))` of the embedding matrix
and measures how many dimensions the representation occupies.

### 3.6 Probes and order tasks

All downstream evaluation uses a frozen encoder. We extract either the pooled feature `(N, D)` or
the flattened token sequence `(N, L D)`, and probe with a linear model (logistic or linear
regression) and, to guard against linear-readout artifacts, a nonlinear multilayer perceptron.

To measure order availability cleanly we construct two label sets from the windows themselves.
Trend labels whether the second half mean exceeds the first half mean, a temporal-order property.
Halfswap classifies a window against the same window with its two halves swapped, which leaves the
exact value multiset unchanged and varies only temporal position; it is content-matched and
isolates position. Forecasting probes a linear head on frozen features either against the
next-horizon per-channel mean or against the full next-horizon trajectory, the latter depending on
temporal structure. Splits are time-ordered and normalisation statistics are fit on the train
split only, so there is no look-ahead.

## 4. Experimental setup

Datasets. PEMS08 traffic flow (170 sensors, five-minute sampling), the ETT family ETTh1, ETTh2
(hourly) and ETTm1 (fifteen-minute), each with seven channels, and UCI HAR human activity
recognition (nine inertial channels, 128-step windows, six activities, about 7350 train sequences).
PEMS and ETT are long continuous series cut into sliding windows; HAR is a collection of labeled
fixed-length sequences. Datasets are downloaded by the user and are not redistributed here.

Training and evaluation. Unless stated, encoders use d_model 32 or 64 as noted per experiment, two
transformer layers, four heads, patch length 16 and stride 8, 32 random slices, lambda 0.5, the
two-view augmentation (jitter, scaling, time masking), 500 optimisation steps, and AdamW with a
cosine schedule. Probes are fit on the frozen train features and evaluated on the held-out split.
We report mean and standard deviation over seeds, and for the headline forecasting comparison we
add a paired t-test and a 95 percent bootstrap confidence interval on the seed-aligned
dual-minus-pooled difference.

## 5. Results

### 5.1 The collapse is real and is controlled by the placement

On a synthetic multivariate series the pooled baseline collapses (across-time variance 0.022,
effective rank 4.66) while dual does not (0.479 and 8.58). On PEMS08 across five seeds the same
holds and is tight: pooled variance 0.064 plus or minus 0.007 against dual 0.607 plus or minus
0.008, with non-overlapping bands, and effective rank 8.1 against 11.5. The placement controls the
collapse as intended.

### 5.2 The collapse is downstream-benign for PEMS forecasting

We expected dual to forecast better. It does not on PEMS. With a temporally sensitive trajectory
probe over five seeds, pooled is reliably slightly better (MAE 0.440 against 0.457). We then tested
our own explanation, that PEMS short-horizon forecasting is near persistence so the level a
collapsed representation keeps suffices, by sweeping the horizon. The explanation predicts the gap
should close at long horizons; it does not. Pooled stays about 0.005 to 0.007 MAE better at every
horizon from 3 to 48, flat. A Mahalanobis anomaly-detection comparison on token features saturates
for both placements (AUROC near 1.0 for spike, full-shuffle, and block-permutation anomalies),
because even a collapsed encoder remains content-sensitive. Across these probes the collapse is
real and large but downstream-benign on PEMS.

### 5.3 The factorial: across-time variance does not track order

The central experiment crosses architecture with placement and reports the collapse diagnostics
and the halfswap and trend accuracies from both the token and the pooled feature. On PEMS08, five
seeds:

| arch \| placement      | across-time var | eff rank      | halfswap token | halfswap pooled | trend token   | trend pooled  |
|------------------------|-----------------|---------------|----------------|-----------------|---------------|---------------|
| positional \| pooled   | 0.111 +- 0.015  | 6.47 +- 0.68  | 1.000 +- 0.000 | 0.977 +- 0.011  | 0.977 +- 0.005| 0.953 +- 0.006|
| positional \| dual     | 0.612 +- 0.011  | 8.31 +- 0.32  | 1.000 +- 0.001 | 0.973 +- 0.016  | 0.982 +- 0.004| 0.964 +- 0.004|
| tcn \| pooled          | 1.467 +- 0.217  | 8.23 +- 0.77  | 0.931 +- 0.032 | 0.522 +- 0.015  | 0.969 +- 0.011| 0.862 +- 0.019|
| tcn \| dual            | 5.247 +- 1.633  | 6.25 +- 0.92  | 0.982 +- 0.007 | 0.651 +- 0.029  | 0.960 +- 0.012| 0.888 +- 0.021|
| bagofpatches \| pooled | 1.661 +- 0.131  | 1.45 +- 0.04  | 0.502 +- 0.016 | 0.500 +- 0.000  | 0.993 +- 0.003| 0.887 +- 0.008|
| bagofpatches \| dual   | 1.697 +- 0.093  | 1.62 +- 0.04  | 0.489 +- 0.006 | 0.500 +- 0.000  | 0.996 +- 0.002| 0.897 +- 0.003|

Read the columns against each other. The position-free bag-of-patches pooled feature recovers a
content-matched half-swap at exactly 0.500 plus or minus 0.000, chance, because its pooled feature
is permutation-invariant. The positional transformer recovers it at 0.977 even in its most
collapsed configuration, across-time variance 0.111. The across-time variance is therefore
anticorrelated with order recovery: the lowest-variance configuration keeps order best and the
high-variance bag-of-patches keeps none. The TCN is intermediate, retaining order in its token
features (0.93 to 0.98) and partially in its pooled feature (0.52 to 0.65), which places order
availability on a graded axis set by how globally the architecture mixes position: attention and
positional encoding keep the most through pooling, convolution some, a position-free bag none.
Effective rank dissociates from across-time variance as well: bag-of-patches has high variance but
the lowest rank. Two geometric diagnostics, neither tracking order. Figure `figures/halfswap.png`
shows the half-swap result.

### 5.4 Simulated full collapse and the positional-encoding ablation

To test whether the collapse, taken to its limit, would destroy order, we probed the pooled
time-mean feature directly, which simulates a fully collapsed representation. The prediction was
that the time-mean is permutation-invariant, so half-swap should fall to chance. It did not: the
time-mean feature still recovers half-swap at about 0.98 on PEMS. The reason is architectural. The
time-mean of PatchTST tokens is not permutation-invariant, because positional encoding and
attention write order into the token values before any pooling. We confirmed the mechanism by
ablating positional encoding: with non-overlapping, patch-aligned permutations the encoder without
positional encoding is permutation-equivariant (unit-tested), and removing positional encoding on
PEMS drops the pooled-feature half-swap from about 0.98 toward 0.90. It does not reach chance only
because the half-swap rolls overlapping stride-8 patches, which leaks content at the boundary; the
clean chance result is the bag-of-patches encoder. So positional encoding, not the absence of
collapse, is what lets a pooled or collapsed representation keep order.

### 5.5 External validity: ETT and multi-dataset forecasting with significance

The factorial replicates on ETTh1, a different domain. The position-free pooled feature is again at
exactly 0.500 on half-swap, and the clean order probe on ETT is trend from the pooled feature,
where the positional encoder keeps order (0.72 to 0.77) and the position-free encoder does not
(0.54, chance). One methodological subtlety surfaced: the two order probes trade confounds between
datasets. On ETT, half-swap is muted because hourly data has daily periodicity and the 48-step roll
is almost two periods, nearly a no-op; on PEMS, trend was inflated because traffic level correlates
with trend direction, a correlation absent on ETT. Each dataset has one clean probe; the conclusion
holds on both, anchored by the exact 0.500.

For forecasting we ran eight seeds on all four datasets at identical settings (d_model 64) and
report paired significance on the dual-minus-pooled MAE gap:

| dataset | pooled MAE | dual MAE | gap (dual-pooled) | p      | 95% CI on gap     | verdict             |
|---------|------------|----------|-------------------|--------|-------------------|---------------------|
| PEMS08  | 0.4421     | 0.4563   | +0.0142           | 0.001  | [+0.009, +0.019]  | pooled better (sig) |
| ETTh1   | 1.1833     | 1.1614   | -0.0219           | 0.52   | [-0.086, +0.033]  | no difference       |
| ETTh2   | 0.8589     | 0.7378   | -0.1211           | 0.0004 | [-0.156, -0.088]  | dual better (sig)   |
| ETTm1   | 0.6337     | 0.5843   | -0.0494           | <1e-4  | [-0.056, -0.043]  | dual better (sig)   |

The significance pass corrected an overclaim. At three seeds dual looked about one standard
deviation better on ETTh1; at eight seeds the gap is not significant. The robust results are: dual
significantly better on ETTh2 (about 14 percent) and ETTm1 (about 8 percent), pooled significantly
better on PEMS, and no reliable difference on ETTh1. Two caveats on the mechanism, from the ETTh1
horizon sweep: the benefit is capacity gated, vanishing into seed noise at d_model 32 and appearing
only at d_model 64; and contrary to our prediction it does not grow with horizon, the gap being
largest at horizon 3 (-0.070) and shrinking to -0.038 at horizon 48, because long horizons revert
toward the periodic mean where level matters more. Dual is also consistently the more stable
representation, with several times lower seed variance on every ETT variant.

### 5.6 A real labeled task: HAR activity recognition

The order probes above are synthetic. UCI HAR is a real labeled, order-dependent task in a third
domain. We pretrain self-supervised on the train sequences and probe the frozen features, five
seeds:

| placement | linear pooled  | MLP pooled     | linear token   | MLP token      | across-time var |
|-----------|----------------|----------------|----------------|----------------|-----------------|
| pooled    | 0.576 +- 0.008 | 0.633 +- 0.014 | 0.636 +- 0.009 | 0.679 +- 0.016 | 0.019 +- 0.003  |
| dual      | 0.655 +- 0.023 | 0.718 +- 0.021 | 0.740 +- 0.024 | 0.788 +- 0.023 | 0.602 +- 0.052  |

This is the clearest case where preventing the collapse helps. Dual beats pooled by 8 to 11
accuracy points on every probe, with non-overlapping bands, and the advantage holds under both a
linear and a nonlinear MLP probe, so it is not an artifact of linear readout, and from both the
pooled and the token feature. On this task pooled is heavily collapsed (across-time variance 0.019)
and classifies poorly, while dual is uncollapsed (0.602) and classifies well. Figure
`figures/har_classification.png`. We note that order availability and downstream benefit are not in
conflict here: HAR is a task whose temporal structure the richer dual representation supplies.

### 5.7 Lambda sweep and label-free model selection

Sweeping lambda across placements (PEMS, three seeds) shows two things. Across-time variance is
controlled by the placement, not lambda: dual stays near 0.62 and pooled near 0.10 for all lambda,
with lambda only nudging within a placement. And SIGReg is in mild tension with forecasting:
raising lambda raises effective rank but monotonically worsens dual's forecasting MAE (0.3495 to
0.3559 from lambda 0.1 to 0.9), so the isotropic regularisation is not what the linear head rewards.
We also tested LeJEPA's label-free model selection claim, that final SIGReg loss tracks downstream
performance. It does not transfer here. The cross-configuration Spearman correlation is 0.552
(p = 0.098, not significant), and the label-free pick disagrees with the label-based pick. The
positive correlation is moreover a confound, since the dual loss sums two terms and is on a
different scale from pooled; within a placement the relationship is flat for pooled and reversed for
dual. SIGReg loss is not a reliable label-free selector for this task.

## 6. Discussion

The results compose into a two-axis account. The first axis is order availability: whether a
downstream linear or nonlinear readout can recover temporal order from the representation. We find
this is governed by the encoder's positional structure and is essentially untouched by the
placement or by the across-time collapse. A positional transformer keeps order even when maximally
collapsed; a position-free encoder loses it regardless of placement; a temporal convolution sits in
between. The collapse diagnostic, across-time variance, does not measure this axis and in our
experiments runs opposite to it.

The second axis is representation richness, for which effective rank is a reasonable proxy. The
dual placement increases richness, and whether that richness helps depends on whether the task
needs it. On a near-persistence forecast (PEMS) it does not and pooled is slightly better. On
structured forecasts (ETTh2, ETTm1) and on a real order-dependent classification task (HAR) it
helps, significantly. The benefit is capacity gated and, for forecasting, concentrated at short
horizons.

For SIGReg on time series this reframes the practical guidance. The time-axis collapse described in
issue #27 is genuine, and the dual placement is a clean fix for it, but on the positional encoders
that dominate the field the collapse is not by itself the right quantity to monitor, because the
information downstream tasks need is written into token values by positional encoding independently
of how the tokens vary across time. The dual placement should be motivated by the richness it adds,
and adopted where the downstream task is structured enough to use it, rather than as a universal
remedy for a diagnostic that does not predict downstream quality.

## 7. Limitations and threats to validity

The forecasting evaluation uses a frozen-feature linear probe rather than the standard
lookback-to-horizon protocol with a tuned baseline, so the absolute errors should be read as
representation-quality measurements and not as state of the art. The five datasets span three
domains, and the three ETT variants are closely related, so external validity rests mainly on the
HAR and PEMS contrasts and the ETT family as a unit. Training is fixed at 500 steps and is not
shown to be converged. The richness-to-benefit link is correlational through effective rank and is
not causally isolated. The synthetic order probes have dataset-dependent confounds, which we
document and mitigate by reporting the exact chance anchor and by using the clean probe per dataset.
Finally, the HAR study uses the positional encoder only; we did not run the full architecture
factorial on HAR.

## 8. Conclusion and future work

We extended SIGReg to time series, confirmed the time-axis collapse and that the dual placement
prevents it, and then showed through a factorial across three architectures and two datasets that
the collapse diagnostic does not measure downstream order availability and is anticorrelated with
it, the true determinant being positional encoding. We separated order availability (architecture)
from representation richness (placement) and showed, with significance tests and a real labeled
task, that the dual placement helps where the task needs richness (HAR, ETTh2, ETTm1) and not where
it does not (PEMS, ETTh1). Future work should test the two-axis account with a position-free encoder
on a richness-sensitive task where the collapse should genuinely bite, run the architecture
factorial on HAR and further real labeled datasets, anchor forecasting to standard protocols and
baselines, and probe the richness-to-benefit link causally by controlling effective rank.

## Reproducibility

The library, configs, tests, and the scripts that produce every table above are in this repository.
Each results section quotes the exact command. Datasets are downloaded by the user. Random seeds are
fixed and the resolved configuration is saved per run. The unit tests include the
permutation-equivariance and permutation-invariance proofs underpinning Section 5.3, the
scipy ground-truthing of the SIGReg statistic, and structural checks on every experiment runner.

## References

LeJEPA. arXiv:2511.08544. https://arxiv.org/abs/2511.08544

PatchTST. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023.

PEMS traffic datasets, from the ASTGCN repository.

ETT datasets, from the Informer / ETDataset repository.

UCI HAR. Human Activity Recognition Using Smartphones. UCI Machine Learning Repository.

Epps and Pulley. A test for normality based on the empirical characteristic function. Biometrika,
1983.
