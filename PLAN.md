# PLAN.md

Build plan for ChronoJEPA. The project compares three placements of the SIGReg objective
to diagnose and fix the time-axis collapse described in LeJEPA issue #27. Run the phases in
order, one phase per session, and commit between phases so each phase starts from a clean,
known state.

## Repo layout

```
chronojepa/
  sigreg/   SIGReg objective: univariate tests, random slicing, placement variants
  models/   encoders (PatchTST, TCN), optional MLP predictor, RevIN
  data/     dataset loaders (PEMS first, then ETT and financial), augmentations
  train/    trainer and training loop
  eval/     linear and kNN probe, forecasting, Mahalanobis anomaly scoring, model selection
  utils/    seeding, device selection, logging
configs/    Hydra configs
scripts/    runnable entry points (init.sh, later train.py)
tests/      pytest suite
```

## Placement milestones (the core contribution)

The whole project is organized around three placements of SIGReg relative to the time axis.
These are the milestones the experiments must compare:

- [x] `pooled`: one SIGReg over the pooled sequence embedding. Baseline, expected to
  collapse to a constant per-sequence "ID vector" along time. Implemented and
  ground-truthed in Phase 1.
- [x] `dual`: SIGReg within each sequence across time, plus across samples in the batch.
  Expected to prevent the collapse. Implemented in Phase 4.
- [x] `structured`: a multivariate or joint formulation. Open research question. Phase 4
  ships an initial concrete instantiation: regularize the joint `(N * L, D)` set to
  N(0, I). The richer formulation remains open.

## Phase checklist

- [x] **Phase 0: Plan and scaffold.** Repo layout, pyproject managed by uv, ruff and pytest
  config, pre-commit, package skeleton with empty modules, a trivial passing test, and
  `scripts/init.sh`. Extended for publishing: MIT license, Python `.gitignore`, README,
  GitHub Actions CI, and this plan.
- [x] **Phase 1: SIGReg core, ground-truthed against scipy.** Epps-Pulley univariate test,
  random-slicing wrapper, and `PooledSIGReg`. Tests cross-check the statistic against scipy
  quadrature, confirm near-zero loss on `N(0, I)`, confirm gradient flow, and confirm CPU
  and MPS agreement.
- [x] **Phase 2: Encoders and RevIN.** PatchTST-style transformer encoder, a TCN baseline
  with a shared tensor contract `(B, C, T) -> (tokens (B, L, D), pooled (B, D))`, and RevIN
  with normalize and denormalize.
- [x] **Phase 3: Data and augmentations (PEMS first).** Time-ordered splits, sliding
  windows, a train-only `StandardScaler`, a config-driven two-view augmentation pipeline,
  and a `WindowDataset` plus `build_dataloaders` factory. No look-ahead bias: statistics
  fit on the train split only and applied forward. `load_pems` reads a user-downloaded
  `.npz`; the real download is left to the user and is not yet wired into a run.
- [x] **Phase 4: SIGReg placements and training loop.** `DualSIGReg` and `StructuredSIGReg`
  behind a common interface (`make_sigreg(name)`), the combined LeJEPA objective
  `lam * sigreg + (1 - lam) * invariance` with an optional MLP predictor, and a minimal
  device-agnostic, seeded training loop with cosine-warmup schedule and pluggable logging
  (offline `RunLogger` by default, opt-in `WandbLogger`).
- [x] **Phase 5: Collapse diagnostics and downstream evaluation.** Across-time variance and
  effective rank diagnostics, frozen-encoder linear and kNN probes, a forecasting head, and a
  Mahalanobis anomaly scorer, plus `run_placement_comparison`. On a synthetic multivariate
  run, pooled collapsed (across-time variance 0.022, effective rank 4.66) while dual did not
  (0.479, 8.58), and dual forecasting was at least as good (MSE 0.363 vs 0.393). PEMS numbers
  pending the dataset download.
- [x] **Phase 6: Label-free model selection.** `label_free_model_selection` ranks runs by
  final SIGReg loss and reports the Spearman correlation with the labeled downstream metric,
  the label-free pick versus the label-based pick, and whether they agree, with a thin CLI.
- [ ] **Phase 7: Experiment runner, configs, sweeps.** Hydra configs and named experiments
  (smoke, pems_pooled, pems_dual, pems_structured), `scripts/train.py` as the single entry
  point, and a sweep over lambda and the three placements.
- [ ] **Phase 8: README, reproducibility, and writeup.** Final README with the placement
  comparison result, and a RESULTS.md with the comparison table and plots.

## Definition of done (every phase)

ruff passes, the targeted tests pass, new behavior has a test, and a smoke run completes
without error. Results are reported as facts: what ran and what passed.
