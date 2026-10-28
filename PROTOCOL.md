# DBRL research decision and bounded comparison protocol

Version 1 — 28 October 2026

Status: implementation specification. No new training or scientific result is reported here. This is a development decision protocol written after inspecting earlier BRE outcomes, not a preregistration of an untouched test.

## 1. Decision now

Retain device-level representation learning as the candidate contribution. Use input reconstruction as the working training objective. Suspend further optimization of BSE, BIL, EBRL, forecasting and hierarchical anomaly-score fusion. Preserve their source, checkpoints, selections and results as historical experiments. Do not delete or silently rewrite them.

The proposed question is: Does explicit cross-device context improve the preservation of observable sensor information over a shared temporal encoder, under matched training and readout conditions?

Do not yet claim human-activity recognition, user identification, anomaly detection, or superiority of hierarchical abstraction. Do not describe reconstruction error as a validated anomaly score.

## 2. Two corrections that determine the experiment

The retained reconstruction checkpoint was produced with the full hierarchy and four reconstruction losses. Its DBRL parameters may reflect gradients from upper levels. It is a historical reference, not a fairly trained DBRL-only arm. The decisive comparison starts fresh.

Final binary state, rising transition and falling transition are deterministic functions of the observed input window. An exact rule can recover these labels from valid raw observations. These tasks test information accessibility and preservation, not discovery of latent human activities. Include the exact rules as a diagnostic ceiling; do not require DBRL to outperform them or advertise the proxy task as a standalone behavioral application.

The currently implemented residual-context DBRL uses dense endpoint-eligible connectivity, not the original top-k graph construction. Its forward implementation is the architecture to test and describe. Configuration keys retained for compatibility are not evidence that top-k selection executes.

## 3. Fixed experimental arms

| Arm | Encoder / input | Optimization | Role |
|---|---|---|---|
| Raw windows | Flatten the same six-feature inputs | None | Direct input reference |
| Simple summaries | Per-channel last valid value, valid-value mean/std, endpoint difference; validity indicators | None | Cheap reference; undefined summaries carry explicit masks |
| Exact state/event rules | Existing valid-pair target definitions | None | Diagnostic ceiling only |
| Initial temporal encoder | Shared GRU plus projection | None | Temporal initialization control |
| Initial DBRL | The same temporal encoder plus residual graph context | None | Graph initialization control |
| Trained temporal encoder | Shared GRU plus projection, no encoder graph operation | Reconstruction only | Main simpler learned baseline |
| Trained DBRL | ResidualContextDBRL | Reconstruction only | Candidate |

No BSE, BIL, EBRL or HBF participates in the new optimization. Both learned encoders expose N by D outputs with D=64 and use the same reconstruction decoder: flatten, Linear(N*D,64), GELU, Linear(64,W*N), reshape. This is the existing Z-decoder design. Its dense decoding mixes devices in both arms; the comparison isolates explicit graph context before decoding, not all possible cross-device interaction.

For each seed, initialize temporal GRU, projection and reconstruction decoder once and copy their exact tensors into both trained arms. Initialize graph parameters once for that seed; retain the exact corresponding initial DBRL. Do not migrate trained parameters from the historical checkpoint. Verify tensor equality at step zero.

Report encoder and decoder parameter counts separately. Equal output dimensions and common components do not imply equal parameter counts: graph parameters are additional capacity. A positive result would support this graph-enabled implementation, not prove topology is the sole cause.

## 4. Inputs, selections and provenance

Reuse the prepared BRE six-feature artifacts, W=30 seconds at 1-second sampling, their frozen training-fitted scaler, exact channel order, and validity semantics. Do not resample time, select new features or change filling during this comparison.

Prepared manifest SHA256:
`3d34c85214cbd1c6a8cda24b57b7c4f8df8abc5820877c80d13831433b12e3e2`.

Historical checkpoint SHA256 (provenance only, not an initialization constraint):
`98dd287013098294cb220a869b11a9f22240b3db585bce99b9d292ff5109e67e`.

Use the already frozen event selection: 1,024 update windows (525 event, 499 quiet) and 1,024 later-training readout-fit windows (216 event, 808 quiet). Retain update rows [0,1143042), fitting rows [1143102,1632960), the 60-row exclusion interval, episode grouping and all original window starts.

Reuse the frozen later Actor1 evaluation selections from hdeg_bre_frozen_temporal_eval_v1.zip: 1,024 uniform windows and 1,024 event-enriched windows (188 event, 836 quiet). Do not reconstruct a different selection from the same random seed. Resolve the exact saved manifests and record their hashes before any optimizer step. These two cohorts overlap in some windows: analyze separately, never concatenate as independent evidence.

All these periods have already informed development. Label them development evaluation even though models and readouts are frozen during inference. Actor2 evaluation is excluded from this decision to avoid another confounded actor-period optimization target.

No training on readout-fit or evaluation windows. Fit labels are allowed only for the explicitly supervised diagnostic readout. Verify source-row nonoverlap and timestamp chronology. Source-preparation missingness limitations remain applicable even if prepared status flags are all valid.

## 5. Fixed optimization budget

- Three seeds: 20261041, 20261042, 20261043.
- Two trained arms per seed; 1,024 optimizer updates per arm; batch size 8.
- Eight deterministic shuffled passes through the 1,024 update windows. Save the complete index schedule. Both arms receive identical minibatches for a given seed.
- Adam: learning rate 1e-4, betas (0.9,0.999), epsilon 1e-8, weight decay 0. No scheduler or early stopping. No gradient clipping.
- GRU hidden size 64, one layer, output dimension 64, dropout zero; residual DBRL one graph head and graph dropout zero.
- Loss: mean over windows of valid-value squared reconstruction error divided by the valid target count for that window. Reject windows with no valid target. No event-label auxiliary loss, class weighting or upper-level loss.
- Evaluate initialization and the final fixed update only for the decision. Training loss may be logged every 128 updates. Do not select a checkpoint by development evaluation performance.
- Fixed total: 6,144 optimizer updates. A numerical failure is reported, not repaired by silently changing learning rate, seed or budget. Infrastructure-only retries retain the same contract.

The longer budget than the previous 512-step BSE interventions is fixed here before new results. It is not equivalent to the historical encoder's cumulative multi-objective training.

## 6. Fair readout and compression controls

Use the existing diagnostic ridge objective: mean squared label error plus 1.0 times coefficient squared norm, unpenalized intercept. Fit standardization and readout coefficients only on readout-fit windows, with scale 1 for standard deviations at or below 1e-6. Keep the decision threshold 0.5. No threshold search or post-hoc score reversal.

Report native flattened learned embeddings, where both trained encoders expose the same N*64 dimensions. In addition, report a common 320-dimensional PCA view for raw input, initial temporal, initial DBRL, trained temporal and trained DBRL. Fit each PCA using update windows only, with deterministic full SVD, centering, no whitening and no label access. Retain at most 320 numerically nonzero components and zero-pad to 320; report numerical rank. Do not interpret zero padding as extra information. PCA rank tolerance: max(matrix shape)*float64 epsilon*largest singular value. Readout standardization remains fit-only after projection.

Native summary features and exact-rule results are reference rows, not parameter- or dimension-matched models. Report the compressed view as a diagnostic of compact information preservation; this PCA operation is not a replacement hierarchy or an end-to-end trained bottleneck.

## 7. Targets, coverage and metrics

Reuse the saved 12 training-eligible targets and validity definitions. Do not add or drop targets based on model scores. Retain positive, negative, missing and distinct-episode counts for every target/cohort.

The fixed transition decision set in the enriched later-period cohort is rise and fall for OC-00-H1_contact_x and A6-01-H2_motion (four targets). These are the targets with at least 20 positive and 20 negative windows in the previously reported evaluation. A6-00-K1_motion remains descriptive because of insufficient evaluation coverage. This target choice is based on already-seen coverage and must be disclosed.

Primary decision metric: equal-weight macro average precision over those four enriched-cohort transition targets. Report per-target AP, prevalence and the prevalence-only AP baseline. Report final-state balanced accuracy, per-target recall/precision at the frozen threshold, and reconstruction MSE separately. Do not hide low recall behind AP.

Uniform-cohort transition metrics remain descriptive when support is insufficient. Enrichment changes prevalence: do not compare enriched and uniform AP as if they measured performance under the same population distribution. Do not estimate natural anomaly rates from either proxy task.

## 8. Temporal uncertainty and stop rule

Keep all seeds and per-target outcomes, including failures. Use paired resampling of the same source calendar-day clusters for all methods: 1,000 bootstrap replicates, fixed seed 20261044. A sampled day includes all its selected windows, retaining episode groups. Compute each model's AP on the concatenated resampled windows, then the paired difference; average seed-specific differences without treating duplicated observations across seeds as independent samples.

Report percentile intervals, the number of distinct observed days, and replicates lacking both classes for any primary target. Do not substitute zero for undefined AP. If more than 5% of replicates are undefined, mark the interval unreliable and the continuation decision inconclusive. With approximately one week of observations these intervals describe limited temporal sensitivity, not broad population uncertainty.

Continue graph-enabled DBRL development only if ALL of the following hold:

1. Its mean native primary AP exceeds both the trained temporal baseline and its initial DBRL by at least 0.03 absolute AP, with a positive difference in each of the three seeds.
2. The paired day-bootstrap interval for each of those two mean differences has a lower endpoint above zero.
3. Its 320-dimensional primary AP exceeds the trained temporal and raw-PCA references by at least 0.03 on the seed mean; report all seed differences without selecting a favorable projection.
4. Its mean final-state balanced accuracy is no more than 0.02 below the trained temporal baseline on the fixed six enriched-cohort final-state targets.
5. Parameter counts, total training time, inference latency and peak memory are reported. A performance improvement is not automatically an efficiency improvement.

The 0.03 and 0.02 margins are pragmatic development thresholds fixed for this decision, not externally established scientific constants or a power calculation. No statistical significance claim should exceed the temporal and seed sampling design.

If these conditions fail: do not extend training or redesign the graph in this branch. If the temporal model is adequate, retain it as the practical baseline and drop the DBRL graph contribution. If both are weak, stop this representation-method claim. An inconclusive result is not success and does not authorize an automatic longer experiment.

Passing is permission to pursue independent task-level validation, not permission to claim anomaly detection or publishable novelty. The original hierarchy is not reactivated by a DBRL success.

## 9. Paper revision map

Working topic: Device-Level Representation Learning for Sparse Smart-Home Sensor Streams. This is a scope label, not a novelty or publication guarantee.

| Original emphasis | Revised treatment |
|---|---|
| Hierarchical behavioral abstraction | Historical hypothesis and measured limitations; no superiority claim |
| HBF forecasting | Documented development branch; suspend from main method |
| Device representations | Candidate method, tested against temporal and raw-input references |
| Reconstruction | Explicit training objective; distinguish representation quality from residual scoring |
| Behavior discrimination | Observable state/transition preservation only, unless independently annotated activities become available |
| Anomaly detection | Unvalidated downstream task; do not retain in the title as an established contribution |
| Four original validation levels | Replace with evidence questions appropriate to the revised method; do not mechanically complete obsolete levels |

Potential manuscript structure: problem and sparse-event setting; source/preprocessing and validity provenance; device encoder and reconstruction objective; matched baselines and temporal protocol; representation and compression results; negative hierarchy/forecasting ablations; limitations and downstream validation needs.

Do not draft favorable results, an abstract claiming improvements, or a final conclusion before this comparison. Even if proxy preservation improves, direct raw rules already solve these proxy targets. A methods paper still needs a meaningful downstream task or a defensible compression/robustness/efficiency contribution against strong baselines. That is a separate evidence requirement, not something this diagnostic manufactures.

## 10. Next implementation boundary

Implement one standalone runner, proposed as run_v4_bre_dbrl_decision.py, with only necessary utility modules. Preserve yaml.safe_load of configs/config.yaml, project_root_dir override, existing prepared-data loader and shared path helpers. Record a resolved experiment-specific settings snapshot without changing global configuration or old files.

Reuse and test the actual DBRL temporal encoding, ResidualContextDBRL, observed-target validity logic, event selection verifier and readout code. Add a local temporal-only wrapper, paired initialization checks, a Z-only decoder/loss and the comparison reporting. Do not call the full BREReconstructionModel training path, which instantiates upper levels and supports the old four-level objective.

Before optimization: resolve saved selections, verify their hashes and prepared-data identity, freeze the run contract, validate no source overlap, verify matched initial tensors, perform a bounded forward/backward test, and verify exact-rule target recovery. These are gates inside the one runner, not a new series of standalone audit projects.

Save final model states and initial-state hashes, schedule, selections, per-window scores, targets/masks, fitted PCA/readout parameters, coverage, source/configuration hashes, timing/memory, per-seed metrics, uncertainty and the explicit decision. Create a new experiment directory; never overwrite previous outputs.

This document completes the decision/specification step. The comparison runner has not been implemented or executed by this document.
