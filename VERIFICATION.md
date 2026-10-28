# Implementation verification

Completed 28 October 2026. This file reports software verification, not new scientific results.

## Automated checks

14 tests pass. They cover exact common temporal/decoder initialization, seed recreation, all eight complete training passes, a forward/backward gate and two actual optimizer updates for each encoder on synthetic data, finite updated parameters, unchanged state during extraction, multi-target ridge equivalence to the existing individual readout, PCA centering/rank/zero-padding, missing-value target semantics, summary validity indicators, paired day resampling without seed pooling, undefined replicate handling, stop-rule conditions, invalid input budgets, undefined macro handling, and saved readout replay without evaluation-label leakage.

The two-update synthetic integration checks are not abbreviated scientific experiments and do not modify the public fixed training budget.

## Real prepared BRE artifacts

Executed the runner with `--verify_only` using the exact prepared BRE manifest and frozen training/evaluation selections.

- Status: `VERIFIED_ONLY`.
- Decision: `NOT_RUN`.
- Optimizer updates: **0**.
- Windows checked: **4,096** — update, fit, uniform and event-enriched, 1,024 each.
- All actual targets, validity masks, source ranges and timestamps matched the saved artifacts.
- Independent exact-rule target computation matched the vectorized target utility.
- Three seeds passed paired-initialization and real-data forward/backward gates for both arms.
- Source, metadata and consumed shard hashes remained unchanged.
- All **9** recorded output hashes were independently verified after execution.

Encoder parameters: temporal **17,984**, DBRL **22,272**. Both decoders: **374,916** parameters. Encoder output dimensions match; total parameter counts do not.

The first local verification attempt revealed that training target NPZ files store only `y`, `valid` and `starts`; source coordinates reside in their selection JSON. The implementation now verifies training coordinates against those JSON records and evaluation coordinates against both records and NPZ. The successful rerun used unchanged data and selections. Only its artifacts are delivered.

## Environment and limits

Verified on CPU with Python 3.12, PyTorch 2.14.1+cu130 and PyTorch Geometric 2.8.1. Exact environment details are in the saved report. CUDA execution was not tested. A PyTorch/PyG deprecation warning about `torch.jit.script` did not affect the checks.

The full six-arm, 6,144-update comparison has not been executed in this delivery. There is no new AP, anomaly-detection result or decision to continue DBRL development. Running the normal command performs the full experiment and includes all verification gates.
