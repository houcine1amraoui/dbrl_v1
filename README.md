# Standalone BRE DBRL decision comparison

This package implements `run_v4_bre_dbrl_decision.py` and only directly needed utilities. It adds no training path to the standard HDEG model and does not alter the old hierarchy or runners. The unchanged event-selection verifier is included so this package works even if that earlier utility has not been installed.

## Install

Extract this ZIP into the existing project/code directory containing `configs/` and `src/`. Preserve the included directory structure. Existing BRE preprocessing, DBRL, graph, reconstruction-runtime and behavior-target modules are dependencies; this is not a replacement codebase.

Run from that code directory. Configuration still comes from `configs/config.yaml` through `yaml.safe_load`; `--project_root_dir` overrides its root, and the shared folder helper chooses the output location. The configuration must identify BRE, 1-second sampling and window size 30. The fixed experiment settings are recorded separately and do not change the YAML.

Dependencies: your existing PyTorch/PyTorch Geometric/NumPy/PyYAML environment, plus `threadpoolctl`. Tests additionally use `pytest`. If missing:

```python
%pip install threadpoolctl
```

## Run the comparison

`BRE_INPUT_DIR` must point to the PREPARED BRE directory containing `manifest.json`, `devices.json`, `scaler.json` and the four split directories. Do not point it at BREMaster.csv, the old CU artifacts, or a single split.

```python
# Keep your existing PROJECT_ROOT_DIR and prepared BRE_INPUT_DIR values.
import subprocess
import sys

subprocess.run([
    sys.executable, '-m', 'run_v4_bre_dbrl_decision',
    '--project_root_dir', str(PROJECT_ROOT_DIR),
    '--input_dir', str(BRE_INPUT_DIR),
    '--selection_report', 'frozen_inputs/train/selection_report.json',
    '--evaluation_report', 'frozen_inputs/evaluation/evaluation_report.json',
    '--device', 'cpu',
], check=True)
```

The compact `frozen_inputs/` bundle contains exact copies of the already selected windows and directly consumed provenance files. It does not contain the raw/prepared data or historical model checkpoints. Supplying your own original report paths is also supported if their companion artifacts remain intact. The exact selection hashes are enforced: this runner does not resample windows.

CPU is the tested execution path. You may select `cuda`; CUDA was not tested in this environment. Deterministic algorithms are enforced. If your installed CUDA/PyG stack rejects a nondeterministic operation, use CPU rather than disable the contract's determinism requirement.

No old checkpoint argument is needed: all six trained arms begin from fresh, matched initializations. Historical checkpoint hashes identify where the frozen selection came from, not the new weights.

## What executes

- Three fixed seeds, two trained arms per seed, 1,024 updates per arm: 6,144 optimizer updates total.
- Batch 8, eight shuffled passes over 1,024 update windows; Adam 1e-4; no early stopping, clipping, scheduler or hyperparameter search.
- 1,024 later-training readout-fit windows; 1,024 later Actor1 uniform and 1,024 event-enriched evaluation windows.
- Temporal-only and residual-context DBRL with matched temporal initialization and identical reconstruction decoders.
- Raw six-feature inputs, simple summaries, exact-rule ceiling, initial temporal and initial DBRL references.
- Native readouts and update-fitted 320-dimensional PCA controls; fit-only standardization and ridge=1; fixed threshold 0.5.
- Per-target AP, balanced accuracy, recall, precision, prevalence, reconstruction MSE and representation variance.
- 1,000 paired calendar-day bootstrap replicates and explicit continuation/stop/inconclusive decision.

PCA and readout fitting run on CPU even when neural training uses CUDA. Full SVD may be a substantial part of runtime. No speed or peak-memory estimate is claimed before the full experiment. Parameter counts and timing are saved. CPU RSS is process lifetime high-water memory, not an isolated measurement for each arm; CUDA peak allocated memory is recorded per training arm when available.

## Optional verification-only execution

Adding `--verify_only` runs all source/selection/input checks and forward/backward gates for all three seeds, then exits with `VERIFIED_ONLY` and `NOT_RUN`. It creates no optimizer and does not train, fit readouts or produce a scientific decision. It is optional: the full command already includes those checks.

Do not use `VERIFIED_ONLY` as evidence that the comparison completed. Only `COMPLETE` with exactly 6,144 optimizer updates represents a full execution. An unsuccessful scientific result still has execution status `COMPLETE`; inspect the separate decision.

## Outputs

A unique directory is created under:

`data/processed/BRE/v2_validation/bre_dbrl_decision/<run_id>/`

Key files:

- `decision_report.json`: execution status, per-seed results, decision inputs, resource measurements and explicit outcome.
- `experiment_contract.json`, `configuration_snapshot.json`, `selections.json`, `coverage.json`, `step_zero.json`.
- `<seed>_<arm>_initial.pt`, `<seed>_<arm>_trained.pt`, `schedule_<seed>.npy`.
- Fitted PCA/readout NPZ files and per-window score files, plus reconstruction errors and targets/source coordinates.
- `uncertainty.json` and `paired_day_bootstrap.npz`. NaNs in the latter explicitly mark undefined resampling replicates; they are never converted to zero AP.
- `traceback.txt` if execution blocks.

Existing outputs are never overwritten. No legacy checkpoint or output-protection settings are consulted. To limit disk use, repeated full embedding matrices are not saved; they can be re-extracted from the saved initial/final models and exact frozen windows. Model weights, projection/readout parameters and per-window scores are retained.

## Decision meaning

`CONTINUE_DBRL_VALIDATION` means the fixed development margins and uncertainty checks passed. It is not evidence of anomaly detection, human-activity recognition, or guaranteed publishability.

`STOP_GRAPH_DEVELOPMENT` means the candidate did not meet all fixed criteria. Do not automatically increase the budget or tune using the same evaluation outcomes.

`INCONCLUSIVE` means excessive undefined temporal-bootstrap replicates prevent the prescribed decision. This is not a positive result.

The raw exact-rule ceiling is intentional: sensor states and transitions are deterministic functions of valid observations. This experiment measures accessible preservation of that information, not discovery of hidden human behavior. The later period has already informed development; it is not an untouched final test. Uniform and enriched cohorts overlap in 101 windows and are analyzed separately.

## Verification delivered

See `VERIFICATION.md` and `verification/`. The package has passed unit/integration tests and real-data verification-only execution. The full 6,144-update experiment has NOT been launched for this implementation delivery, and there are no new scientific performance results to interpret yet.

Run tests from the project directory:

```bash
python -m pytest tests/test_dbrl_decision.py -q
```

See `PROTOCOL.md` for the complete research decision specification and paper revision boundary.
