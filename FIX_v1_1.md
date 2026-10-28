# Packaging fix v1.1

Includes the previously omitted src/scientific_validation/behavior_preservation/core.py unchanged from the tested codebase. No runner, model, selection, training budget or configuration changes.

Extract this archive into /content/dbrl_v1 (or your existing code root), keeping directory structure, then rerun the same command. This remains an incremental package for the existing BRE codebase.

The import error happened before main() and before optimization; there is no trained state to recover from this failed launch.

Validation of this packaging correction: all original Python files are byte-identical to v1; the added core.py is byte-identical to the utility used in the original successful 14-test and real-data verification. Python compilation passed and 19 local dependency modules resolved against the existing BRE base. The original verification report is retained as historical evidence; runtime tests were not repeated for this packaging-only correction.
