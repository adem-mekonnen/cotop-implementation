# Phase 6 — DDQN Checkpoint Reload & Independent Evaluation Integrity Audit

This directory contains the audit artifacts proving that DDQN checkpoints can be saved, retained, strictly reloaded, and evaluated deterministically across independent Python processes without synthetic data or unverified fallbacks.

## Files
- `REPORT.md`: Comprehensive Phase 6 scientific audit report answering all 18 mandatory questions.
- `checkpoint_manifest.json`: Metadata, file size, Git commit, and SHA-256 of the evaluated DDQN checkpoint.
- `config.json`: Execution configuration, realization path, and subprocess command strings.
- `deterministic_comparison.json`: Bitwise comparison between Evaluation #1 and Evaluation #2.
- `ddqn_reload_test.json`: Standalone verification test summary.
- `checkpoints/ddqn_smoke_checkpoint.pt`: Physical DDQN checkpoint retained outside Git tracking.
- `evaluation_1/`: Full task trace (`task_trace.csv`) and evaluation manifest (`evaluation_metrics.json`) from the first evaluation run.
- `evaluation_2/`: Full task trace (`task_trace.csv`) and evaluation manifest (`evaluation_metrics.json`) from the second evaluation run.
