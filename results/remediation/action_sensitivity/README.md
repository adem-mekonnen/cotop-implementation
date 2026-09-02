# Deterministic Action-Sensitivity Audit Artifact

This directory contains the paired, deterministic action-sensitivity audit artifacts comparing **AlwaysLocal** (Action 0) and **AlwaysCollaborate** (Actions 1..6) executed on frozen realization `realization_corridor_2400m_w20_seed42.json`.

## Files
- `config.json`: Experimental metadata, Git SHA, realization SHA-256, and protected physics hashes.
- `task_trace.csv`: Task-by-task paired telemetry for all 200 evaluated tasks.
- `summary.json`: Aggregated metrics, comparison counts, and scientific pass/fail verdicts.
- `REPORT.md`: Comprehensive scientific analysis distinguishing Hypothesis 1 from Hypothesis 2.

## Summary Verdict
- **Verdict**: **PASS**
- **Action Differences**: 200 / 200 (100.0%)
- **Execution-Case Differences**: 200 / 200 (100.0%)
- **Energy Differences**: 200 / 200 (100.0%)
