# Phase 5 — Controlled Paper-Result Reproduction & Metric Discrepancy Audit Artifacts

This directory contains the audit artifacts investigating the numerical, physical, and methodological relationship between published paper values and the reproduced experimental results.

## Files
- `REPORT.md`: Comprehensive scientific report explaining the parameter alignment, metric definitions, unit verification, and scale discrepancy forensics.
- `paper_protocol_matrix.csv`: 23-parameter alignment matrix comparing paper specifications against repository parameters (`EXACT`, `DERIVED`, `ASSUMED`, `CONFLICTING`).
- `metric_definitions.md`: Mathematical and algorithmic definitions of delay, energy, and completion ratio.
- `discrepancy_analysis.csv`: Quantitative comparison of published values vs reproduced values across CoTOP, Local, Greedy, and DDQN.
- `unit_audit.csv`: End-to-end verification of physical units and conversion factors.
- `sensitivity_analysis.csv`: Controlled parameter sensitivity diagnostics (bandwidth, payload size, CPU capacity, transmit power).
- `run_inventory.csv`: Provenance and execution records for Phase 5 experiment runs.
- `experiment_manifest.json`: Configuration, Git SHA, and physics hashes.
- `summaries/baseline_evaluation_summary.csv`: Detailed metric distributions and vehicle-level aggregation forensics across all 5 evaluated policies.
