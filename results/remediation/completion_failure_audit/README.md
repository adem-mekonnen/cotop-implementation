# Phase 3 — Completion / Failure & Local-Execution Audit Artifacts

This directory contains the audit artifacts investigating task completion, failure predicates, coverage boundary dynamics, and local-execution distributions on frozen realization `realization_corridor_2400m_w20_seed42.json`.

## Files
- `config.json`: Audit configuration, parameters, Git SHA, and realization SHA-256.
- `failure_trace.csv`: Detailed telemetry for all failed tasks, tracking arrival positions, velocities, primary RSUs, deadline slacks, coverage slacks, and failure classifications.
- `completion_summary.json`: Aggregated completion ratios, failure counts by classification, and Local execution metric distributions (mean, median, p50, p95, max).
- `REPORT.md`: Comprehensive scientific report answering all 8 required audit questions and providing the Phase 3 gate verdict.

## Key Summary
- **Total Evaluated Tasks**: 200
- **Completed Tasks**: 193 ($96.50\%$)
- **Failed Tasks**: 7 ($3.50\%$)
- **Coverage Exit Failures**: 7 ($100.0\%$ of failures)
- **Deadline Miss Failures**: 0 ($0.0\%$ of failures)
- **Root Cause**: High-speed vehicles ($35\text{ m/s}$) arriving at $x = 2400\text{ m}$ exit RSU 5 coverage before task execution finishes.
