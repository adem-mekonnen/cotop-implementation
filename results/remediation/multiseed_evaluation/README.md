# Phase 7 — Full Factorial Multi-Seed Evaluation & Statistical Robustness Audit Artifacts

This directory contains the audit artifacts for the complete 420-run factorial evaluation campaign evaluating 7 algorithms across 2 scenarios, 3 workloads, and 10 seeds on 60 frozen realizations.

## Artifacts
- `REPORT.md`: Comprehensive scientific audit report containing full cross-seed statistics, paired comparisons, and paper discrepancy rankings.
- `config.json`: Master machine-readable configuration and provenance manifest.
- `run_summary.csv`: Run-level metrics for all 420 evaluation runs.
- `run_inventory.csv`: Full provenance inventory linking every run to its Git commit, realization SHA-256, and checkpoint SHA-256.
- `seed_summary.csv`: Aggregated cross-seed statistics (mean, std, median, min, max, p5, p95, 95% CI) for each algorithm, scenario, and workload.
- `algorithm_summary.csv`: Grand cross-scenario aggregated statistics per algorithm.
- `comparison_summary.csv`: Paired cross-algorithm comparisons (CoTOP vs Local, Greedy, DDQN, ablations).
- `figures/`: Publication figures generated directly from raw multi-seed campaign data:
  - `fig1_delay_vs_workload_corridor.png`: Mean delay vs workload across 10 seeds.
  - `fig2_energy_vs_workload_corridor.png`: Mean energy vs workload across 10 seeds.
  - `fig3_completion_vs_workload_corridor.png`: Completion ratio vs workload across 10 seeds.
- `task_traces/`: Task-level telemetry traces for representative runs.
