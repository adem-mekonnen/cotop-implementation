# Phase 8 — Ablation Validity, Statistical Significance & Component Contribution Audit

This directory contains the audit artifacts for the Phase 8 investigation of CoTOP ablation variants (`wo_md`, `wo_tp`, `wo_co`), multi-seed statistical significance testing across 60 frozen realizations, and multi-objective algorithm ranking.

## Artifacts
- `REPORT.md`: Master comprehensive audit report answering all mandatory scientific questions.
- `ablation_implementation_matrix.csv`: Detailed audit of the 4 CoTOP variants, mechanisms removed, execution paths, and root causes for observed behaviors.
- `ablation_behavioral_comparison.csv`: Action distributions, model parameter hashes, action sequence hashes, and metric differences.
- `statistical_significance.csv`: Full paired statistical significance tests (Student's t-test, Wilcoxon signed-rank, Cohen's d effect sizes, 95% CI, positive/negative counts).
- `scenario_workload_analysis.csv`: Detailed breakdown of distributions (Mean, Median, Std, P5, P25, P75, P95, Min, Max, 95% CI, failure categories) across scenarios and workloads.
- `algorithm_ranking.csv`: Multi-objective Pareto ranking (Delay, Energy, Completion) and trade-off classifications.
- `paired_realization_integrity.csv`: Realization hash audit proving all 7 algorithms were evaluated on 100% identical realization instances.
- `metric_definitions.md`: Mathematical definitions and formulas for all statistical metrics.
- `provenance_manifest.json`: Machine-readable metadata and provenance manifest.
- `figures/`:
  - `fig1_ablation_delay_comparison.png`: Mean delay across ablations and baselines.
  - `fig2_ablation_energy_comparison.png`: Dynamic energy consumption comparison.
  - `fig3_pareto_delay_energy_tradeoff.png`: Delay vs Energy Pareto scatter plot.
  - `fig4_paired_delay_differences.png`: Boxplot distributions of paired delay differences.
