# Phase 10 — Paper-to-Implementation Fidelity, Numerical Reconciliation & Claim Validation Audit

This directory contains the audit artifacts for the Phase 10 comprehensive scientific comparison between the published manuscript (*Du et al., IEEE TMC 2026*) and the repository implementation.

## Deliverables
- `REPORT.md`: Authoritative manuscript-level scientific review addressing all Phase 10 audit objectives and answering mandatory questions.
- `paper_specification.json` & `paper_specification.md`: Formal machine-readable and markdown extraction of the target paper's system model, task model, compute parameters, and published values.
- `equation_implementation_matrix.csv`: Equation-by-equation mapping from paper equations (Eq. 1–28) to repository source code, functions, units, and tests.
- `parameter_fidelity_matrix.csv`: Comprehensive parameter-level comparison detecting all units, scales, and conversions.
- `scenario_fidelity_matrix.csv`: Detailed topological and geometric audit of `corridor_2400m` and `grid_200m` scenarios.
- `training_fidelity_matrix.csv`: Complete audit of training hyperparameters, architectures, and data-leakage boundaries.
- `baseline_fidelity_matrix.csv`: Complete audit of baseline algorithms (`Local`, `Greedy`, `DDQN`, `wo_md`, `wo_tp`, `wo_co`, `QRMP-DQN`).
- `published_vs_reproduced.csv`: Exact numerical comparison of published vs. reproduced metrics, differences, and statistical confidence intervals.
- `numerical_discrepancy_root_cause.md`: Rigorous scientific analysis of the $\approx 7\times - 10\times$ numerical scale gap.
- `discrepancy_decomposition.csv`: Component-by-component latency and energy decomposition under Table III equations.
- `scientific_claim_matrix.csv`: Claim-by-claim classification of paper hypotheses (SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED).
- `manifest.json`: Machine-readable provenance and audit metadata manifest.
- `figures/`:
  - `fig1_paper_vs_reproduced_delay.png`: Bar chart of published vs. reproduced total delay.
  - `fig2_paper_vs_reproduced_energy.png`: Bar chart of published vs. reproduced energy consumption.
  - `fig3_discrepancy_decomposition.png`: Physical latency breakdown under literal Table III equations.
  - `fig4_claim_validation_breakdown.png`: Classification distribution of major paper claims.
