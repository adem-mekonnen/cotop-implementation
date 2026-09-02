# Phase 9 — Training Provenance, Checkpoint Generalization & True Ablation Activation Audit

This directory contains the audit artifacts for the Phase 9 investigation of model training provenance, strict reloadability, empirical GAT-GRU mobility predictor activation rates, controlled diagnostic GAT activation experiments, true task priority disabling, and formal `wo_co` vs. `Local` mathematical equivalence.

## Artifacts
- `REPORT.md`: Comprehensive scientific audit report addressing all mandatory Phase 9 audit sections and gate criteria.
- `checkpoint_provenance.csv` & `checkpoint_provenance.json`: Full provenance audit of all trained model checkpoints (CoTOP, DDQN, MobilityGAT_GRU, and smoke checkpoints) with SHA-256 and parameter hashes.
- `mobility_activation_audit.csv`: Step-by-step measurement of GAT-GRU predictor activation across all 60 frozen realization files.
- `diagnostic_gat_activation_results.csv`: Controlled diagnostic evaluation comparing CoTOP (with GAT active) vs. `wo_md` (with linear fallback).
- `task_priority_activation_audit.csv`: Verification that `wo_tp` sets `use_priority=False` and modifies queue ordering and state priority features.
- `wo_co_local_equivalence.csv`: Mathematical proof and empirical verification that `wo_co` is formally equivalent to `Local`.
- `generalization_audit_matrix.csv`: Evaluation horizon and mechanism activation audit matrix.
- `manifest.json`: Machine-readable provenance and audit manifest.
- `figures/`:
  - `fig1_gat_activation_breakdown.png`: Mobility predictor activation breakdown in official evaluation.
  - `fig2_controlled_diagnostic_comparison.png`: Controlled diagnostic comparison between CoTOP and `wo_md`.
