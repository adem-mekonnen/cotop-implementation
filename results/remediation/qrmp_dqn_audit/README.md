# Phase 11 — QRMP-DQN Baseline Fidelity & Comparative Audit

This directory contains the forensic audit, formal mathematical specification, and scientific exclusion records for the QRMP-DQN baseline (*Reference [33], Guo et al.*) in the context of the CoTOP paper reproduction.

## Artifacts
- `REPORT.md`: Authoritative scientific audit report addressing all Phase 11 questions, PAMDP domain mismatch analysis, and final comparative verdict.
- `qrmp_specification.json` & `qrmp_specification.md`: Formal machine-readable and markdown specification of QRMP-DQN (Reference [33]) and its compatibility analysis against the discrete VEC environment.
- `missing_information_matrix.csv`: Detailed matrix of missing architecture, quantile, and training parameters in the target paper.
- `evidence_for_unreproducibility.csv`: Forensic evidence establishing why QRMP-DQN is unreproducible from available evidence.
- `reproducibility_limitation.md`: Formal scientific limitation statement explaining why ad-hoc generic QR-DQN substitution is rejected.
- `implementation_fidelity.csv`: Implementation and evaluation status across all 8 potential algorithms.
- `scientific_claim_matrix.csv`: Scientific validation status of paper claims concerning QRMP-DQN.
- `manifest.json`: Machine-readable provenance and audit metadata manifest.
- `figures/`:
  - `fig1_qrmp_unreproducibility_breakdown.png`: Bar chart of key missing parameter dimensions.
  - `fig2_action_space_mismatch.png`: Visual breakdown of target environment (discrete 7-action) vs. Ref [33] hybrid PAMDP.
