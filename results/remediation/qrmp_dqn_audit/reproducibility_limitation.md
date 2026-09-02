# Scientific Reproducibility Limitation: QRMP-DQN Baseline Exclusion

## 1. Scientific Integrity Principle
In scientific reproducibility investigations, substituting an ad-hoc generic surrogate (such as standard single-pass QR-DQN) and labeling it "QRMP-DQN" violates scientific integrity by:
1. **Misattributing Reference [33]**: Reference [33] is defined by its Multi-Pass (MP) parameterization for STAR-RIS.
2. **Methodological Pollution**: Passing off an ungrounded surrogate as the author's baseline obscures genuine literature gaps.
3. **Unverifiable Degrees of Freedom**: Constructing an ad-hoc implementation requires inventing multiple arbitrary hyperparameters ($N, \kappa, C, \mu(	heta)$) without experimental justification.

## 2. Definitive Exclusion Decision
In accordance with strict scientific standards:
- **QRMP-DQN is formally classified as**: `NOT_REPRODUCIBLE_FROM_AVAILABLE_EVIDENCE`.
- **Primary Comparative Matrix**: Formally evaluated across the 7 fully verified algorithms (`CoTOP`, `DDQN`, `Local`, `Greedy`, `wo_md`, `wo_tp`, `wo_co`) spanning 420 completed factorial runs.
- **Reporting Requirement**: All comparative tables and manuscript sections must explicitly display `N/A (EXCLUDED — REF [33] STAR-RIS DOMAIN MISMATCH)` rather than silently fabricating surrogate data.
