# CoTOP Scientific Claim-to-Evidence Matrix

This document maps all primary project claims directly to their empirical evidence, source code locations, and statistical support.

---

## 1. Primary Scientific Claims (Claims A through G)

### Claim A: Mathematical Fidelity
- **Claim**: The mathematical formulations specified in Eqs. 1–13, 23, and 25 were faithfully implemented with 0.00% analytical deviation under the validated implementation.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - `sanity_check.py` closed-form unit verification passes 100% with $0.00\text{e}+00$ error across all 16 equations.
  - Automated test suite in `tests/` passes 22/22 unit tests.
  - Code locations: `envs/comm_model.py:calculate_v2r_rate()`, `envs/comp_model.py:calculate_case1_standalone()`, `calculate_case2_collaboration()`.

### Claim B: A3C Asymptotic Training Stability
- **Claim**: A3C training reaches asymptotic stability by approximately epoch 35–40 across five independent seeds, and extending training to 50 or 100 epochs produces no material change in policy behavior, latency, or energy.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - `results/final/04_training_sufficiency.csv`: Critic MSE loss drops below $0.0006$, reward plateaus at $-47.21 \pm 0.05$ across seeds `[42, 123, 456, 789, 2026]`.
  - Visualized in `figures/final/training_convergence.png`.

### Claim C: CoTOP vs Local Performance in Clean Channel
- **Claim**: Under clean-channel conditions, no statistically significant latency difference was detected between CoTOP and Local.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - `results/final/03_final_statistical_analysis.csv`: Paired $t$-test $t(249) = -1.542, p = 0.1244$, paired mean difference $-0.0232\text{ s}$ ($95\%\text{ CI}: [-0.0528, +0.0064]$).
  - Both algorithms execute Standalone offloading in an idle corridor.

### Claim D: CoTOP vs Greedy Energy Advantage
- **Claim**: CoTOP demonstrates an approximately 92.95% reduction in energy relative to Greedy under the controlled evaluation protocol, with statistically significant results after multiple-testing correction.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - `results/final/03_final_statistical_analysis.csv`: Paired difference $-4.2060\text{ J}$ ($-92.95\%$), $t(249) = -62.40, p = 1.2 \times 10^{-140}$, Holm-adjusted $p < 10^{-4}$, CLES $= 100.0\%$.
  - Visualized in `figures/final/energy_comparison.png`.

### Claim E: Numerical Gap on Published Target Metrics
- **Claim**: The published 13.90 s latency and 25.14 J energy values were not independently reproduced under the implemented clean-channel/single-scope protocol.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - `results/final/05_published_vs_reproduced.csv`: Measured delay is $4.402 \pm 0.060\text{ s}$ vs $13.90\text{ s}$; measured energy is $0.319 \pm 0.005\text{ J}$ vs $25.14\text{ J}$.
  - Visualized in `figures/final/published_vs_reproduced.png`.

### Claim F: Post-Hoc Diagnostic Nature of Queue and Energy Hypotheses
- **Claim**: Queue backlog (~18.96 Gcycles) and task aggregation (40 tasks) provide plausible post-hoc diagnostic explanations for published values, but the original paper's exact configurations cannot be established from disclosed information.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - `results/stage17/10_queue_diagnostic.csv`: $18.96\text{ Gcycles}$ backlog produces $13.854\text{ s}$ ($99.67\%$ match).
  - `results/stage17/11_task_scope_diagnostic.csv`: 40-task batch at $100\text{ W}$ server compute produces $21.765\text{--}25.14\text{ J}$ (matching Figure 6).
  - Both conditions are unstated in the original paper text (Table III).

### Claim G: Overall Reproduction Classification
- **Claim**: The overall reproduction level is Class B — Method-Level Reproduction.
- **Verification Status**: **VERIFIED**
- **Direct Evidence**:
  - Algorithms, mathematical physics, and neural networks are fully reproduced and validated. Numerical replication is constrained by unstated protocol parameters.
