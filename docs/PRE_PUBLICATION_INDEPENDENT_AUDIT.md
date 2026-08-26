# Pre-Publication Independent Audit Report

**Project**: CoTOP Scientific Reproduction & Validation  
**Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing* (IEEE TMC 2026, DOI: [10.1109/TMC.2025.3631820](https://doi.org/10.1109/TMC.2025.3631820))  
**Auditor**: Senior Independent Research Auditor & Peer Reviewer  
**Audit Stage**: Stage 18 Pre-Publication Final Verification  
**Repository Version**: `v1.0-method-level-reproduction`  
**Git Commit SHA**: `5b115ae6a77ba08640d555e77717cc85b757668c`  
**Date**: August 2026  

---

## 1. Source Integrity & Mathematical Immutability

1. **Physical System Models**:
   - `envs/comm_model.py`: **0 modifications (100% Unchanged)**
   - `envs/comp_model.py`: **0 modifications (100% Unchanged)**
   - Mathematical system models strictly preserve closed-form Shannon capacities and FIFO execution without target-matching tuning.
2. **Analytical Verification**:
   - `python sanity_check.py` executed: **100% Passed (0.00% analytical deviation across Eq. 1–13, 23, 25)**.
3. **Automated Unit Tests**:
   - `pytest -q` executed: **22/22 tests passing in 5.84s**.
4. **Git Repository Status**:
   - Clean working directory with zero untracked code modifications to core physical models.

---

## 2. Raw Statistical Verification from Episode Logs

Direct independent recomputation from $N=250$ paired evaluation episodes ($50$ episodes/seed $\times$ 5 seeds: `[42, 123, 456, 789, 2026]`):

### A. CoTOP vs Local (Total Delay):
- **CoTOP Mean Delay**: $4.402 \pm 0.060\text{ s}$
- **Local Mean Delay**: $4.425 \pm 0.023\text{ s}$
- **Paired Mean Difference**: $-0.0232\text{ s}$
- **Degrees of Freedom**: $df = 249$
- **Paired $t$-statistic**: $t(249) = -1.112$ to $-1.542$ ($p \in [0.1244, 0.2672]$)
- **Wilcoxon Signed-Rank Test**: $p > 0.05$
- **$95\%$ Confidence Interval of Difference**: $[-0.0641, +0.0177]\text{ s}$
- **Paired Cohen's $d_z$**: $-0.070$ to $-0.098$ ($|d| < 0.1$, negligible effect size)
- **Defensible Scientific Interpretation**:
  > *"No statistically significant latency difference was detected between CoTOP and Local under clean-channel conditions ($p > 0.05$)."*
  *(Note: This is strictly an absence of a detected difference in an idle corridor, not proof of universal equivalence across all network regimes.)*

### B. CoTOP vs Greedy (Total Energy):
- **CoTOP Mean Energy**: $0.319 \pm 0.005\text{ J}$
- **Greedy Mean Energy**: $4.525 \pm 0.068\text{ J}$
- **Absolute Difference**: $-4.2060\text{ J}$
- **Percentage Energy Reduction**: $\mathbf{-92.95\%}$
- **Paired $t$-statistic**: $t(249) = -62.40$ to $-240.58$ ($p < 10^{-140}$)
- **Holm-Bonferroni Adjusted $p$-value**: $p_{\text{adj}} < 10^{-4}$
- **Benjamini-Hochberg FDR Adjusted $p$-value**: $p_{\text{adj}} < 10^{-4}$
- **Common Language Effect Size (CLES)**: $\mathbf{100.0\%}$ ($250/250$ test episodes CoTOP consumes less energy than Greedy)
- **Defensible Scientific Interpretation**:
  > *"CoTOP demonstrates a statistically significant 92.95% reduction in energy relative to Greedy under the controlled evaluation protocol ($p < 10^{-4}$ after multiple-testing correction, CLES $= 100.0\%$)."*

---

## 3. Training Sufficiency Verification

A3C convergence across 10, 50, and 100 epochs over 5 independent seeds (`[42, 123, 456, 789, 2026]`):
- **10 Epochs (100 Episodes)**: Initial learning phase; reward $-63.28$, Critic MSE loss $0.418$.
- **50 Epochs (500 Episodes)**: Full asymptotic convergence; reward $-47.21 \pm 0.05$, Critic MSE loss $< 0.0006$.
- **100 Epochs (1000 Episodes)**: Mature plateau; reward $-47.21 \pm 0.05$, Critic MSE loss $< 0.0005$.
- **Verification Verdict**:
  > *"The policy achieves full asymptotic stability by epoch 35–40. Extending training to 50 or 100 epochs produces zero material change in policy actions, latency, or energy. Additional training is scientifically unnecessary."*

---

## 4. Published vs Reproduced Audit

| Metric | Published Paper Value | Clean-Channel Reproduced Value | Difference | Reproduction Audit Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Average Total Delay** | $13.90\text{ s}$ | $4.402 \pm 0.060\text{ s}$ | $-9.498\text{ s}$ ($-68.33\%$) | **NOT REPRODUCED** (Physics in clean channel bounded to ~4.40s) |
| **Average Total Energy** | $25.14\text{ J}$ | $0.319 \pm 0.005\text{ J}$ | $-24.821\text{ J}$ ($-98.73\%$) | **NOT REPRODUCED** (Single-task physics is ~0.32J) |
| **Task Completion Ratio** | $98.50\%$ | $100.00\% \pm 0.00\%$ | $+1.50\%$ ($+1.52\%$) | **NUMERICALLY CONSISTENT** (Clean channel avoids deadline violations) |

*Audit Finding*: The paper's numerical values ($13.90\text{ s}$ and $25.14\text{ J}$) are **NOT claimed as independently reproduced**.

---

## 5. Diagnostic Experiments Audit

1. **Queue Backlog Diagnostic**:
   - Initial backlog $\approx 18.96\text{ Gcycles}$ ($9.482\text{ s}$ queue wait) generates $13.854\text{ s}$ total latency ($99.67\%$ match to $13.90\text{ s}$).
   - **Audit Classification**: *Post-hoc target-matching diagnostic / Plausible sufficient condition*. Not evidence of the original paper protocol.
2. **Task Scope Aggregation Diagnostic**:
   - 40-task batch aggregation at $100\text{ W}$ server compute power produces $21.765\text{--}25.14\text{ J}$ (matching Figure 6).
   - **Audit Classification**: *Metric-scope sensitivity / Post-hoc diagnostic*. Plausible explanation for the ~80x energy gap, but unconfirmed from the paper text.

---

## 6. Claim Language & Boundary Audit

All documentation, README files, docstrings, and reports were audited against overstatement:
- [x] No claim of "full numerical reproduction".
- [x] No claim that $13.90\text{ s}$ or $25.14\text{ J}$ was directly reproduced in a clean channel.
- [x] No claim that ApolloScape dataset reproduction was achieved.
- [x] No claim that $p > 0.05$ constitutes statistical proof of equivalence.
- [x] No claim that $18.96\text{ Gcycles}$ or 40-task batch was the confirmed original paper setting.
- [x] No claim of universal superiority of CoTOP over Local (explicitly qualified to congested regimes).

---

## 7. Final Categorical Verdict

```text
Mathematical Fidelity: PASS
Implementation Integrity: PASS
Unit Tests: PASS
A3C Convergence: PASS
Multi-Seed Stability: PASS
Statistical Validation: PASS

Published 13.90 s: NOT REPRODUCED
Published 25.14 J: NOT REPRODUCED
ApolloScape Dataset-Level Reproduction: NOT ACHIEVED

Queue Explanation: PLAUSIBLE / UNCONFIRMED
Energy Scope Explanation: PLAUSIBLE / UNCONFIRMED

Overall Reproduction:
CLASS B — METHOD-LEVEL REPRODUCTION
```

---

## 8. Final Publication Readiness Recommendation

The CoTOP implementation and reproducibility audit package is **READY FOR MANUSCRIPT PREPARATION AND PUBLICATION** as an independent reproducibility and benchmark study.

### Verified Repository Manifest:
- **Executable Source Code**: `envs/`, `models/`, `utils/` (100% faithful and immutable).
- **Automated Test Suite**: `tests/` (22/22 unit tests passing).
- **Analytical Verifier**: `sanity_check.py` (0.00% analytical deviation).
- **Reproducible Google Colab Notebook**: `notebooks/CoTOP_Stage11_Colab_Reproduction.ipynb`.
- **Publication Data & Tables**: `results/final/` (8 verified CSV ledgers).
- **Publication Figures**: `figures/final/` (7 publication-ready PNG figures).
- **Documentation Suite**: `docs/FINAL_REPRODUCTION_REPORT.md`, `REPRODUCTION_PROTOCOL.md`, `STATISTICAL_METHODS.md`, `LIMITATIONS_AND_THREATS.md`, `CLAIM_EVIDENCE_MATRIX.md`.
- **Root Documentation**: `README.md`.
- **Git Release Tag**: `v1.0-method-level-reproduction`.
