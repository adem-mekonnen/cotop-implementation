# Stage 21: Final Pre-Submission Artifact Audit & Release Manifest

**Audit Scope**: Final Publication Integrity & Pre-Submission Audit  
**Auditor**: Senior Scientific Reproducibility Auditor & Publication Editor  
**Audited Release**: `v1.0-method-level-reproduction`  
**Commit SHA**: `5b115ae6a77ba08640d555e77717cc85b757668c`  
**Date**: August 2026  

---

## 1. Source-Code Freeze Audit

| Verification Check | Target Status | Measured Result | Audit Verdict |
| :--- | :--- | :--- | :---: |
| `envs/comm_model.py` | 0 modifications | Unchanged (0 diff lines) | **PASS** |
| `envs/comp_model.py` | 0 modifications | Unchanged (0 diff lines) | **PASS** |
| `sanity_check.py` | 0.00% analytical deviation | Passed 100% (0.00e+00 error across Eq. 1–13, 23, 25) | **PASS** |
| `pytest -q` | 22/22 unit tests | 22 passed in 6.08s | **PASS** |
| Git Release Tag | `v1.0-method-level-reproduction` | Verified present | **PASS** |
| Verified Base Commit | `5b115ae6a77ba08640d555e77717cc85b757668c` | Verified HEAD / Base | **PASS** |
| Uncommitted Code | 0 changes to physics/math | Core physical logic 100% frozen | **PASS** |

---

## 2. Statistical Ground Truth Verification

Direct recomputation from $N=250$ raw paired evaluation episodes across 5 seeds:

### CoTOP vs Local (Total Delay):
- **Paired Sample Size**: $N = 250$ ($df = 249$)
- **Paired Mean Difference ($\bar{D}$)**: $-0.0232\text{ s}$
- **Std of Differences ($s_D$)**: $0.3300\text{ s}$
- **Standard Error ($\text{SEM}$)**: $0.0209\text{ s}$
- **Paired $t$-statistic**: $t(249) = -1.1121$
- **Raw $p$-value**: $p = 0.2672$
- **Wilcoxon Signed-Rank**: $W = 14728.0, p = 0.4018$
- **$95\%$ Confidence Interval**: $[-0.0643, +0.0179]\text{ s}$
- **Paired Cohen's $d_z$**: $-0.0703$
- **Common Language Effect Size (CLES)**: $53.20\%$

### Seed-Level Hierarchical Analysis ($N=5$ Seeds):
- **Seed-Level $t$-statistic**: $t(4) = -0.8018$
- **Seed-Level $p$-value**: $p = 0.4676$
- **Seed-Level $95\%$ Confidence Interval**: $[-0.1036, +0.0572]\text{ s}$
- **Seed-Level Cohen's $d_z$**: $-0.3586$

### CoTOP vs Greedy (Total Energy):
- **CoTOP Mean Energy**: $0.319 \pm 0.005\text{ J}$
- **Greedy Mean Energy**: $4.525 \pm 0.068\text{ J}$
- **Mean Difference**: $-4.2060\text{ J}$
- **Percentage Energy Reduction**: $\mathbf{-92.95\%}$
- **Paired $t$-statistic**: $t(249) = -240.58$ ($p < 10^{-140}$, Holm/FDR $p_{\text{adj}} < 10^{-4}$)
- **Paired Cohen's $d_z$**: $-15.22$
- **Common Language Effect Size (CLES)**: $\mathbf{100.00\%}$ ($250/250$ episodes)
- **$95\%$ Confidence Interval**: $[-4.2405, -4.1716]\text{ J}$

---

## 3. Performance Table Consistency Audit

All tables and manuscript text report identical baseline metrics:
- **Local**: Total Delay = $4.425 \pm 0.023\text{ s}$ ($95\%\text{ CI}: [4.397, 4.453]$), Total Energy = $0.320 \pm 0.005\text{ J}$ ($95\%\text{ CI}: [0.314, 0.326]$), Completion $= 100.0\%$.
- **CoTOP**: Total Delay = $4.402 \pm 0.060\text{ s}$ ($95\%\text{ CI}: [4.327, 4.477]$), Total Energy = $0.319 \pm 0.005\text{ J}$ ($95\%\text{ CI}: [0.313, 0.325]$), Completion $= 100.0\%$.
- **Greedy**: Total Delay = $4.393 \pm 0.050\text{ s}$ ($95\%\text{ CI}: [4.331, 4.455]$), Total Energy = $4.525 \pm 0.068\text{ J}$ ($95\%\text{ CI}: [4.441, 4.609]$), Completion $= 100.0\%$.

---

## 4. Manuscript Claim Audit & Prohibited Words Check

- [x] **No Unsupported Equivalence Claims**: Replaced with *"No statistically significant difference was detected ($p = 0.2672$)"*.
- [x] **No Full Reproduction Claims**: Replaced with *"Class B — Method-Level Reproduction, as defined by this study's reproduction taxonomy"*.
- [x] **No Impossibility Claims**: Replaced with *"Under the disclosed clean-channel parameters, single-task physical latency is bounded to $\approx 4.40\text{ s}$"*.
- [x] **No Claims of Real ApolloScape Usage**: Replaced with *"Synthetic kinematic trajectories used to validate spatial graph attention tensors"*.
- [x] **Diagnostic Clarity**: $18.96\text{ Gcycles}$ backlog and 40-task batch aggregation are explicitly identified as *post-hoc diagnostic sensitivity hypotheses*, not confirmed historical protocol settings.

---

## 5. Pseudoreplication & Replication Scope

The manuscript explicitly distinguishes:
1. **Scenario / Episode Level ($N=250$)**: Paired evaluations across identical traffic scenarios.
2. **Seed Level ($N=5$)**: Independent neural training realizations (`[42, 123, 456, 789, 2026]`) with hierarchical reporting in Table 5 ($t(4) = -0.8018, p = 0.4676$).

---

## 6. Table & Figure Audit

- **Tables 1 through 9**: Formatted in Markdown (`.md`) and LaTeX (`.tex`) in `manuscript/tables/`. All numerical entries cross-verified against raw CSV files.
- **Figures 1 through 7**: Validated PNG figures in `manuscript/figures/`. Figures 6 and 7 feature explicit "POST-HOC SENSITIVITY DIAGNOSTIC" labels.

---

## 7. Pre-Submission Release Manifest

```text
====================================================================================================
COTOP REPRODUCIBILITY RELEASE MANIFEST
====================================================================================================
Repository Tag:            v1.0-method-level-reproduction
Git Commit SHA:            5b115ae6a77ba08640d555e77717cc85b757668c
Target Research Paper:     IEEE Transactions on Mobile Computing (TMC 2026, DOI: 10.1109/TMC.2025.3631820)
Python Version:            3.11.1 / 3.10.12 (Colab Verified)
PyTorch Version:           2.4.1+cu121
SUMO Simulator Version:    Eclipse SUMO 1.25.0
Random Seeds Evaluated:    [42, 123, 456, 789, 2026]
A3C Training Horizons:     10, 50, 100 Epochs (100 to 1000 Episodes)
Evaluation Episodes:       N = 250 Paired Episodes per Method (1500 Total)
Unit Test Pass Rate:       22/22 (100% Passing)
Analytical Deviation:      0.00% across Eq. 1–13, 23, 25
Manuscript Version:        v1.0 (IEEE/ACM Submission Ready)
Documentation Suite:       docs/FINAL_REPRODUCTION_REPORT.md, REPRODUCTION_PROTOCOL.md,
                           STATISTICAL_METHODS.md, LIMITATIONS_AND_THREATS.md,
                           CLAIM_EVIDENCE_MATRIX.md, STAGE20_ADVERSARIAL_PEER_REVIEW.md,
                           STAGE21_FINAL_PRE_SUBMISSION_AUDIT.md
====================================================================================================
```

---

## 8. Final Decision Scorecard

```text
SOURCE CODE:         PASS
STATISTICS:          PASS
MANUSCRIPT:          PASS
FIGURES:             PASS
TABLES:              PASS
REPRODUCIBILITY:     PASS
CLAIM DISCIPLINE:    PASS
GIT RELEASE:         PASS

OVERALL PRE-SUBMISSION STATUS:
READY FOR EXTERNAL PEER REVIEW

VERDICT:
STOP EXPERIMENTATION. PROCEED TO SUBMISSION.
```
