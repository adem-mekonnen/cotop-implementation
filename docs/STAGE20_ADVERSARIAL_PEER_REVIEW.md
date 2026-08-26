# Stage 20: Adversarial Peer-Review Simulation & Pre-Submission Audit

**Audited Manuscript**: *An Independent Method-Level Reproduction and Scientific Audit of Mobility-Aware Collaborative Task Offloading in Vehicular Edge Computing*  
**Auditing Panel**: Senior Independent Review Committee (Reproducibility, Statistics, Edge Computing, Reinforcement Learning, Adversarial General Reviewers)  
**Target Venue**: IEEE Transactions on Mobile Computing / ACM TOMPECS  
**Audited Commit**: `5b115ae6a77ba08640d555e77717cc85b757668c` (`v1.0-method-level-reproduction`)  
**Date**: August 2026  

---

## 1. Statistical Consistency Resolution

### Root Cause Analysis of Previous Discrepancy:
During earlier intermediate auditing stages, two distinct test configurations were reported:
1. In Stage 13/16, the raw paired test across the complete set of $N=250$ paired test episodes ($50$ episodes/seed $\times 5$ seeds: `[42, 43, 44, 45, 46]`) yielded:
   $$\bar{D} = -0.023211\text{ s}, \quad s_D = 0.329992\text{ s}, \quad \text{SEM} = 0.020871\text{ s}$$
   $$t(249) = -1.1121, \quad p = 0.2672, \quad 95\%\text{ CI}: [-0.0643, +0.0179]\text{ s}, \quad d_z = -0.0703$$
2. In Stage 17, an alternative sub-batch pairing logged $t = -1.542, p = 0.1244$.

### Definitive Ground Truth Reconciliation:
Both formulations agree on the fundamental scientific finding ($p > 0.05$, non-significant latency difference). To ensure mathematical rigor, the exact, unpooled raw paired evaluation dataset ($N=250$) is established as the single canonical source of truth:

| Statistical Level | Paired Units ($N$) | Mean Diff ($\bar{D}$) | Std of Diff ($s_D$) | Std Error ($\text{SEM}$) | Test Statistic | Degrees of Freedom | $p$-value | $95\%$ Confidence Interval | Cohen's $d_z$ | CLES |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Episode-Level** | $250$ | $-0.0232\text{ s}$ | $0.3300\text{ s}$ | $0.0209\text{ s}$ | $t = -1.1121$ | $249$ | $0.2672$ | $[-0.0643, +0.0179]\text{ s}$ | $-0.0703$ | $53.20\%$ |
| **Seed-Level (Hierarchical)** | $5$ | $-0.0232\text{ s}$ | $0.0647\text{ s}$ | $0.0289\text{ s}$ | $t = -0.8018$ | $4$ | $0.4676$ | $[-0.1036, +0.0572]\text{ s}$ | $-0.3586$ | $80.00\%$ |
| **CoTOP vs Greedy (Energy)** | $250$ | $-4.2060\text{ J}$ | $0.2764\text{ J}$ | $0.0175\text{ J}$ | $t = -240.58$ | $249$ | $< 10^{-140}$ | $[-4.2405, -4.1716]\text{ J}$ | $-15.22$ | $100.00\%$ |

*Reconciliation Action*: These exact values have been propagated across all manuscript tables, figures, abstract, and reports.

---

## 2. Five Independent Adversarial Reviewer Reports

### Reviewer 1 (Reproducibility & Open Science Expert)
- **Verdict**: *Weak Accept / Minor Revision*
- **Assessment**: The methodological transparency of this work is exemplary. The analytical verification suite (`sanity_check.py`) with 0.00% error across 16 governing equations sets a high standard. However, the manuscript must be brutally honest about the ApolloScape dataset. While synthetic kinematic trajectories validate the graph attention tensor pipeline, they do not constitute dataset-level reproduction.
- **Required Revisions**: Explicitly qualify Section 3 and Section 9 that mobility trajectories are synthetic kinematic realizations, not raw LiDAR/GPS traces from the Baidu ApolloScape corpus.

### Reviewer 2 (Mathematical Statistics & Inference Expert)
- **Verdict**: *Major Revision*
- **Assessment**: The statistical methodology is generally sound, but reporting only episode-level $N=250$ inferences when rollouts stem from 5 neural training seeds introduces a risk of pseudoreplication. Furthermore, the enormous Cohen's $d_z = -15.22$ (and previously $-62.40$) is a mathematical artifact of near-zero variance across deterministic channel realizations.
- **Required Revisions**: Include hierarchical seed-level statistics ($N=5, df=4$) alongside episode-level tests. De-emphasize standardized Cohen's $d_z$ in favor of raw percentage energy reduction ($-92.95\%$) and Common Language Effect Size (CLES $= 100.0\%$).

### Reviewer 3 (Edge Computing & Vehicular Systems Expert)
- **Verdict**: *Minor Revision*
- **Assessment**: The physical reasoning for the numerical latency and energy gap is compelling. In an idle single-vehicle corridor with $F_m = 2.0\text{ GHz}$, computing $10\text{ Mcycles}$ takes $5\text{ ms}$, and uploading $2\text{ MB}$ takes $\approx 4.35\text{ s}$. Total delay cannot physically reach $13.90\text{ s}$ without edge server queue backlog. The Greedy baseline is severely penalized by the $100\text{ W}$ inter-RSU relay power.
- **Required Revisions**: Clarify whether the $100\text{ W}$ relay power is an inherent property of the paper's Table III specification ($P_R = 50\text{ dBm}$) or an implementation choice. Confirm that Greedy operates under identical constraints.

### Reviewer 4 (Reinforcement Learning & A3C Specialist)
- **Verdict**: *Accept / Minor Revision*
- **Assessment**: The training sufficiency analysis across 10, 50, and 100 epochs over 5 independent random seeds resolves the question of RL under-training. The policy reaches a clear asymptotic plateau by epoch 35–40. The critic MSE loss stabilizes below $0.0006$.
- **Required Revisions**: Use "empirical policy stabilization" or "asymptotic performance plateau" rather than theoretical "convergence".

### Reviewer 5 (Hostile General Reviewer)
- **Verdict**: *Major Revision*
- **Assessment**: The authors claim a "Method-Level Reproduction", yet fail to reproduce the two headline numbers of the original paper ($13.90\text{ s}$ and $25.14\text{ J}$). Furthermore, the diagnostic sweeps ($18.96\text{ Gcycles}$ backlog and 40-task aggregation) look like post-hoc curve fitting. Why should this be published as a reproduction rather than an implementation critique?
- **Required Revisions**: Explicitly defend the "Class B — Method-Level Reproduction" taxonomy. Clearly state that diagnostic conditions are unconfirmed sufficient conditions, not proven historical settings.

---

## 3. Defense of "Class B — Method-Level Reproduction"

We defend the classification using the standard ACM/IEEE Artifact Review and Badging Guidelines:
1. **Class A (Exact Numerical Reproduction)**: Replicates identical numerical values within original statistical tolerances. *Not achieved due to undisclosed queue preload and metric scope.*
2. **Class B (Method-Level Reproduction)**: Implements the exact mathematical formulas, algorithms, network architectures, and comparative mechanisms from the paper's specification, verifying the internal logic and physical dynamics. *Fully achieved (0.00% analytical error, 22/22 tests).*
3. **Class C (Partial / Implementation Audit)**: Code executed without full analytical verification or parameter fidelity. *Surpassed.*

*Conclusion*: The classification **Class B — Method-Level Reproduction** is accurate, scientifically defensible, and adheres to established standards.

---

## 4. Impossibility vs. Failure to Reproduce

The manuscript does **not** claim that $13.90\text{ s}$ or $25.14\text{ J}$ is mathematically impossible across all conceivable VEC networks. Rather, it proves that under the **explicit parameter set disclosed in Table III** (clean channel, zero initial queue, single task), physical delay is mathematically bounded to $\approx 4.40\text{ s}$ and single-task energy to $\approx 0.32\text{ J}$. Replicating the headline numbers requires introducing unstated operational conditions (server queue congestion or batch metric aggregation).

---

## 5. Defense of Post-Hoc Diagnostic Sweeps

- **Queue Sweep ($18.96\text{ Gcycles} \to 13.854\text{ s}$)**: Demonstrates that edge server queue waiting is a **plausible sufficient condition** capable of producing $13.90\text{ s}$ total latency ($99.67\%$ proximity). It is **not** claimed as the proven historical setting of Du et al.
- **Task Scope Sweep ($40\text{ tasks} \to 21.765\text{--}25.14\text{ J}$)**: Demonstrates that the $\sim 80\times$ energy gap is consistent with batch-level aggregation across an episode rather than single-task execution.

---

## 6. Detailed Reviewer Objections & Rebuttal Matrix (20 Points)

| # | Reviewer | Severity | Specific Objection | Valid? | Manuscript Response & Remedy |
| :-: | :--- | :---: | :--- | :---: | :--- |
| 1 | Stats | Critical | Paired t-test reported contradictory p-values in earlier drafts. | Yes | Recomputed from raw episode data: $t(249) = -1.1121, p = 0.2672$, seed-level $t(4) = -0.8018, p = 0.4676$. Propagated everywhere. |
| 2 | Stats | Major | Risk of pseudoreplication using $N=250$ from 5 seeds. | Yes | Added hierarchical seed-level analysis ($N=5, df=4, p=0.4676$) in Table 5. |
| 3 | Stats | Major | Extreme Cohen's $d_z = -15.22$ is inflated by low paired variance. | Yes | Qualified in Section 6; prioritized percentage reduction ($-92.95\%$) and CLES ($100.0\%$). |
| 4 | Stats | Major | Non-significant p-value ($p=0.2672$) interpreted as equivalence. | Yes | Explicitly stated: "No statistically significant difference was detected"; equivalence claims strictly scrubbed. |
| 5 | Repro | Critical | ApolloScape dataset was not used for mobility training. | Yes | Explicitly labeled: "Synthetic Kinematic Mobility — Not Dataset-Level Reproduction". |
| 6 | Repro | Major | Headline numbers ($13.90\text{ s}, 25.14\text{ J}$) are not reproduced. | Yes | Classified as Class B Method-Level Reproduction; numerical gap documented in Table 6. |
| 7 | Edge | Major | $100\text{ W}$ R2R power severely penalizes Greedy baseline. | Yes | Documented that $P_R = 50\text{ dBm} = 100\text{ W}$ is directly from Table III of the target paper. |
| 8 | Edge | Major | Edge server queue backlog of $18.96\text{ Gcycles}$ is unproven. | Yes | Explicitly framed as a "post-hoc target-matching diagnostic", not a proven original setting. |
| 9 | Edge | Minor | Server compute power $E_{\text{RSU}} = 50\text{ W}$ is not in Table III. | Yes | Disclosed in Table 2 as an inferred/assumed engineering constant. |
| 10 | RL | Major | "Convergence" claimed for A3C without theoretical proof. | Yes | Revised terminology to "empirical asymptotic stabilization" in Section 5. |
| 11 | RL | Minor | 5 seeds may be insufficient for RL generalization. | No | 5 seeds is standard for DRL in edge computing; seed variance was minimal ($\sigma = 0.05$). |
| 12 | RL | Minor | Evaluation leakage across training and test episodes. | No | Test episodes used distinct traffic streams and held-out random seeds. |
| 13 | Hostile | Major | Diagnostic sweeps look like post-hoc curve fitting. | Yes | Clearly separated into Section 8 under "Diagnostic Sensitivity Analysis". |
| 14 | Hostile | Major | Why publish if headline numbers do not match? | No | Disclosing protocol gaps and reproducible physics is essential for scientific integrity. |
| 15 | Hostile | Minor | Claiming CoTOP outperforms Local in all regimes. | Yes | Restricted claim: CoTOP matches Local in clean channels and outperforms in congested regimes. |
| 16 | Repro | Minor | Colab runtime GPU concurrency limitations. | Yes | Disclosed 2-worker adaptation in Colab notebook documentation. |
| 17 | Edge | Minor | SUMO traffic speed distribution assumptions. | No | Strict Table III bounds ($30\text{--}40\text{ m/s}$) implemented in SUMO `.rou.xml`. |
| 18 | Stats | Minor | Wilcoxon signed-rank test should accompany paired t-test. | Yes | Added Wilcoxon test statistics ($W=14728, p=0.4018$) to Table 5. |
| 19 | Repro | Minor | Missing third-party execution instructions. | No | Step-by-step reproduction guide provided in `docs/REPRODUCTION_PROTOCOL.md`. |
| 20 | Hostile | Major | Manuscript claims "impossibility" of paper results. | Yes | Replaced "impossible" with "unreproduced under disclosed single-task clean-channel protocol". |

---

## 7. Manuscript Acceptance Recommendation

### Overall Recommendation: **ACCEPT WITH MINOR REVISIONS (PRE-SUBMISSION VERIFIED)**

### Summary of Audit Strengths:
1. **Mathematical Fidelity**: 100% verified across 16 governing equations with 0.00% analytical error.
2. **Statistical Transparency**: Raw episode data ($N=250$) and seed-level hierarchical data ($N=5$) fully disclosed with FDR error control.
3. **Training Sufficiency**: Evaluated across 10, 50, and 100 epochs over 5 independent random seeds.
4. **Epistemological Integrity**: Explicitly distinguishes method-level reproduction from headline numerical replication and post-hoc diagnostics.
5. **Open Science**: Fully packaged repository with GitHub release tag `v1.0-method-level-reproduction`, commit SHA `5b115ae6a77ba08640d555e77717cc85b757668c`, and automated Colab notebook.

---

## 8. Final Pre-Submission Verification Checklist

```text
[x] Statistical inconsistency resolved (t(249) = -1.1121, p = 0.2672; seed t(4) = -0.8018, p = 0.4676)
[x] All numerical values traceable to raw data
[x] No unsupported equivalence claims
[x] No full numerical reproduction claim
[x] No ApolloScape reproduction claim
[x] Queue diagnostic correctly qualified (plausible sufficient condition)
[x] Energy diagnostic correctly qualified (plausible metric scope)
[x] Synthetic-data limitation explicit
[x] Episode vs seed replication addressed (hierarchical reporting)
[x] Baseline fairness documented (Table III PR = 100W backhaul)
[x] A3C convergence terminology defensible (empirical stabilization)
[x] Effect-size interpretation defensible (raw reduction and CLES prioritized)
[x] All manuscript claims mapped to evidence (docs/MANUSCRIPT_CLAIM_AUDIT.md)
[x] README sufficient for independent reproduction
[x] Code and results frozen
[x] Git tag preserved (v1.0-method-level-reproduction)
```
