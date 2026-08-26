# Manuscript Claim-to-Artifact Audit

This document provides a line-by-line audit mapping every major claim, table, and figure in the academic manuscript (`manuscript/manuscript.md`) to its exact empirical artifact in the repository.

---

## 1. Research Questions Audit

| Research Question | Manuscript Location | Empirical Finding | Supporting Artifact / File |
| :--- | :--- | :--- | :--- |
| **RQ1 (Mathematical Fidelity)** | Section 1, Section 4 | 0.00% analytical deviation across Eq. 1–13, 23, 25; 22/22 unit tests passing. | `sanity_check.py`, `tests/`, `results/final/01_reproduction_fidelity.csv` |
| **RQ2 (Training Sufficiency)** | Section 1, Section 5 | Policy stabilizes by epoch 35–40; 50 vs 100 epochs produces 0 change. | `results/final/04_training_sufficiency.csv`, `figures/final/training_convergence.png` |
| **RQ3 (Comparative Baselines)** | Section 1, Section 6 | CoTOP matches Local ($p=0.1244$); CoTOP reduces energy by 92.95% vs Greedy ($p<10^{-4}$). | `results/final/02_final_performance_comparison.csv`, `03_final_statistical_analysis.csv` |
| **RQ4 (Numerical Replicability)** | Section 1, Section 7 | Headline values ($13.90\text{ s}, 25.14\text{ J}$) are NOT independently reproduced. | `results/final/05_published_vs_reproduced.csv`, `figures/final/published_vs_reproduced.png` |
| **RQ5 (Diagnostic Gaps)** | Section 1, Section 8 | $18.96\text{ Gcycles} \to 13.854\text{ s}$; 40-task batch $\to 21.76\text{--}25.14\text{ J}$. | `results/stage17/10_queue_diagnostic.csv`, `11_task_scope_diagnostic.csv` |

---

## 2. Table Artifact Audit

| Manuscript Table | Content | Source CSV File |
| :--- | :--- | :--- |
| **Table 1** | Mathematical Implementation Fidelity Matrix | [`results/final/01_reproduction_fidelity.csv`](file:///d:/cotop-implementation/results/final/01_reproduction_fidelity.csv) |
| **Table 2** | Experimental Configuration & Parameters | [`manuscript/tables/table2_experimental_configuration.md`](file:///d:/cotop-implementation/manuscript/tables/table2_experimental_configuration.md), `configs/paper_parameters.yaml` |
| **Table 3** | A3C Training Sufficiency (10 vs 50 vs 100 Epochs) | [`results/final/04_training_sufficiency.csv`](file:///d:/cotop-implementation/results/final/04_training_sufficiency.csv) |
| **Table 4** | Final Controlled Performance Comparison ($N=250$) | [`results/final/02_final_performance_comparison.csv`](file:///d:/cotop-implementation/results/final/02_final_performance_comparison.csv) |
| **Table 5** | Statistical Hypothesis Tests & Multiple Testing Adjustments | [`results/final/03_final_statistical_analysis.csv`](file:///d:/cotop-implementation/results/final/03_final_statistical_analysis.csv) |
| **Table 6** | Published Target vs Reproduced Performance Matrix | [`results/final/05_published_vs_reproduced.csv`](file:///d:/cotop-implementation/results/final/05_published_vs_reproduced.csv) |
| **Table 7** | Claim-to-Evidence Matrix (Claims A–G) | [`results/final/06_claim_evidence_matrix.csv`](file:///d:/cotop-implementation/results/final/06_claim_evidence_matrix.csv) |
| **Table 8** | Threats to Scientific Validity | [`results/final/07_limitations.csv`](file:///d:/cotop-implementation/results/final/07_limitations.csv) |

---

## 3. Figure Artifact Audit

| Manuscript Figure | Caption Summary | Source Figure File |
| :--- | :--- | :--- |
| **Figure 1** | A3C Multi-Seed Convergence (Epochs 1–100 across 5 seeds) | [`figures/final/training_convergence.png`](file:///d:/cotop-implementation/figures/final/training_convergence.png) |
| **Figure 2** | Total Delay Comparison across Local, CoTOP, Greedy | [`figures/final/delay_comparison.png`](file:///d:/cotop-implementation/figures/final/delay_comparison.png) |
| **Figure 3** | Total Energy Comparison (-92.95% vs Greedy) | [`figures/final/energy_comparison.png`](file:///d:/cotop-implementation/figures/final/energy_comparison.png) |
| **Figure 4** | Multi-Seed Stability Boxplots across 5 Seeds | [`figures/final/seed_stability.png`](file:///d:/cotop-implementation/figures/final/seed_stability.png) |
| **Figure 5** | Published Target vs Reproduced Values | [`figures/final/published_vs_reproduced.png`](file:///d:/cotop-implementation/figures/final/published_vs_reproduced.png) |
| **Figure 6** | Diagnostic A: Edge Server Queue Sensitivity ($0\text{--}25\text{ Gcycles}$) | [`figures/final/queue_sensitivity.png`](file:///d:/cotop-implementation/figures/final/queue_sensitivity.png) |
| **Figure 7** | Diagnostic B: Task Scope Aggregation Sensitivity ($1\text{--}50\text{ tasks}$) | [`figures/final/task_scope_sensitivity.png`](file:///d:/cotop-implementation/figures/final/task_scope_sensitivity.png) |

---

## 4. Final Claim Audit & Boundary Check

1. **Equivalence Formulation**:
   - Verified: The manuscript states *"No statistically significant latency difference was detected between CoTOP and Local under clean-channel conditions ($p = 0.1244$)"* and strictly avoids claiming mathematical or statistical equivalence.
2. **Headlines Reproduction**:
   - Verified: The manuscript explicitly states that $13.90\text{ s}$ and $25.14\text{ J}$ were **not independently reproduced** under the clean-channel protocol.
3. **Diagnostic Status**:
   - Verified: The queue backlog ($\approx 18.96\text{ Gcycles}$) and 40-task batch aggregation are explicitly identified as **post-hoc diagnostic sensitivity explanations**, not proven original protocol settings.
4. **Dataset Status**:
   - Verified: The manuscript transparently discloses that synthetic kinematic trajectories were used in place of the unbundled raw ApolloScape dataset.
