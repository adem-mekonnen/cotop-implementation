# Phase 2 — Full Gap Register (Stage 2)

**Document ID**: `PHASE2_FULL_GAP_REGISTER`  
**Audited Branch**: `reproduction/scientific-fidelity`  
**Purpose**: Systematically track, classify, and provide authoritative scientific resolutions for all remaining experimental and analytical gaps between the target paper and active implementation.

---

## 1. Gap Register Matrix

| Gap ID | Component / Area | Priority | Scientific Risk | Implementation Difficulty | Dependencies | Evidence Required | Recommended Resolution |
|---|---|---|---|---|---|---|---|
| **GAP-01** | **Statistical Significance & Attribution** (Comp X) | **P1 (Critical)** | High | Low | Step 14 dataset (`PHASE2_EXPERIMENT_INDEX.csv`) | Paired exact permutation test / Wilcoxon signed-rank $p$-values, 95% bootstrap CIs, Cohen's $d_z$ effect sizes. | Implement `scripts/run_phase2_statistical_analysis.py` consuming the 60-run locked dataset; produce `docs/PHASE2_STATISTICAL_ATTRIBUTION_REPORT.md`. |
| **GAP-02** | **Factorial Latency & Energy Trajectory Plots** (Comp AB) | **P1 (Critical)** | Medium | Low | Step 14 training & evaluation CSVs | High-resolution publication plots of delay, energy, completion ratio vs workload ($w \in \{20, 30, 40\}$) across both geometries. | Implement `scripts/generate_phase2_figures.py` to generate reproduction equivalents of Figures 6, 7, 8, 9 with dual-geometry panels. |
| **GAP-03** | **Comparative Markdown/LaTeX Tables IV & V** (Comp AC) | **P2 (High)** | Low | Low | Step 14 summary statistics | Formatted tabular comparison (CoTOP vs DDQN vs Local vs Greedy) under varying workloads. | Implement automated table generation script outputting formatted tables directly into `docs/PHASE2_TABLES_IV_V.md`. |
| **GAP-04** | **Module Ablation Suite** (Comp Y / Table VI) | **P2 (High)** | Medium | Medium | CoTOP checkpoint & runner | Multi-seed evaluation of CoTOP variants: (1) Full CoTOP, (2) w/o Mobility Detection (`MD`), (3) w/o Task Priority (`TP`), (4) w/o Collaboration (`CO`). | Implement `scripts/run_phase2_ablations.py` to evaluate the 4 ablation conditions across 5 seeds and produce Table VI reproduction. |
| **GAP-05** | **Hyperparameter Sensitivity Sweeps** (Comp Z / Figs 4, 5) | **P3 (Medium)** | Low | Medium | Simulation runner & config sweeps | Sensitivity trajectories for learning rate $\text{lr} \in \{10^{-4}, 2\cdot 10^{-4}, 10^{-3}, 10^{-2}\}$ and priority weight $\alpha \in [0.1, 0.9]$. | Implement parameter sweep runner `scripts/run_phase2_sensitivity.py` to reconstruct Figures 4 and 5. |
| **GAP-06** | **Dense Real-World Scale (>100 Veh)** (Comp AA / Fig 11) | **P3 (Medium)** | Low | Medium | SUMO large-scale network | Large-scale evaluation results under $N_V > 100$ in Hangzhou network. | Execute verification run with $N_V = 100$ using `hangzhou.sumocfg` to validate scalability and plot Figure 11 reproduction. |
| **GAP-07** | **QRMP-DQN Exclusion Formalization** (Comp S) | **Closed / Preserved** | Low | N/A | Ref [33] Forensic Audit | Step 3 Forensic Audit report documenting continuous-action mismatch. | Maintain hard exclusion from primary matrix; document exclusion in methodology section as a formal scientific finding. |

---

## 2. Risk & Impact Assessment

1. **Scientific Validity (GAP-01 & GAP-02)**:
   - *Risk*: Without formal non-parametric hypothesis testing and bootstrap confidence intervals, performance differences cannot be rigorously attributed between algorithmic treatment and environmental noise.
   - *Mitigation*: Step 14 already locked the 60-run dataset with paired realization hashes. GAP-01 will execute small-$n$ paired exact permutation tests and Cohen's $d_z$ to complete the attribution.

2. **Ablation Traceability (GAP-04)**:
   - *Risk*: Table VI in the paper attributes specific performance gains to Mobility Detection (MD), Task Priority (TP), and Collaborative Offloading (CO).
   - *Mitigation*: The codebase already includes toggles (`use_mobility_model`, $\alpha/\beta$ weighting, R2R offload masking). Running the controlled ablation sweep will verify whether each module contributes as claimed.

3. **QRMP-DQN Ref [33] Exclusion (GAP-07)**:
   - *Risk*: Implementing generic QR-DQN under the name "QRMP-DQN" would violate Rule 4 and introduce methodological pollution.
   - *Mitigation*: Exclusion is fully audited, documented, and locked in `PHASE2_FORENSIC_AUDIT.md`.

---

## 3. Recommended Execution Sequence

$$\boxed{\text{GAP-01 (Statistical Analysis)} \longrightarrow \text{GAP-02 (Figures)} \longrightarrow \text{GAP-03 (Tables IV–V)} \longrightarrow \text{GAP-04 (Ablations)}}$$
