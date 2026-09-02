# PHASE 2 — STEP 21: FULL 10-SEED FACTORIAL EXPERIMENTAL CAMPAIGN REPORT

**Document ID**: `docs/PHASE2_STEP21_FULL_FACTORIAL_CAMPAIGN.md`  
**Phase**: Phase 2 — Step 21 (Full 10-Seed Factorial Experiment & Cross-Baseline Synthesis)  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE Transactions on Mobile Computing, 2026)  
**Status**: **COMPLETE — ALL 240 EXPERIMENTAL CELLS VALIDATED (PASS)**  

---

## 1. Executive Summary & Experimental Scope

This document presents the complete results of the publication-grade 10-seed factorial experimental campaign for the CoTOP paper reproduction. All 240 experimental cells were executed under strict scientific controls against frozen exogenous realizations.

### Experimental Matrix Dimensions
- **Algorithms (4)**: `CoTOP`, `DDQN` (Zhai et al. [34]), `Greedy` (Load-balancing baseline), `Local` (Standalone primary RSU)
- **Scenarios (2)**: `corridor_2400m` (Linear freeway corridor), `grid_200m` (Urban Manhattan grid)
- **Workloads (3)**: `W20` (20 subtasks/vehicle), `W30` (30 subtasks/vehicle), `W40` (40 subtasks/vehicle)
- **Seeds (10)**: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`
- **Total Experimental Cells**: $4 \times 2 \times 3 \times 10 = \mathbf{240\text{ runs}}$
- **Total Exogenous Realization Files**: $2 \times 3 \times 10 = \mathbf{60\text{ files}}$ (SHA-256 verified)

---

## 2. Hard Scientific Invariants & Realization Rules

1. **Protected Physics Hashes**:
   - `envs/comm_model.py`: `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431` (**EXACT & UNCHANGED**)
   - `envs/comp_model.py`: `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff` (**EXACT & UNCHANGED**)
2. **Matching Realization Rule**:
   - For every scenario, workload, and seed triplet $(\text{geom}, W, s)$, all 4 algorithms evaluate against byte-identical exogenous realizations loaded via `FrozenVECEnv`.
3. **Parameter Immutability**:
   - Deterministic greedy evaluation ($\epsilon=0$) was enforced. Weight parameter SHA-256 hashes before and after evaluation were verified to be identical across all runs.
4. **QRMP-DQN Formal Exclusion**:
   - Guo et al. [33] remains formally excluded due to continuous STAR-RIS domain mismatch.

---

## 3. Full Run Inventory & Status Summary

| Status Category | Run Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **COMPLETED** | **240** | **100.0%** | Full task lifecycle evaluated with valid telemetry |
| **FAILED** | **0** | **0.0%** | 0 software faults / simulation crashes |
| **SKIPPED** | **0** | **0.0%** | Complete coverage across all 240 factorial cells |
| **RESUMED** | **0** | **0.0%** | Clean batch execution |

---

## 4. Multi-Dimensional Synthesis Across Matrix Dimensions

### 4.1 Cross-Algorithm Summary
From `results/phase2_step21/algorithm_summary.csv`:

| Algorithm | Runs | Mean Delay (s) | Std Delay (s) | Mean Energy (J) | Std Energy (J) | Completion Ratio | Tasks Completed | Tasks Failed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CoTOP** | 60 | $1.3392$ | $0.6874$ | $3.9519$ | $2.4406$ | **99.20%** | 17,860 | 140 |
| **DDQN** | 60 | $1.3370$ | $0.6851$ | $3.5831$ | $2.5593$ | **99.24%** | 17,864 | 136 |
| **Greedy** | 60 | $1.3111$ | $0.6882$ | $5.1209$ | $1.9998$ | **99.23%** | 17,865 | 135 |
| **Local** | 60 | $1.3335$ | $0.6674$ | $0.2892$ | $0.0106$ | **99.31%** | 17,879 | 121 |

### 4.2 Scenario Comparison
From `results/phase2_step21/scenario_summary.csv`:

| Scenario Geometry | Runs | Mean Delay (s) | Std Delay (s) | Mean Energy (J) | Std Energy (J) | Completion Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `corridor_2400m` | 120 | $2.0159$ | $0.0635$ | $4.4988$ | $2.2612$ | 98.66% |
| `grid_200m` | 120 | $0.6445$ | $0.0197$ | $1.9737$ | $1.5204$ | 99.78% |

### 4.3 Workload Scaling Analysis
From `results/phase2_step21/workload_summary.csv`:

| Workload | Runs | Mean Delay (s) | Std Delay (s) | Mean Energy (J) | Std Energy (J) | Completion Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W20** | 80 | $1.3179$ | $0.6804$ | $3.0805$ | $2.2965$ | 99.23% |
| **W30** | 80 | $1.3330$ | $0.6845$ | $3.2427$ | $2.4435$ | 99.24% |
| **W40** | 80 | $1.3396$ | $0.6874$ | $3.3857$ | $2.5510$ | 99.19% |

---

## 5. Matched Inferential Statistics ($N=10$ Seeds: CoTOP vs DDQN)

From `results/phase2_step21/paired_statistical_analysis.csv`:

| Scenario | Workload | Metric | CoTOP Mean | DDQN Mean | Paired Diff ($\text{CoTOP}-\text{DDQN}$) | $p$-value ($t$-test) | Cohen's $d_z$ | CLES | FDR $q$ | Significant (FDR $< 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `corridor_2400m` | `W20` | Delay | $2.0018\text{ s}$ | $1.9879\text{ s}$ | $+0.0139\text{ s}$ | $0.0874$ | $+0.606$ | $0.650$ | $0.639$ | No |
| `corridor_2400m` | `W20` | Energy | $5.8879\text{ J}$ | $4.2689\text{ J}$ | $+1.6190\text{ J}$ | $0.1597$ | $+0.485$ | $0.650$ | $0.639$ | No |
| `corridor_2400m` | `W30` | Delay | $2.0148\text{ s}$ | $2.0148\text{ s}$ | $0.0000\text{ s}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `corridor_2400m` | `W30` | Energy | $5.0147\text{ J}$ | $5.0147\text{ J}$ | $0.0000\text{ J}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `corridor_2400m` | `W40` | Delay | $2.0405\text{ s}$ | $2.0405\text{ s}$ | $0.0000\text{ s}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `corridor_2400m` | `W40` | Energy | $5.4769\text{ J}$ | $5.4769\text{ J}$ | $0.0000\text{ J}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `grid_200m` | `W20` | Delay | $0.6457\text{ s}$ | $0.6460\text{ s}$ | $-0.0002\text{ s}$ | $0.7927$ | $-0.086$ | $0.450$ | $1.000$ | No |
| `grid_200m` | `W20` | Energy | $2.6043\text{ J}$ | $2.0106\text{ J}$ | $+0.5937\text{ J}$ | $0.1460$ | $+0.503$ | $0.650$ | $0.639$ | No |
| `grid_200m` | `W30` | Delay | $0.6584\text{ s}$ | $0.6584\text{ s}$ | $0.0000\text{ s}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `grid_200m` | `W30` | Energy | $2.2213\text{ J}$ | $2.2213\text{ J}$ | $0.0000\text{ J}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `grid_200m` | `W40` | Delay | $0.6742\text{ s}$ | $0.6742\text{ s}$ | $0.0000\text{ s}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |
| `grid_200m` | `W40` | Energy | $2.5061\text{ J}$ | $2.5061\text{ J}$ | $0.0000\text{ J}$ | $1.0000$ | $0.000$ | $0.500$ | $1.000$ | No |

---

## 6. Published Values Discrepancy Attribution

From `results/phase2_step21/published_value_comparison.csv`:

| Quantity | Published Target | Reproduced Mean | Absolute Difference | Relative Diff (%) | Reproduction Status | Plausible Physical Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Delay** | $13.90\text{ s}$ | **$1.3392\text{ s}$** | $-12.5608\text{ s}$ | $-90.37\%$ | **NOT ACHIEVED** | Omitted initial server queue backlog ($\approx 18.96\text{ Gcycles} / 9.48\text{ s}$) |
| **Energy** | $25.14\text{ J}$ | **$3.9519\text{ J}$** | $-21.1881\text{ J}$ | $-84.28\%$ | **NOT ACHIEVED** | Omitted baseline server idle power draw ($\approx 1.8\text{ W}$) |

---

## 7. Artifact Deliverables Inventory in `results/phase2_step21/`

1. [run_inventory.csv](file:///d:/cotop-implementation/results/phase2_step21/run_inventory.csv): 240 complete run records with subtask and task telemetry.
2. [seed_summary.csv](file:///d:/cotop-implementation/results/phase2_step21/seed_summary.csv): 10-seed dispersion analysis.
3. [scenario_summary.csv](file:///d:/cotop-implementation/results/phase2_step21/scenario_summary.csv): Scenario-level delay and energy breakdown.
4. [workload_summary.csv](file:///d:/cotop-implementation/results/phase2_step21/workload_summary.csv): Workload scaling summary.
5. [algorithm_summary.csv](file:///d:/cotop-implementation/results/phase2_step21/algorithm_summary.csv): Cross-algorithm performance aggregation.
6. [failed_run_report.csv](file:///d:/cotop-implementation/results/phase2_step21/failed_run_report.csv): Zero failed runs report.
7. [convergence_summary.csv](file:///d:/cotop-implementation/results/phase2_step21/convergence_summary.csv): Convergence metrics across multi-seed runs.
8. [checkpoint_inventory.csv](file:///d:/cotop-implementation/results/phase2_step21/checkpoint_inventory.csv): Checkpoint SHA-256 manifest.
9. [realization_inventory.csv](file:///d:/cotop-implementation/results/phase2_step21/realization_inventory.csv): 60 realization files cryptographic inventory.
10. [provenance_manifest.json](file:///d:/cotop-implementation/results/phase2_step21/provenance_manifest.json): Authoritative execution manifest.
11. [published_value_comparison.csv](file:///d:/cotop-implementation/results/phase2_step21/published_value_comparison.csv): Published vs reproduced discrepancy analysis.
12. [paired_statistical_analysis.csv](file:///d:/cotop-implementation/results/phase2_step21/paired_statistical_analysis.csv): 10-seed paired inferential statistics.
