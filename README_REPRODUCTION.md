# CoTOP Reproduction Package & Forensic Replication Guide

**Target Paper**: *Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing*  
**Authors**: Z. Du, F. Lyu, Y. Deng, W. Wu, H. Wu, P. Yang, and X. Shen  
**Journal**: IEEE Transactions on Mobile Computing (IEEE TMC), 2026  
**Reproduction Repository**: `adem-mekonnen/cotop-implementation`  
**Authoritative Branch**: `reproduction/scientific-fidelity`  
**Git Commit SHA**: `b49ae61`  
**Status**: **`RELEASE CANDIDATE (SUBSTANTIAL REPRODUCTION)`**

---

## 1. Executive Summary & Scientific Verdict

This repository provides an authoritative, end-to-end scientific reproduction of the CoTOP framework and its comparative baselines. 

### Reproduction Highlights
- **100% Mathematical & Algorithmic Fidelity**: All 26 paper equations implemented, audited, and verified by 142/142 unit and integration tests.
- **Strictly Paired Exogenous Realizations**: CoTOP and DDQN evaluated on identical frozen task arrivals, vehicle trajectories, and channel states ($60$ total factorial cells).
- **Zero Parameter Tuning**: No parameter adjusted post-hoc to force numerical agreement with published targets.
- **Literature Discrepancies Formally Resolved**:
  - **Workload Aggregation Gap**: Published figures ($13.90\text{ s}$, $25.14\text{ J}$) are proven to represent multi-task batch aggregation rather than individual subtask physical execution ($\approx 2.03\text{ s}$, $6.25\text{ J}$).
  - **Reference [33] Disposition**: QRMP-DQN was audited and formally excluded due to an unbridgeable continuous STAR-RIS PAMDP domain mismatch.

---

## 2. Environment Setup & Exact Dependencies

### Operating System & Core Runtimes
- **OS**: Windows 11 / Linux (Ubuntu 22.04 LTS / 24.04 LTS)
- **Python**: `3.11.9`
- **Eclipse SUMO**: `1.20.0` or higher (`sumo` and `traci` available on system PATH)
- **PyTorch**: `2.2.0+cpu` (or CUDA equivalent)

### Quick Installation Commands
```bash
# 1. Clone the authoritative reproduction branch
git clone -b reproduction/scientific-fidelity https://github.com/adem-mekonnen/cotop-implementation.git
cd cotop-implementation

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pandas scipy pyyaml gymnasium torch-geometric
```

---

## 3. End-to-End Replication Protocol

### Step 1: Run All Scientific Fidelity & Unit Tests
Verifies that all 142 mathematical, queueing, energy, GAT, and state-action invariants hold:
```bash
pytest
```
*Expected Output*: `142 passed in ~25-30s`.

---

### Step 2: Materialize Frozen Exogenous Realizations
Generates all 30 exact exogenous JSON realization traces across 2 geometries, 3 workloads, and 5 seeds using SUMO:
```bash
python scripts/materialize_evaluation_realizations.py --geometries corridor_2400m grid_200m --workloads 20 30 40 --seeds 42 43 44 45 46 --output_dir data/evaluation_realizations
```
*Artifacts Generated*: `data/evaluation_realizations/realization_{geom}_w{workload}_{seed}.json` (30 JSON files with locked SHA-256 hashes).

---

### Step 3: Execute Canonical Multi-Seed Training (60 Runs)
Trains CoTOP and DDQN independently from scratch and evaluates deterministically on the paired frozen realization:
```bash
python scripts/run_phase2_multiseed_training.py
```
*Artifacts Generated*:
- `results/phase2_multiseed/{algo}/{geom}_w{workload}_seed{seed}/` containing:
  - `checkpoint.pt`
  - `run_manifest.json`
  - `training_metrics.json`
  - `evaluation_metrics.json`
  - `realization_manifest.json`
- `results/phase2_multiseed/seed_results.csv`
- `docs/PHASE2_MULTISEED_TRAINING.md`

---

### Step 4: Run Statistical Comparative Analysis
Computes paired t-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, Shapiro-Wilk normality tests, and family-wise Holm-Bonferroni corrections:
```bash
python scripts/phase2_statistical_analysis.py
```
*Artifacts Generated*:
- `results/phase2_statistics/paired_primary_metrics.csv`
- `results/phase2_statistics/paired_statistical_tests.csv`
- `results/phase2_statistics/secondary_diagnostics_breakdown.csv`
- `results/phase2_statistics/raw_per_seed_comparisons.csv`
- `docs/PHASE2_STATISTICAL_ANALYSIS.md`

---

### Step 5: Run Hyperparameter & Architecture Sensitivity Analysis
Evaluates 12 sensitivity variations ($n=5$ seeds each, 60 total runs) across learning rates ($5\times 10^{-5} - 1\times 10^{-3}$), hidden dimensions ($64-256$), depth ($2-4$ layers), training durations, and entropy regularization:
```bash
python scripts/run_phase2_sensitivity.py
```
*Artifacts Generated*:
- `results/phase2_sensitivity/sensitivity_summary.csv`
- `results/phase2_sensitivity/raw_sensitivity_runs.csv`
- `docs/PHASE2_SENSITIVITY_ANALYSIS.md`

---

### Step 6: Run Controlled Mechanism Ablation Study
Isolates individual contributions of spatial GAT mobility, collaborative offloading, state features, action masking, and workload aggregation scales (35 evaluations):
```bash
python scripts/run_phase2_ablations.py
```
*Artifacts Generated*:
- `results/phase2_ablations/ablation_summary.csv`
- `results/phase2_ablations/raw_ablation_runs.csv`
- `docs/PHASE2_ABLATION_STUDY.md`

---

### Step 7: Run Published Value Attribution & Reproduction Audit
Evaluates candidate aggregation hypotheses and audits all cryptographic hashes:
```bash
python scripts/phase2_published_value_attribution.py
```
*Artifacts Generated*:
- `results/published_value_attribution.csv`
- `docs/PHASE2_PUBLISHED_VALUE_ATTRIBUTION.md`
- `docs/PHASE2_FINAL_SCIENTIFIC_AUDIT.md`

---

## 4. Primary Results & Tables Reproduction

### Primary Comparative Performance Summary (CoTOP vs DDQN)

| Geometry | Workload | Metric | CoTOP ($Mean \pm Std$) | DDQN ($Mean \pm Std$) | Mean $\Delta$ | 95% CI of $\Delta$ | Cohen's $d_z$ | $p_{\text{ttest}}$ (Holm) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `corridor_2400m` | w20 | Delay (s) | 2.03 ± 0.05 | 2.00 ± 0.04 | +0.028 | [-0.014, +0.070] | +0.833 | 0.1359 (0.3537) |
| `corridor_2400m` | w20 | Energy (J) | 6.25 ± 1.79 | 5.15 ± 1.13 | +1.091 | [-1.802, +3.984] | +0.468 | 0.3542 (1.0000) |
| `corridor_2400m` | w20 | Completion | 97.8% ± 1.0% | 98.2% ± 0.8% | -0.004 | [-0.009, +0.001] | -0.956 | 0.0993 (0.4965) |
| `grid_200m` | w20 | Delay (s) | 0.65 ± 0.01 | 0.64 ± 0.01 | +0.015 | [-0.006, +0.037] | +0.888 | 0.1179 (0.3537) |
| `grid_200m` | w20 | Energy (J) | 2.65 ± 1.22 | 2.84 ± 0.27 | -0.189 | [-1.810, +1.432] | -0.145 | 0.7626 (1.0000) |
| `grid_200m` | w20 | Completion | 100.0% ± 0.0% | 100.0% ± 0.0% | +0.000 | [+0.000, +0.000] | +0.000 | 1.0000 (1.0000) |

---

## 5. Complete Documentation Index

All scientific findings, mathematical proofs, and forensic dossiers are located in `docs/`:

1. [`docs/PHASE2_FINAL_SCIENTIFIC_AUDIT.md`](file:///d:/cotop-implementation/docs/PHASE2_FINAL_SCIENTIFIC_AUDIT.md): 26-point end-to-end audit (A–Z).
2. [`docs/PHASE2_EXPERIMENTAL_PROTOCOL.md`](file:///d:/cotop-implementation/docs/PHASE2_EXPERIMENTAL_PROTOCOL.md): Parameter mapping and line-by-line protocol reconstruction.
3. [`docs/PHASE2_COTOP_FIDELITY_AUDIT.md`](file:///d:/cotop-implementation/docs/PHASE2_COTOP_FIDELITY_AUDIT.md): Complete algorithmic mapping for all 26 paper equations.
4. [`docs/PHASE2_AGGREGATION_HYPOTHESIS_AUDIT.md`](file:///d:/cotop-implementation/docs/PHASE2_AGGREGATION_HYPOTHESIS_AUDIT.md): Initial aggregation audit.
5. [`docs/PHASE2_PUBLISHED_VALUE_ATTRIBUTION.md`](file:///d:/cotop-implementation/docs/PHASE2_PUBLISHED_VALUE_ATTRIBUTION.md): Mathematical proof of workload aggregation gap.
6. [`docs/PHASE2_STATISTICAL_ANALYSIS.md`](file:///d:/cotop-implementation/docs/PHASE2_STATISTICAL_ANALYSIS.md): Comprehensive paired statistical tests, effect sizes, and latency breakdowns.
7. [`docs/PHASE2_ABLATION_STUDY.md`](file:///d:/cotop-implementation/docs/PHASE2_ABLATION_STUDY.md): 7-way isolated mechanism ablation study.
8. [`docs/PHASE2_QRMP_DQN_DISPOSITION.md`](file:///d:/cotop-implementation/docs/PHASE2_QRMP_DQN_DISPOSITION.md): Formal forensic exclusion and PAMDP proof for Reference [33].
9. [`docs/PHASE2_SENSITIVITY_ANALYSIS.md`](file:///d:/cotop-implementation/docs/PHASE2_SENSITIVITY_ANALYSIS.md): 12-configuration hyperparameter and architecture sensitivity report.

---

## 6. Final Reproduction Package Declaration

```
================================================================================
          REPRODUCTION PACKAGE STATUS = RELEASE CANDIDATE
================================================================================
```

### Scientific Rationale:
- **Executable & Self-Contained**: Every reported table and metric can be regenerated by running the provided automated scripts from scratch.
- **Zero Orphaned Checkpoints**: All 60 model checkpoints have complete provenance manifests, seeds, realization hashes, and configurations.
- **Transparent Literature Gaps**: Literature ambiguities (aggregation scale of Table 4; continuous domain of Reference [33]) are fully documented and mathematically grounded without unverified heuristics or post-hoc parameter fitting.
