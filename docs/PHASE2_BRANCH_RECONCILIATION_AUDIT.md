# PHASE 2: BRANCH RECONCILIATION AUDIT & INVENTORY

**Document ID**: `DOC-PHASE2-BRANCH-RECONCILIATION-001`  
**Classification**: Multi-Branch Codebase Reconciliation Audit  
**Reconciliation Target Branch**: `merge/reconcile-reproduction-branches`  
**Audit Date**: August 31, 2026  

---

## 1. Branch Commits and Common Ancestry

| Branch Name | Head Commit SHA | Common Ancestor Commit SHA | Relationship to Common Ancestor |
| :--- | :---: | :---: | :--- |
| **`main`** | `bd34c654e34702428967d1cccac49c57202d8784` | `bd34c654e34702428967d1cccac49c57202d8784` | Author Base Commit |
| **`reproduction/scientific-fidelity`** | `e27fd31dd9bcb392abe1c65cc64dfa6fb0cce12d` | `bd34c654e34702428967d1cccac49c57202d8784` | Authoritative Scientific Baseline (+13 Phase 2 Steps) |
| **`reproduction/published-value-audit`** | `16356a90b3929d21bae72b1f52876ea64363cae5` | `bd34c654e34702428967d1cccac49c57202d8784` | PVA Aggregation Exploration Branch |
| **`reproduction/multivehicle-contention`** | `56a5a84e43c7113123bc18d5ffedd1cdd8d6693e` | `bd34c654e34702428967d1cccac49c57202d8784` | Colab Multi-Vehicle Contention Experiment Branch |

---

## 2. File Change Statistics Relative to `main`

- **`reproduction/scientific-fidelity`**: 134 files changed (Environment corrections, GAT multi-node, DDQN, frozen realizations, 142 tests, statistical suites).
- **`reproduction/published-value-audit`**: 88 files changed (PVA scripts, realization generators, 500-epoch legacy logs).
- **`reproduction/multivehicle-contention`**: 22 files changed (Colab contention runner, statistical CSV summaries).
- **Files Modified Across Multiple Branches**: 68 files.

---

## 3. Critical Conflicting Files & Scientific Reconciliation Decisions

| File Path | Branches Modifying File | Scientific Significance | Reconciliation Decision | Technical Justification |
| :--- | :--- | :--- | :---: | :--- |
| **`envs/comm_model.py`** | Protected File | Wireless channel physics (V2R, R2R) | **LOCKED (RETAIN)** | Hash locked to `041e41...431`. Zero changes permitted. |
| **`envs/comp_model.py`** | Protected File | Standalone & Collaborative latency & energy | **LOCKED (RETAIN)** | Hash locked to `dd9f58...bff`. Zero changes permitted. |
| **`envs/vec_env.py`** | `scientific-fidelity`, `published-value-audit`, `multivehicle-contention` | Multi-vehicle queueing, dynamic task ownership, coverage predicates | **RETAIN `scientific-fidelity`** | Only `scientific-fidelity` implements full multi-node GAT, dynamic entry/exit, and dual Eq. 23/25 modes. |
| **`utils/task_priority.py`** | `scientific-fidelity`, `published-value-audit`, `multivehicle-contention` | Task prioritization (Eq. 23) | **RETAIN `scientific-fidelity`** | Preserves exact paper-literal formula ($P = \alpha \frac{\phi}{T_{\text{stay}}} + \beta \frac{\rho}{d}$) with unit scaling alongside normalized candidate. |
| **`evaluate.py`** | `scientific-fidelity`, `published-value-audit`, `multivehicle-contention` | Evaluation loop & telemetry collection | **RETAIN `scientific-fidelity`** | Includes dynamic action masking, frozen trace loading, and zero-mutation inference gates. |
| **`models/mobility_gat.py`** | `scientific-fidelity`, `published-value-audit`, `multivehicle-contention` | Spatial GAT + Temporal GRU predictor | **RETAIN `scientific-fidelity`** | Implements genuine $N$-node spatial graph with Layer 2 mean-head averaging (Eq. 18) and GRU encoder-decoder. |
| **`data/evaluation_realizations/`** | `scientific-fidelity`, `published-value-audit` | Frozen exogenous evaluation traces | **SELECTIVE IMPORT** | Standardized 30 canonical JSON traces across 2 geometries, 3 workloads, and 5 seeds. |
| **`utils/statistical_analysis.py`** | `multivehicle-contention` (methodology) | Paired comparative statistics | **REIMPLEMENT INDEPENDENTLY** | Clean modular implementation supporting paired t-tests, Wilcoxon, Cohen's $d_z$, CLES, Holm, and FDR. |
| **Checkpoints (`.pt` files)** | All branches | Model weight artifacts | **EXCLUDE UNVERIFIED** | Only the 60 canonical checkpoints with full manifests and seeds are retained. |

---

## 4. Test Status Before & After Reconciliation

- **Test Suite Status**: `142 passed, 0 failed` across 33 test files.
- **Physics Invariants**: `comm_model.py` and `comp_model.py` hashes byte-identical to locked baseline.
