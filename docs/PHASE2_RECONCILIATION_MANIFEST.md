# PHASE 2: RECONCILIATION MANIFEST

**Document ID**: `DOC-PHASE2-RECONCILIATION-MANIFEST-001`  
**Target Branch**: `merge/reconcile-reproduction-branches`  
**Date**: August 31, 2026  

---

## 1. Reconciliation Component Decision Matrix

| Component | Source Branch | Action | Reason |
| :--- | :--- | :---: | :--- |
| **Physics Core (`comm_model.py`, `comp_model.py`)** | `reproduction/scientific-fidelity` | **RETAIN** | SHA-256 protected baseline; zero modification permitted. |
| **Environment Engine (`envs/vec_env.py`)** | `reproduction/scientific-fidelity` | **RETAIN** | Full multi-vehicle dynamic task ownership, entry/exit tracking, queue draining. |
| **Task Prioritization (`utils/task_priority.py`)** | `reproduction/scientific-fidelity` | **RETAIN** | Paper-literal Eq. 23 implementation ($P = \alpha \frac{\phi}{T_{\text{stay}}} + \beta \frac{\rho}{d}$) with unit preservation. |
| **DDQN Baseline (`models/baselines/ddqn_agent.py`)** | `reproduction/scientific-fidelity` | **RETAIN** | Fully audited, tested, and benchmarked comparative baseline. |
| **Evaluation Realizations (`data/evaluation_realizations/`)** | `reproduction/published-value-audit` | **SELECTIVE IMPORT** | Standardized 30 canonical JSON traces across 2 geometries, 3 workloads, and 5 seeds. |
| **Statistical Analysis (`utils/statistical_analysis.py`)** | `reproduction/multivehicle-contention` | **REIMPLEMENT** | Clean, independent, unit-tested module supporting paired t-tests, Wilcoxon, Cohen's $d_z$, CLES, Holm, and FDR. |
| **Unverified Checkpoints (`.pt` files)** | `published-value-audit` / `multivehicle-contention` | **EXCLUDE** | Lack reproducible provenance, seed manifests, or configuration records. |
| **QRMP-DQN Baseline** | Literature Forensic Gate | **EXCLUDE** | Reference [33] continuous STAR-RIS PAMDP domain mismatch; no discrete mapping exists. |

---

## 2. Integrity Verification Checklist

- [x] Hard Safety Rule 1: `envs/comm_model.py` and `envs/comp_model.py` untouched.
- [x] Hard Safety Rule 2: SHA-256 hashes byte-identical to locked reference.
- [x] Hard Safety Rule 3: Scientific-fidelity versions of critical files preserved.
- [x] Hard Safety Rule 4: No parameters tuned toward published headline targets ($13.90\text{ s}$, $25.14\text{ J}$).
- [x] Hard Safety Rule 5: Generic QR-DQN rejected as surrogate for QRMP-DQN.
- [x] Hard Safety Rule 6: Unverifiable checkpoints excluded from primary matrices.
- [x] Hard Safety Rule 7: Git history and commits preserved without force-pushes.
- [x] Hard Safety Rule 8: Conflict resolutions inspected and documented scientifically.
