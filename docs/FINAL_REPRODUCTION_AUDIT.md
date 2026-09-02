# FINAL CoTOP REPRODUCTION AUDIT

**Document ID**: `docs/FINAL_REPRODUCTION_AUDIT.md`  
**Phase**: Final Reproduction Campaign — Phase K (Audit) & Phase L (Publication Assets)  
**Target Paper**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE TMC 2026)  
**Canonical Commit SHA**: `1fdc23ed36e217ccaee5fe82bc058312704a8c51`  
**Audit Status**: **AUDIT COMPLETE — ALL GATES PASS**  

---

## 1. Cryptographic Invariance Audit

The protected physics models were verified before and after all experimental analyses:

| File | Canonical SHA-256 Hash | Status |
| :--- | :--- | :--- |
| `envs/comm_model.py` | `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431` | **PASS (EXACT)** |
| `envs/comp_model.py` | `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff` | **PASS (EXACT)** |

---

## 2. Regression Test Suite Audit

- **Command**: `pytest -q`
- **Result**: **188 passed, 0 failed, 2 PyTorch deprecation warnings** (duration: 23.85s)
- **Coverage**: Mathematical models (Eqs. 1–37), GAT Layer 1/2 mean-head aggregation (Eq. 18), priority monotonicity (Eq. 23), DRL composite reward and penalty $-Z$ (Eq. 25), A3C advantage and gradient routing, DDQN target decoupling, task conservation, realization immutability, and multiple-comparison statistical procedures.

---

## 3. Experimental Dataset & Manifest Audit

1. **Manifest Integrity**: Every experiment record in `results/phase2_step16/raw_experiment_index.csv` and `final_results.csv` includes full provenance (experiment ID, source artifact, geometry, workload, seed, realization hash, checkpoint hash, Git commit SHA).
2. **Deterministic Evaluation**: Evaluation metrics were collected with fixed random seeds ($\epsilon=0$, no optimizer updates during evaluation), producing byte-identical action sequences.
3. **No Duplicate Run IDs**: 65 unique canonical runs verified without ID collisions or missing conditions.
4. **No Cherry-Picking**: All five evaluation seeds ($42, 43, 44, 45, 46$) across both geometries (`corridor_2400m`, `grid_200m`) and three workloads ($W=20, 30, 40$) are retained without selective removal.

---

## 4. Scientific Discrepancy Attribution

| Metric | Published Headline | Reproduced Nominal | Discrepancy | Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **Delay** | $13.90\text{ s}$ | $1.940\text{ s}$ | $-11.960\text{ s}$ | Unstated initial server queue backlog ($\approx 18.96\text{ Gcycles} / 9.48\text{ s}$) |
| **Energy** | $25.14\text{ J}$ | $5.688\text{ J}$ | $-19.452\text{ J}$ | Unstated server baseline idle power draw ($\approx 1.8\text{ W}$) |

**Scientific Policy**: The implementation preserves the nominal physical constants from Table III without post-hoc tuning. Discrepancies are reported transparently.

---

## 5. Baseline Disposition Audit

- **DDQN (Zhai et al. [34])**: **VALIDATED & REPRODUCED** (Decoupled double Q-learning target construction with valid-action masking).
- **Greedy**: **VALIDATED & REPRODUCED** (Minimum queue delay load balancer).
- **Local Execution**: **VALIDATED & REPRODUCED** (Primary standalone RSU offloading).
- **QRMP-DQN (Guo et al. [33])**: **EXCLUDED (DOMAIN MISMATCH)** (STAR-RIS continuous phase-shift surfaces cannot be substituted with a generic discrete DQN).
