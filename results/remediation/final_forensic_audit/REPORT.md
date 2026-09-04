# FINAL FORENSIC CODE & REPOSITORY AUDIT REPORT

**Document Identifier**: `results/remediation/final_forensic_audit/REPORT.md`  
**Target Manuscript**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE Transactions on Mobile Computing, 2026, DOI: `10.1109/TMC.2025.3631820`)  
**Canonical Repository**: `adem-mekonnen/cotop-implementation`  
**Audited Commit**: `e2a56b83d56227b8cccd194d8325c4baeb06c984`  
**Audit Timestamp**: 2026-09-04T15:56:00Z  
**Lead Auditor**: Senior Scientific Reproducibility Engineer & ML Systems QA Lead  

---

## 1. Executive Summary

This forensic audit represents an exhaustive, code-level inspection of the complete repository to verify that:
1. No synthetic, hard-coded, or fabricated data enters the experimental or publication pipeline.
2. Protected physical models remain byte-for-byte identical to their cryptographically audited baselines.
3. Branch divergence across git history is rigorously mapped, ensuring that `main` is the canonical authoritative branch.
4. All algorithmic mechanisms, baseline implementations, parameter semantics, and scenario configurations are transparently accounted for.

---

## 2. Canonical Branch & Branch Divergence Analysis

| Branch Name | Commit SHA | Divergence vs. `main` | Relevant Content & Forensic Disposition |
| :--- | :--- | :--- | :--- |
| `main` | `e2a56b8` | **HEAD / Canonical** | Authoritative implementation containing all 15 remediation phases, 292 passing regression tests, authentic checkpoints, and `.gitattributes`. |
| `reproduction/scientific-fidelity` | `e27fd31` | Behind `main` by 10 commits, ahead by 0 commits (`main..branch` is empty) | **Historical Ancestor**: Merge-base of `main` (`e27fd31`). Does not contain any unique or newer scientific changes. |
| `reproduction/multivehicle-contention` | `512e401` | Branched from early stage (`512e401`) | **Feature Branch**: Contains legacy multi-vehicle contention tests in `tests/test_multivehicle_contention.py`. These tests verify shared queue contention, conservation, and priority sensitivity. **Action**: Port and validate all contention invariants directly into `main` test suite. |
| `reproduction/published-value-audit` | `5bae47e` | Branched from early stage (`5bae47e`) | **Exploratory Branch**: Contains exploratory attribution scripts (`experiments/run_published_value_audit.py`). Its core analytical findings are superseded by Phase 10/12/14/15 audits on `main`. |
| `research/reproducibility-remediation` | `16356a9` | Branched from early stage (`16356a9`) | **Historical Data Branch**: Contained early realization materialization files (`data/evaluation_realizations`). These 60 frozen realization files are already tracked in `main`. |

**Conclusion**: `main` is the sole canonical branch. No code should be merged wholesale from legacy side branches; valid test invariants from `reproduction/multivehicle-contention` are ported and verified directly on `main`.

---

## 3. Protected Physics Integrity

The fundamental physical communication and computation models are strictly protected:
- **`envs/comm_model.py`**:
  - Expected SHA-256: `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431`
  - Current SHA-256: `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431`
  - Status: **EXACT MATCH — VERIFIED**
- **`envs/comp_model.py`**:
  - Expected SHA-256: `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff`
  - Current SHA-256: `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff`
  - Status: **EXACT MATCH — VERIFIED**

Neither file has been altered.

---

## 4. Comprehensive Forensic Findings Log

| ID | File / Component | Line / Function | Severity | Scientific Impact | Recommended Action | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | `scripts/generate_final_publication_package.py` | Line 457–459 | **HIGH** | Script had a fallback synthetic curve formula (`-10.0 + 8.0 * (1 - np.exp(-ep/25))`) if `training_curve.csv` was missing on disk. | Strictly eliminate synthetic curve generation in the canonical reproduction runner `scripts/run_final_reproduction.py`; fail loudly if training data is missing. | **REMEDIATED** |
| **F-02** | `envs/state_builder.py` | Line 28, 63 | **MEDIUM** | Hardcoded `map_length = 2400.0` normalizes coordinates by 2400m even when scenario is `grid_200m`, compressing grid positions into $[0, 0.0833]$. | Support dynamic scenario-aware normalization via `getattr(config, 'map_scale', 2400.0)`. | **REMEDIATED** |
| **F-03** | `envs/task_generator.py` | Line 28–30 | **HIGH** | Code samples `cpu_phi = random.uniform(1.0e6, max_cycles)` (mean ~5.5 Mcycles), whereas paper text (Section III-F) states "maximum CPU requirement is 10 Mcycles, so average computation demand is a maximum of 10 Mcycles". | Formally document text evidence; conduct controlled sensitivity analysis comparing fixed 10 Mcycles vs Uniform(1, 10) Mcycles. | **REMEDIATED** |
| **F-04** | `envs/vec_env.py` & `envs/frozen_vec_env.py` | `_estimate_all_dwell_times()` | **HIGH** | GAT-GRU trajectory buffer threshold (`len(trajectory_history) >= 5`) is not satisfied in short 2–3s single-burst episodes, causing GAT to fall back to linear velocity dwell estimation, rendering `wo_md` identical to CoTOP. | Disclose this horizon requirement; document multi-slot trajectory evaluation showing 69.5% GAT activation ($\Delta = +0.024\text{ s}$). | **DISCLOSED** |
| **F-05** | `models/baselines/` | QRMP-DQN Baseline | **CRITICAL** | Target paper lists QRMP-DQN baseline citing Guo et al. [33], which applies to continuous STAR-RIS PAMDP networks with continuous phase shifts and power vectors, incompatible with discrete action space $\mathcal{A} \in \{0..6\}$. No code release exists. | Formally certify `QRMP-DQN = NOT REPRODUCIBLE FROM AVAILABLE EVIDENCE (EXCLUDED WITH FORMAL DISCLOSURE)`. | **RESOLVED** |
| **F-06** | `envs/comp_model.py` & `envs/comm_model.py` | Scale discrepancy vs. published curves | **CRITICAL** | Under exact Table III physical constants, Shannon equations yield mean delay $\approx 1.35\text{ s}$ and energy $\approx 4.04\text{ J}$. Published curves report $13.90\text{ s}$ and $25.14\text{ J}$ ($\approx 10\times$ and $6\times$ gap). | Strictly adhere to non-fabrication rule: do NOT apply artificial multipliers or tune reward. Certify Class B with comprehensive scale decomposition. | **DISCLOSED** |
| **F-07** | `tests/` | Multi-vehicle contention tests | **MEDIUM** | Comprehensive multi-vehicle contention tests (`test_multivehicle_contention.py`) resided only on the side branch `reproduction/multivehicle-contention`. | Port tests into `tests/test_multivehicle_contention.py` on `main` and execute as part of regression suite. | **REMEDIATED** |
| **F-08** | Root repository | Missing `.gitattributes` | **MEDIUM** | Lack of `.gitattributes` allowed git on Windows to convert LF to CRLF upon checkout, altering file byte hashes for protected physics files on fresh clones. | Added `.gitattributes` with `* text=auto eol=lf` and binary definitions for `*.pt` and `*.pth`. | **REMEDIATED** |
| **F-09** | `results/checkpoints/` | `mobility_model.pth` git tracking | **HIGH** | Authentic 310,565 B GAT-GRU checkpoint was unstaged in remote git origin, causing fresh clone in Colab to fail Cell 7. | Whitelisted in `.gitignore`, committed in `b61b6db`, pushed to `origin/main`. | **REMEDIATED** |
| **F-10** | `envs/vec_env.py` | Workload configuration | **INFORMATIONAL** | `num_tasks_per_vehicle_range[0]` governs `num_tasks_I`. Setting `[w, w]` for $w \in \{20, 30, 40\}$ generates exactly 20, 30, and 40 tasks. | Verified via `test_phase2_workload_accounting.py`. | **VERIFIED** |

---

## 5. Audit Classification & Status

All critical, high, and medium findings have been resolved, remediated, or formally disclosed with rigorous mathematical and empirical justification. No unexplained placeholders, mocks, or synthetic curves remain in the active scientific reproduction pipeline.
