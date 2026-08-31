# Phase 2 — Step 14: Multi-Seed Training & Convergence Diagnostics Report

**Execution Timestamp**: 2026-08-29  
**Branch**: `reproduction/scientific-fidelity`  
**Git Commit SHA**: `52f2d3c81f0b8843edd08594cccedbaca4888ea8`  
**Hardware Fingerprint**: 12 CPU Cores, Python 3.11.9, PyTorch 2.12.1+cpu, SUMO 1.27.1  

---

## 1. Executive Summary

Step 14 has successfully executed and verified the full multi-seed training and evaluation pipeline for the primary two-algorithm factorial matrix:

$$\boxed{\text{2 Geometries} \times \text{2 Algorithms} \times \text{3 Workloads} \times \text{5 Seeds} = 60 \text{ Planned \& Completed Runs}}$$

All 60 runs completed 500 training episodes, saved Episode-500 checkpoints, executed deterministic evaluation passes on pre-materialized locked evaluation traces, and recorded complete run manifests and convergence metrics.

---

## 2. Factorial Matrix Cardinality & Completeness

| Dimension | Levels / Conditions | Cardinality |
|---|---|---|
| **Algorithms** | `CoTOP` (Control), `DDQN` (Audited Baseline) | 2 |
| **Geometries** | `linear_corridor` (`corridor_2400m`), `urban_manhattan` (`grid_200m`) | 2 |
| **Workloads** | $w=20$ (200 tasks), $w=30$ (300 tasks), $w=40$ (400 tasks) | 3 |
| **Random Seeds** | 42, 43, 44, 45, 46 | 5 |
| **Total Factorial Runs** | $2 \times 2 \times 3 \times 5$ | **60 (100.0% Complete)** |

The consolidated dataset is materialized in:
- `results/phase2_algorithmic_fidelity/PHASE2_EXPERIMENT_INDEX.csv`
- Individual run directories: `results/phase2_algorithmic_fidelity/<geometry>/<algorithm>/w<workload>/seed_<seed>/`

---

## 3. Causal Pairing & Evaluation Realization Locking

To guarantee internal validity and eliminate realization confounding between CoTOP and DDQN:
1. For every condition $(G, W, S)$, the exogenous evaluation trace $\mathcal{E}_{(G, W, S)}^{\text{exo}}$ was pre-materialized and locked with SHA-256.
2. Both `CoTOP` and `DDQN` evaluated on the exact same trace:
   $$H\left(\mathcal{E}_{(G, W, S), \text{CoTOP}}^{\text{exo}}\right) \equiv H\left(\mathcal{E}_{(G, W, S), \text{DDQN}}^{\text{exo}}\right)$$
3. Model evaluation was strictly isolated ($\epsilon = 0.0$, `torch.no_grad()`, `model.eval()`), verifying:
   $$H(\theta_{\text{before}}) \equiv H(\theta_{\text{after}})$$

---

## 4. Checkpoint Integrity & Hash Ledger

All 60 runs produced valid PyTorch checkpoint files (`checkpoint_ep500.pt`) with unique recorded SHA-256 hashes in `run_manifest.json` and `PHASE2_EXPERIMENT_INDEX.csv`.

Sample Checkpoint Hashes:
- `linear_corridor / CoTOP / w20 / seed_42`: `0ed23389184ffb8977a6d3fe8b6abaa0dce4cafa5dc86df4430b03a0b2fbd427`
- `linear_corridor / DDQN / w20 / seed_42`: `fc05b745a80e196432ac57ca1bbcbd06e49595a5e46e5a9ac1c107ed238e761a`
- `urban_manhattan / CoTOP / w40 / seed_46`: `50ba77780b2b03b1e155380487cb0b2d2607e3f79878b3505c4c0558cae7fdb9`
- `urban_manhattan / DDQN / w40 / seed_46`: `04e5d5d360f20d9a50ad68b325f4707bab02ed2d306bc55e566d2154eb77fe09`

---

## 5. Environment & Decomposition Invariants Verification

Across all 60 evaluation runs, all 12 Gates (13.1–13.12) passed with zero violations:
1. **Task Conservation**: $N_{\text{generated}} = N_{\text{completed}} + N_{\text{failed}} + N_{\text{pending}}$ holds strictly ($\Delta = 0$).
2. **Latency Decomposition**: $\max |T_{\text{total}} - (T_{\text{comm}} + T_{\text{wait}} + T_{\text{comp}})| < 10^{-15}\text{ s} \le 10^{-4}\text{ s}$.
3. **Energy Non-Negativity**: All energy components $\ge 0.0\text{ J}$.
4. **Numerical Stability**: 0 `NaN` and 0 `Inf` occurrences during both training and evaluation.
5. **Protected Physics Hash Integrity**:
   - `envs/comm_model.py`: `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431` (**0 diff**)
   - `envs/comp_model.py`: `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff` (**0 diff**)

---

## 6. Automated Test Suite Results

The dedicated Step 14 automated verification suite ([`tests/test_phase2_step14_multiseed.py`](file:///d:/cotop-implementation/tests/test_phase2_step14_multiseed.py)) was executed:

| Test ID | Test Description | Result |
|---|---|---|
| Test 01 | Factorial matrix cardinality ($n=60$) | **PASS** |
| Test 02 | Required seeds present ([42, 43, 44, 45, 46]) | **PASS** |
| Test 03 | Required geometries present (`linear_corridor`, `urban_manhattan`) | **PASS** |
| Test 04 | Required workloads present ($w \in \{20, 30, 40\}$) | **PASS** |
| Test 05 | Zero duplicate condition rows | **PASS** |
| Test 06 | Evaluation trace pairing & SHA-256 equality | **PASS** |
| Test 07 | Training vs. evaluation seed separation | **PASS** |
| Test 08 | Checkpoint existence and SHA-256 recording | **PASS** |
| Test 09 | Evaluation weight immutability | **PASS** |
| Test 10 | Evaluation trace immutability | **PASS** |
| Test 11 | Deterministic evaluation reproducibility | **PASS** |
| Test 12 | Task conservation accounting | **PASS** |
| Test 13 | Latency decomposition identity ($\le 10^{-4}\text{ s}$) | **PASS** |
| Test 14 | Energy decomposition non-negativity ($\ge 0.0\text{ J}$) | **PASS** |
| Test 15 | Physical queue bounds and completion validity | **PASS** |
| Test 16 | Zero NaN / Inf diagnostics | **PASS** |
| Test 17 | Convergence diagnostics classification completeness | **PASS** |
| Test 18 | Protected physics file hash integrity | **PASS** |

**Step 14 Test Suite**: **18/18 PASS (100%)**  
**Full Repository Regression Suite**: **88/88 PASS (100%)**

---

## 7. Protocol Boundary Confirmation

In strict compliance with Phase 2 implementation rules:
- Step 14 produced the clean, locked experimental dataset and convergence diagnostics only.
- No final paired hypothesis tests, confidence intervals, effect sizes, or superiority conclusions are claimed here.
- The raw dataset is frozen and ready for downstream statistical attribution in Step 18.
