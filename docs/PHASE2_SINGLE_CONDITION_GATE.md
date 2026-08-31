# Phase 2 — Single-Condition Comparative Gate Audit

**Document ID**: `docs/PHASE2_SINGLE_CONDITION_GATE.md`  
**Stage**: STAGE 9 — SINGLE-CONDITION COMPARATIVE GATE  
**Audited Target**: CoTOP, DDQN, Greedy, Local under canonical pilot condition  
**Canonical Condition**: Geometry: `corridor_2400m` | Workload: `w20` (200 tasks) | Pilot Seed: `0` (Eval Seed: `30000`)  
**Shared Realization**: [`data/evaluation_realizations/corridor_2400m_w20_seed0_realization.json`](file:///d:/cotop-implementation/data/evaluation_realizations/corridor_2400m_w20_seed0_realization.json)  
**Realization SHA-256 Hash**: `11f3549c1e5dbb54c448763da011f774ef914ab6eb3f4c653701aa658b53714f`  
**Test Suite**: [`tests/test_phase2_stage9_single_condition_gate.py`](file:///d:/cotop-implementation/tests/test_phase2_stage9_single_condition_gate.py) (Passed, 100%)  
**Status**: **ALL 12 GATES PASSED — FACTORIAL BENCHMARK PERMITTED**  

---

## 1. Executive Summary

Prior to launching the full multi-seed, multi-workload, multi-geometry factorial benchmark matrix, a strict **Single-Condition Comparative Gate** was executed under exactly one canonical condition (`corridor_2400m`, `w20`, seed `0`).

All 4 competing algorithms (**CoTOP**, **DDQN**, **Greedy**, and **Local**) consumed the **exact same pre-materialized evaluation realization trace** under frozen weights, zero learning, $\epsilon = 0$, and pure deterministic evaluation mode.

All 12 pre-flight verification gates passed without exception.

---

## 2. Twelve-Gate Verification Matrix

| # | Verification Gate | Target Contract & Formal Requirement | Measured Audit Evidence | Gate Status |
|:---|:---|:---|:---|:---:|
| **1** | **Realization Hash Identical** | All 4 algorithms consume the exact same SHA-256 hashed realization | SHA-256 `11f3549c1e5d...` verified identical across Local, Greedy, DDQN, and CoTOP | **PASS (Exact)** |
| **2** | **Environment Config Identical** | Physical constants ($P_V=0.01\text{ W}, P_R=100\text{ W}, N_0=0.001\text{ W}, K=1000$) locked | Fingerprint and physics checksums identical across all runs | **PASS (Exact)** |
| **3** | **Task Count Identical** | Exactly $N = 10 \times 20 = 200$ tasks evaluated for all policies | $N_{\text{total}} = 200$ tasks for Local, Greedy, DDQN, CoTOP | **PASS (Exact)** |
| **4** | **Vehicle Trajectories Identical** | Identical vehicle routes, speeds, entry times, and waypoints | 10 vehicle trajectories and waypoints verified byte-identical | **PASS (Exact)** |
| **5** | **Action-Space Semantics Identical** | Action $a=0$ (Standalone) + $a \in \{1..6\}$ (Collaborative RSU ID) | $a \in [0, 6]$ discrete actions strictly respected; Local selects $a=0$ | **PASS (Exact)** |
| **6** | **Evaluation Weights Immutable** | Model parameters must not update during evaluation ($\nabla_\theta = 0$) | SHA-256 parameter hashes before and after evaluation are bitwise identical | **PASS (Exact)** |
| **7** | **Deterministic Action Sequence** | Consecutive evaluation passes produce identical decision sequences | Repetition passes yielded 100% identical decision traces | **PASS (Exact)** |
| **8** | **Deterministic State Sequence** | State observations and task delays are bitwise reproducible | Repetition passes yielded bitwise identical delay and energy sequences | **PASS (Exact)** |
| **9** | **Task Conservation** | $N_{\text{gen}} = N_{\text{comp}} + N_{\text{fail}} + N_{\text{pend}}$ ($200 = 200 + 0 + 0$) | 200/200 tasks accounted for with zero unallocated tasks | **PASS (Exact)** |
| **10** | **Latency Decomposition** | $T_{\text{total}} = T_{\text{comm}} + T_{\text{wait}} + T_{\text{comp}}$ | Additive latency identity satisfied with residual $\le 10^{-6}\text{ s}$ | **PASS (Exact)** |
| **11** | **Energy Decomposition** | $E_{\text{total}} = E_{\text{comm}} + E_{\text{comp}}$ with $E \ge 0$ | Non-negative energy decomposition satisfied across all tasks | **PASS (Exact)** |
| **12** | **No NaN/Inf** | Zero NaN or infinite values in delays, energies, states, or rewards | 0 NaN/Inf across all 200 task evaluation transitions | **PASS (Exact)** |

---

## 3. Comparative Performance Telemetry ([`results/stage9_single_condition_gate/single_condition_gate_results.json`](file:///d:/cotop-implementation/results/stage9_single_condition_gate/single_condition_gate_results.json))

Evaluation results across 200 tasks under the canonical pilot condition:

| Algorithm | Total Tasks | Completed | Failed | Completion Ratio (%) | Mean Delay (s) | Mean Energy (J) | Comm Delay (s) | Comp Delay (s) | Wait Delay (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Local** | 200 | 200 | 0 | **100.0%** | **0.67** | **0.14** | 0.67 | 0.00 | 0.00 |
| **Greedy** | 200 | 200 | 0 | **100.0%** | **0.71** | **3.63** | 0.67 | 0.04 | 0.00 |
| **DDQN** | 200 | 200 | 0 | **100.0%** | **0.69** | **1.98** | 0.67 | 0.02 | 0.00 |
| **CoTOP** | 200 | 200 | 0 | **100.0%** | **0.70** | **2.50** | 0.67 | 0.03 | 0.00 |

### Observed Algorithmic Behavior:
- **Local Policy**: Forces standalone execution at the primary RSU ($a=0$). Results in lowest energy consumption ($0.14\text{ J}$) because inter-RSU transmission is bypassed.
- **Greedy Policy**: Actively offloads to secondary RSUs to balance queue backlogs, incurring additional R2R transmission energy ($3.63\text{ J}$).
- **DDQN Policy**: Utilizes decoupled Q-value estimates to selectively exploit collaborative R2R paths when beneficial ($1.98\text{ J}$).
- **CoTOP Policy**: Dynamically balances standalone and collaborative execution based on spatial proximity and predicted dwell times ($2.50\text{ J}$).

---

## 4. Formal Gate Decision

All 12 criteria of the Single-Condition Comparative Gate have been **verified, automated via unit tests, and satisfied without exception**.

**Final Gate Decision: APPROVED — THE FULL MULTI-SEED FACTORIAL EXPERIMENT MAY PROCEED.**
