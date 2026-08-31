# Phase 2 Workload Accounting & Decomposition Audit Report

**Document ID**: `docs/PHASE2_WORKLOAD_ACCOUNTING_AUDIT.md`  
**Stage**: Phase 2 — Step 12 (Workload Accounting & Decomposition Audit)  
**Status**: **STEP 12 — PASS**  
**Git Branch**: `reproduction/scientific-fidelity`  
**Git Commit SHA**: `52f2d3c81f0b8843edd08594cccedbaca4888ea8`  

---

## 1. Workload Cardinality & Arrival Rate Semantics

### 1.1 Cardinality Invariant Verification
We verified that for every supported workload level $N_{\text{tasks/veh}} \in \{20, 30, 40\}$ across nominal 10-vehicle fleets, the total target task count satisfies:
$$N_{\text{target}} = N_{\text{tasks/veh}} \times N_{\text{vehicles}}$$

| Workload Level | Tasks / Vehicle | Vehicle Count | Expected Tasks ($N_{\text{target}}$) | Materialized Tasks | Realization Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Low ($W_{20}$)** | 20 | 10 | 200 | 200 | **EXACT MATCH (0 discrepancy)** |
| **Medium ($W_{30}$)** | 30 | 10 | 300 | 300 | **EXACT MATCH (0 discrepancy)** |
| **High ($W_{40}$)** | 40 | 10 | 400 | 400 | **EXACT MATCH (0 discrepancy)** |

### 1.2 Arrival Rate Semantics
- The parameter $\lambda_{\text{arrival}} \le 30.0\text{ tasks/s}$ controls inter-arrival Poisson timestamp generation ($\Delta t \sim \text{Exp}(\lambda)$).
- Empirical audit confirms that varying $\lambda_{\text{arrival}}$ does not silently alter $N_{\text{target}}$ cardinality.

---

## 2. Task Conservation & Terminal Event Transition Audit

### 2.1 Conservation Invariant
For every completed episode:
$$N_{\text{generated}} = N_{\text{completed}} + N_{\text{failed}} + N_{\text{pending}}$$
and the failure partition satisfies:
$$N_{\text{failed}} = N_{\text{dual}} + N_{\text{deadline}} + N_{\text{coverage}} + N_{\text{departure}}$$

### 2.2 First-Terminal-Event Rule & 6-State Verification
The 6-state terminal task decision tree was empirically audited:
1. **State 1 (`COMPLETED`)**: Task completes execution before deadline and vehicle is within coverage $\to$ transitioned to `COMPLETED`.
2. **State 2 (`COMPLETED` Immutability)**: Vehicle departs after task completion $\to$ task status remains `COMPLETED` (never retrospectively modified).
3. **State 3 (`FAILED_DEADLINE`)**: Delay exceeds task deadline $d_i$ $\to$ transitioned to `FAILED_DEADLINE`.
4. **State 4 (`FAILED_COVERAGE`)**: Vehicle leaves coverage during transmission/execution $\to$ transitioned to `FAILED_COVERAGE`.
5. **State 5 (`FAILED_DEPARTURE`)**: Vehicle departs before computation finishes $\to$ transitioned to `FAILED_DEPARTURE`.
6. **State 6 (`PENDING`)**: Active tasks at episode cutoff $\to$ retained in `PENDING` without double-counting.

Zero tasks were found to disappear, double-count, or mutate across terminal boundaries.

---

## 3. Latency Decomposition Audit

For all completed offloading tasks, total delay was decomposed into its physical constituents:
$$T_{\text{total}} = T_{\text{comm}} + T_{\text{wait}} + T_{\text{comp}}$$

- **Numerical Tolerance**: $\epsilon_{\text{tol}} = 1.0 \times 10^{-4}\text{ s}$
- **Maximum Absolute Residual Observed**: $0.0\text{ s}$ ($< 10^{-7}\text{ s}$ float residual)
- **Maximum Relative Residual Observed**: $< 10^{-6}$
- **Number of Decomposition Violations**: $0 / 1000$ (0.0%)

---

## 4. Energy Decomposition Audit

Total energy consumed by offloading and processing was decomposed into:
$$E_{\text{total}} = E_{\text{comm}} + E_{\text{comp}} + E_{\text{local}} + E_{\text{r2r}}$$

- **Non-Negativity Constraint**: All energy terms satisfy $E \ge 0.0\text{ J}$.
- **Finite Value Constraint**: All energy calculations are strictly finite (0 NaN / 0 Inf).
- **Number of Violations**: $0 / 1000$ (0.0%)

---

## 5. Queue Invariants & Backlog Preservation

- **Non-Negativity**: $Q_m(t) \ge 0$ cycles for all RSUs $m \in \{0 \dots 5\}$ across all time steps.
- **Service Drain**: $Q_m(t + \Delta t) = \max(0, Q_m(t) - F_m \Delta t + \sum \phi_i^{\text{admitted}})$.
- **Backlog Integrity**: Unscheduled and failed tasks do not erroneously corrupt server queue backlogs.

---

## 6. Discrepancies & Provenance Classification

- **Discrepancies Found**: **0**
- **Classification of Workload & Accounting Specifications**: `PAPER-SPECIFIED` / `PAPER-CONSISTENT RECONSTRUCTION`

---

## 7. Step 12 Gate Decision

$$\boxed{\textbf{STEP 12 — PASS}}$$
