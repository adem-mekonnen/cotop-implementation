# Phase 2 — Workload Accounting Final Audit & Invariant Specification

**Document ID**: `docs/PHASE2_WORKLOAD_ACCOUNTING_FINAL.md`  
**Stage**: STAGE 8 — WORKLOAD ACCOUNTING  
**Audited Subsystems**: [`envs/task_generator.py`](file:///d:/cotop-implementation/envs/task_generator.py), [`envs/vec_env.py`](file:///d:/cotop-implementation/envs/vec_env.py), [`experiments/realizations/`](file:///d:/cotop-implementation/experiments/realizations/)  
**Test Suite**: [`tests/test_phase2_workload_accounting.py`](file:///d:/cotop-implementation/tests/test_phase2_workload_accounting.py) (7/7 Passed, 100%)  
**Status**: **PASSED & CONSERVATION LOCKED**  

---

## 1. Mathematical Distinction: $N_{\text{target}}$ vs. $\lambda_{\text{arrival}}$

A fundamental pitfall in edge computing RL benchmarks is confusing **workload cardinality** ($N_{\text{target}}$) with **temporal arrival intensity** ($\lambda_{\text{arrival}}$). This audit formally distinguishes and locks both dimensions:

```
+---------------------------------------------------------------------------------------------------+
|  DIMENSION          | DEFINITION & UNITS        | ROLE IN SIMULATION                               |
+---------------------+---------------------------+--------------------------------------------------+
|  N_target           | Discrete task count       | Integral bound on total tasks assigned to the    |
|  (Workload Card.)   | (dimensionless integer)   | vehicle system: N_target = V * w                 |
+---------------------+---------------------------+--------------------------------------------------+
|  lambda_arrival     | Temporal arrival rate     | Continuous Poisson/Exponential arrival intensity |
|  (Arrival Int.)     | (tasks / second)          | governing inter-arrival spacing: delta_t ~ Exp(l)|
+---------------------+---------------------------+--------------------------------------------------+
```

### Orthogonality Proof:
- **Cardinality Invariance**: Sampling inter-arrival times $\Delta t_i \sim \text{Exp}(\lambda_{\text{arrival}})$ assigns timestamps $t_i = t_0 + \sum_{k=1}^i \Delta t_k$ to tasks without altering the discrete task count $\sum_{i=1}^{N} 1 \equiv N_{\text{target}}$.
- **No Early Truncation**: Episodes terminate only when all $N_{\text{target}}$ tasks reach a terminal state or the simulation hard deadline horizon expires, ensuring complete task conservation.

---

## 2. Workload Cardinality Matrix ($N_{\text{target}}$)

The discrete task count is verified across all canonical workloads ($w \in \{20, 30, 40\}$ tasks/vehicle) and vehicle densities ($V \in \{10, 15, 20, 25, 30\}$):

| Vehicle Count ($V$) | Workload ($w$) | Tasks per Vehicle | Target Cardinality ($N_{\text{target}} = V \times w$) | Generated Tasks ($N_{\text{gen}}$) | Verification Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **10** (Canonical) | **$w20$** | 20 | **200** | 200 | **PASS (Exact)** |
| **10** (Canonical) | **$w30$** | 30 | **300** | 300 | **PASS (Exact)** |
| **10** (Canonical) | **$w40$** | 40 | **400** | 400 | **PASS (Exact)** |
| **15** | $w20$ | 20 | **300** | 300 | **PASS (Exact)** |
| **15** | $w30$ | 30 | **450** | 450 | **PASS (Exact)** |
| **15** | $w40$ | 40 | **600** | 600 | **PASS (Exact)** |
| **20** | $w20$ | 20 | **400** | 400 | **PASS (Exact)** |
| **20** | $w30$ | 30 | **600** | 600 | **PASS (Exact)** |
| **20** | $w40$ | 40 | **800** | 800 | **PASS (Exact)** |
| **25** | $w20$ | 20 | **500** | 500 | **PASS (Exact)** |
| **25** | $w30$ | 30 | **750** | 750 | **PASS (Exact)** |
| **25** | $w40$ | 40 | **1000** | 1000 | **PASS (Exact)** |
| **30** | $w20$ | 20 | **600** | 600 | **PASS (Exact)** |
| **30** | $w30$ | 30 | **900** | 900 | **PASS (Exact)** |
| **30** | $w40$ | 40 | **1200** | 1200 | **PASS (Exact)** |

---

## 3. Universal Task Conservation Law

At any point in time $t \ge 0$, every task generated in the simulation obeys strict state conservation:

$$N_{\text{generated}} = N_{\text{completed}} + N_{\text{failed}} + N_{\text{pending}}$$

Where:
- $N_{\text{pending}}$: Tasks currently resident in vehicle queues or awaiting scheduling.
- $N_{\text{completed}}$: Tasks successfully transmitted and computed within deadline ($T_{\text{total}} \le d$) under valid coverage.
- $N_{\text{failed}}$: Tasks that encountered an operational termination failure.

### Mutually Exclusive and Collectively Exhaustive (MECE) Failure Decomposition:

$$N_{\text{failed}} = N_{\text{dual}} + N_{\text{deadline}} + N_{\text{coverage}} + N_{\text{departure}}$$

1. **$N_{\text{deadline}}$**: Task delay exceeded maximum tolerable latency ($T_{\text{total}} > d$) while vehicle remained in valid coverage.
2. **$N_{\text{coverage}}$**: Vehicle departed RSU communication coverage radius before transmission or completion while remaining within deadline ($T_{\text{total}} \le d$).
3. **$N_{\text{dual}}$**: Both deadline was exceeded ($T_{\text{total}} > d$) and vehicle departed coverage before completion.
4. **$N_{\text{departure}}$**: Vehicle exited the road network / simulation boundary with uncompleted tasks pending.

---

## 4. Invariant Rules for Task Accounting

### Rule 1: Zero Double-Counting
Every task possesses a unique globally incremented integer ID (`task_id`). Once a task transitions to a terminal state (`COMPLETED`, `FAILED_DEADLINE`, `FAILED_COVERAGE`, `FAILED_DUAL`, `FAILED_DEPARTURE`), it is removed from active queues and appended to immutable result lists. The intersection $\mathcal{S}_{\text{completed}} \cap \mathcal{S}_{\text{failed}} = \emptyset$ is strictly empty.

### Rule 2: First Terminal Event Governance
Terminal classification is strictly determined by the **first event in continuous simulation time**:
- If coverage failure occurs at $t_{\text{cov}} = 4.2\text{ s}$ and deadline would expire at $t_{\text{dead}} = 22.0\text{ s}$, the task is classified as **`FAILED_COVERAGE`** at $t=4.2\text{ s}$.
- If deadline expires at $t_{\text{dead}} = 12.0\text{ s}$ and coverage is lost at $t_{\text{cov}} = 18.5\text{ s}$, the task is classified as **`FAILED_DEADLINE`** at $t=12.0\text{ s}$.
- If both thresholds are breached simultaneously, the task is classified as **`FAILED_DUAL`**.

---

## 5. Exact Latency Decomposition Identity

Total task turnaround latency $T_{\text{total}}$ decomposes additively into communication, queuing wait, and computation components without unexplained residuals:

$$T_{\text{total}} = T_{\text{comm}} + T_{\text{wait}} + T_{\text{comp}}$$

### 1. Case 1 (Standalone Execution):
$$T_{\text{comm}} = t_{\text{trans}} = \frac{8 \rho}{w_{n,m}^{V2R}}$$
$$T_{\text{wait}} = t_{\text{wait}} = \frac{Q_m(t)}{f_m}$$
$$T_{\text{comp}} = t_{\text{comp}} = \frac{\phi}{f_m}$$
$$\|T_{\text{total}} - (T_{\text{comm}} + T_{\text{wait}} + T_{\text{comp}})\| \equiv 0.0$$

### 2. Case 2 (Collaborative Execution):
$$T_{\text{comm}} = t_{\text{up}} + t_2 = \frac{8 \rho}{w_{n,m}^{V2R}} + \frac{8 \rho_{\text{rest}}}{w_{m,m'}^{R2R}}$$
$$T_{\text{wait}} = t_{\text{wait}, 2} = \frac{Q_{m'}(t)}{f_{m'}}$$
$$T_{\text{comp}} = \max(t_1, t_2 + t_3) - t_2$$
$$\|T_{\text{total}} - (T_{\text{comm}} + T_{\text{wait}} + T_{\text{comp}})\| \equiv 0.0$$

---

## 6. Exact Energy Decomposition Identity

Total system energy consumption $E_{\text{total}}$ decomposes additively across communication and computation domains:

$$E_{\text{total}} = E_{\text{comm}} + E_{\text{comp}} = \underbrace{(E_{\text{v2r}} + E_{\text{r2r}})}_{E_{\text{comm}}} + \underbrace{(E_{\text{rsu1}} + E_{\text{rsu2}})}_{E_{\text{comp}}}$$

Where:
- $E_{\text{v2r}} = P_V \times t_{\text{up}}$ (Vehicle V2R uplink transmit energy)
- $E_{\text{r2r}} = P_{R1} \times t_2$ (RSU1 to RSU2 relay transmit energy)
- $E_{\text{rsu1}} = P_{\text{comp}} \times t_1$ (RSU1 primary computation energy)
- $E_{\text{rsu2}} = P_{\text{comp}} \times t_3$ (RSU2 secondary computation energy)
- **Non-Negativity Constraint**: $E_{\text{v2r}} \ge 0, E_{\text{r2r}} \ge 0, E_{\text{rsu1}} \ge 0, E_{\text{rsu2}} \ge 0 \implies E_{\text{total}} \ge 0$.

---

## 7. Exhaustive Test Suite Summary ([`tests/test_phase2_workload_accounting.py`](file:///d:/cotop-implementation/tests/test_phase2_workload_accounting.py))

```text
tests/test_phase2_workload_accounting.py::test_01_workload_cardinality_vs_arrival_intensity_orthogonality PASSED
tests/test_phase2_workload_accounting.py::test_02_task_conservation_identity PASSED
tests/test_phase2_workload_accounting.py::test_03_failure_decomposition_and_no_double_counting PASSED
tests/test_phase2_workload_accounting.py::test_04_first_terminal_event_temporal_governance PASSED
tests/test_phase2_workload_accounting.py::test_05_latency_decomposition_exact_identity PASSED
tests/test_phase2_workload_accounting.py::test_06_energy_decomposition_exact_identity PASSED
tests/test_phase2_workload_accounting.py::test_07_queue_nonnegativity_and_depletion PASSED
```

**Full Repository Status**: `pytest tests/` passed **122/122 tests (100%)**.

**Final Gate Decision: APPROVED & CONSERVATION LOCKED.**
