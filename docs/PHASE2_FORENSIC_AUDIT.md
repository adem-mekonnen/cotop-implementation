# Phase 2 Forensic Paper, Baseline References [33, 34], and Repository Audit

**Document ID**: `docs/PHASE2_FORENSIC_AUDIT.md`  
**Stage**: Phase 2 — Step 3 (Forensic Audit & Algorithmic Traceability Gate)  
**Status**: COMPLETE & LOCKED  
**Git Commit SHA**: `52f2d3c81f0b8843edd08594cccedbaca4888ea8`  
**Git Branch**: `reproduction/scientific-fidelity`  

---

## 1. System & Environment Fingerprint (Step 1 & Step 2 Verification)

### 1.1. Host Environment Fingerprint
- **Operating System**: Windows 10/11 Enterprise (AMD64)
- **Python Version**: `3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024)`
- **PyTorch Version**: `2.12.1+cpu`
- **PyTorch Geometric (PyG)**: `2.8.0.post1`
- **SUMO Engine Version**: `Eclipse SUMO sumo 1.27.1 (MSVC 19.29.30133.0 Release)`
- **TraCI Control Interface**: Verified and importable
- **Execution Target**: Deterministic CPU execution runtime

### 1.2. Protected-File SHA-256 Hash Locks (Step 2 Verification)
To guarantee zero contamination of the validated Phase 1 physical and environmental models, the SHA-256 hashes of all core environment files are recorded and locked:

| File Path | SHA-256 Checksum | Phase 1 Status | Phase 2 Diff |
| :--- | :--- | :--- | :--- |
| `envs/comm_model.py` | `041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431` | Validated & Locked | **0 diff** |
| `envs/comp_model.py` | `dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff` | Validated & Locked | **0 diff** |
| `envs/vec_env.py` | `16c9750a8fa4736d33c282d5ea1d8cfbfb99041124d5786048137dd71e8fdceb` | Validated & Locked | **0 diff** |
| `envs/state_builder.py` | `70ee3f9b3a4b4ef40533a2d96741a518b590558893a3a688d547e38fcb4e5a1d` | Validated & Locked | **0 diff** |
| `envs/sumo_manager.py` | `903b7f09e34e7bd3dd627e05ffdfe46c3ea3fc3d91c9f2465441731b0f4e8043` | Validated & Locked | **0 diff** |
| `envs/task_generator.py` | `34823ffb2515b561f5d44c9d8b5c115deaf4dbb2391f07ddeff51349d3e645d5` | Validated & Locked | **0 diff** |
| `envs/entities.py` | `c59069ddc7fcac9355a490745464b5711144abdd8c4f600b8cf7ce868480124f` | Validated & Locked | **0 diff** |

---

## 2. Formal Bibliographic Record Verification

The literature anchors cited in the target manuscript and experimental design have been forensically verified:

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              CANONICAL BIBLIOGRAPHIC AUDIT                                              |
+-------------------------------------------------------------------------------------------------------------------------+
| 1. Target Manuscript:                                                                                                   |
|    - Authors: Jiaxin Du, Jinfan Zhang, Guangjie Han, Mengmeng Wang, Guojiang Shen, Zhi Liu, Xiangjie Kong              |
|    - Title: "Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"              |
|    - Journal: IEEE Transactions on Mobile Computing (TMC)                                                               |
|    - Volume & Issue: Vol. 25, No. 4, pp. 5540–5556, April 2026                                                         |
|    - DOI: 10.1109/TMC.2025.3631820                                                                                      |
|    - History: Received 25 Apr 2025; Revised 28 Oct 2025; Accepted 9 Nov 2025; Published 12 Nov 2025; Version 6 Mar 2026  |
+-------------------------------------------------------------------------------------------------------------------------+
| 2. Foundational Baseline Reference [33] (QRMP-DQN):                                                                     |
|    - Authors: Liang Guo, Jie Jia, Jian Chen, Xingwei Wang                                                               |
|    - Title: "QRMP-DQN Empowered Task Offloading and Resource Allocation for the STAR-RIS Assisted MEC Systems"          |
|    - Journal: IEEE Transactions on Vehicular Technology (TVT)                                                           |
|    - Volume & Issue: Vol. 74, No. 1, pp. 1252–1266, January 2025                                                       |
|    - DOI: 10.1109/TVT.2024.3453904                                                                                      |
+-------------------------------------------------------------------------------------------------------------------------+
| 3. Foundational Baseline Reference [34] (DDQN):                                                                         |
|    - Authors: Huazhen Zhai, Xiaotian Zhou, Haixia Zhang, Dongfeng Yuan                                                  |
|    - Title: "Delay Minimization in Hybrid Edge Computing Networks: A DDQN-Based Task Offloading Approach"               |
|    - Journal: IEEE Transactions on Vehicular Technology (TVT)                                                           |
|    - Volume & Issue: Vol. 73, No. 10, pp. 15098–15108, October 2024                                                    |
|    - DOI: 10.1109/TVT.2024.3409876                                                                                      |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Deep Forensic Audit of Baseline Algorithms

### 3.1. Reference [34] (DDQN) Mathematical & Algorithmic Audit
In Reference [34] (*Zhai et al., IEEE TVT 2024*), Double Deep Q-Network is formulated for task offloading in hybrid edge computing networks.

#### 3.1.1. Core Mathematical Formulation
Standard DQN evaluates and selects actions using the same parameter vector $\theta_t$, creating systematic overestimation bias:

$$y_t^{\text{DQN}} = r_t + \gamma (1 - d_t) \max_{a'} Q(s_{t+1}, a'; \theta_t^-)$$

Reference [34] implements Van Hasselt's decoupled Double Q-learning target:

$$y_t^{\text{DDQN}} = r_t + \gamma (1 - d_t) \, Q\left(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta_t); \theta_t^-\right)$$

where:
- $a^* = \arg\max_{a'} Q(s_{t+1}, a'; \theta_t)$ is the greedy action chosen by the **online network** $\theta_t$.
- $Q(s_{t+1}, a^*; \theta_t^-)$ evaluates that chosen action using the **target network** $\theta_t^-$.
- $d_t \in \{0, 1\}$ is the episode/task terminal flag.

#### 3.1.2. Loss Function & Update Mechanics
- **Loss Function**: Smooth L1 (Huber) loss over minibatch transitions:
  $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{k=1}^B \text{Huber}\left(y_k^{\text{DDQN}} - Q(s_k, a_k; \theta)\right)$$
  where $\text{Huber}(u) = 0.5 u^2$ if $|u| \le 1.0$, and $|u| - 0.5$ otherwise.
- **Target Network Update**: Periodic hard parameter copy $\theta^- \leftarrow \theta$ every $C = 100$ update steps.
- **Replay Buffer**: Capacity $N_{\text{replay}} = 10{,}000$ transitions, sampled uniformly with minibatch size $B = 64$.
- **Exploration Schedule**: $\epsilon$-greedy decaying linearly from $\epsilon_{\text{start}} = 1.0 \to \epsilon_{\text{end}} = 0.05$ over 200 episodes.
- **Optimizer & Hyperparameters**: Adam optimizer, learning rate $\alpha = 0.0002$ (matching Table III), discount factor $\gamma = 0.99$.

---

### 3.2. Reference [33] (QRMP-DQN) Forensic Audit & Hard Gate Resolution

#### 3.2.1. What "MP" Means in Reference [33]
Forensic analysis of Guo et al. (2025) reveals:
1. **"MP" stands for Multi-Pass Deep Q-Network** (originating from *Bester et al., AAAI 2019* for parameterized action spaces).
2. **Problem Setting in [33]**: A hybrid discrete-continuous optimization problem for Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surfaces (**STAR-RIS**) assisted MEC systems.
   - The action space in [33] is parameterized: $a = (d, x_d)$, where $d \in \{1, \dots, D\}$ selects the discrete offloading decision and $x_d \in \mathbb{R}^k$ represents continuous parameters (STAR-RIS reflection/transmission phase shifts and continuous power allocation).
   - In MP-DQN, multiple forward passes evaluate each discrete action paired with its continuous parameter vector.
   - [33] combines Quantile Regression (QR-DQN) with Multi-Pass DQN (MP-DQN) and a KKT convex optimization module for continuous resource allocation.

#### 3.2.2. Discrepancy with the Target Manuscript (Du et al., IEEE TMC 2026)
1. **Action Space Mismatch**: The target paper (Du et al.) has a **strictly discrete 7-action space** (offload to $R_m$ or collaborate with $R_n \in \{1 \dots 6\}$). There are **no STAR-RIS surfaces, no continuous phase shifts, and no continuous action parameters**.
2. **Lack of Specification**: The target manuscript cites [33] in Section V-B (line 66) as a baseline, but provides **zero equations, zero adaptations, and zero hyperparameter values** explaining how a STAR-RIS Multi-Pass hybrid algorithm was mapped onto a purely discrete 6-RSU vehicular network.
3. **Absence in Author Repository**: The author's original repository (`bd34c65`) contains **no implementation, no scripts, and no references** to QRMP-DQN.

#### 3.2.3. Formal QRMP-DQN Gate Evaluation & Hard Block Decision
> [!CAUTION]
> **Authoritative Gate Decision on QRMP-DQN**:
> 1. Implementing generic QR-DQN and labelling it "QRMP-DQN" would constitute a scientifically invalid reconstruction that misrepresents Reference [33].
> 2. Because Reference [33]'s multi-pass mechanism is fundamentally tied to continuous STAR-RIS parameterization which does not exist in Du et al.'s model, and because Du et al. provides no adaptation mapping, **QRMP-DQN fails the forensic equivalence gate**.
> 3. **Outcome**: QRMP-DQN is formally classified as **`UNRESOLVED / EXCLUDED FROM PRIMARY MATRIX`**.
> 4. The primary factorial matrix is locked to the **Conditional Two-Algorithm Primary Matrix (60 planned replications: CoTOP vs. DDQN)**. This exclusion is documented as an authoritative scientific finding on baseline specification gaps in the target literature.

---

## 4. State & Action Space Forensic Mapping

### 4.1. 114-Dimensional Observation Vector $s(t)$
Constructed by `envs/state_builder.py` according to Section IV-B and Eq. (24):

```
+-----------------------------------------------------------------------------------+
|                        114-DIMENSIONAL STATE VECTOR STRUCTURE                     |
+-----------------------------------------------------------------------------------+
| 1. Ego-Vehicle State (4 features):                                                |
|    - [0] Normalized X position: pos[0] / 2400.0                                   |
|    - [1] Normalized Y position: pos[1] / 2400.0                                   |
|    - [2] Normalized vehicle speed: speed / 40.0                                   |
|    - [3] Normalized dwell time: T_stay / 100.0                                    |
+-----------------------------------------------------------------------------------+
| 2. Local Task Queue (4 features * 20 max tasks = 80 features):                    |
|    For each task slot i in [0..19]:                                               |
|    - [4 + 4i]   Normalized task data size: rho / 5.0e6                            |
|    - [5 + 4i]   Normalized CPU demand: phi / 4.0e9                                |
|    - [6 + 4i]   Normalized tolerable delay: max_delay / 30.0                      |
|    - [7 + 4i]   Task priority score: priority in [0, 1]                           |
+-----------------------------------------------------------------------------------+
| 3. RSU Network Status (5 features * 6 RSUs = 30 features):                        |
|    For each RSU m in [0..5]:                                                      |
|    - [84 + 5m]  Normalized RSU X location: loc[0] / 2400.0                       |
|    - [85 + 5m]  Normalized RSU Y location: loc[1] / 2400.0                       |
|    - [86 + 5m]  Normalized CPU compute capacity: f_m / 4.0e9                     |
|    - [87 + 5m]  Normalized queued CPU backlog: Q_m / 1.0e9                        |
|    - [88 + 5m]  Normalized transmission power: P_R / 100.0                        |
+-----------------------------------------------------------------------------------+
| Total Dimension: 4 + 80 + 30 = 114 features (All bounded in [0.0, 1.0])           |
+-----------------------------------------------------------------------------------+
```

### 4.2. Action Space Semantics ($|A| = 7$)
The action space is a discrete set of 7 offloading decisions:
- $a = 0$: Standalone execution at the current receiving RSU $R_m$ (no R2R forwarding).
- $a \in \{1, \dots, 6\}$: Collaborative offloading to RSU $R_n$ ($n = a$).

### 4.3. Action Feasibility Masking ($\mathcal{A}_{\text{valid}}$)
Generated exclusively by the environment:
- Action $a = 0$ is always valid if vehicle is within communication range of at least one RSU.
- Action $a \in \{1, \dots, 6\}$ is valid if target RSU $R_n$ is within maximum collaborative relay distance and has available queue capacity.
- Invariant: $\mathcal{A}_{\text{valid}}^{\text{CoTOP}}(s) = \mathcal{A}_{\text{valid}}^{\text{DDQN}}(s)$ for any identical state.

---

## 5. Candidate Metric Aggregation Estimators Taxonomy (Step 16 Preparation)

To forensically explain why Phase 1 empirical latency ($\approx 4.4\text{ s}$) was lower than published Table IV ($13.90\text{ s}$), Step 16 will evaluate 7 candidate estimators:

| Candidate ID | Aggregation Estimator | Mathematical Formulation | Metric Role | Provenance Classification |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | Mean Successful-Task Latency | $\bar{T}_{\text{success}} = \frac{1}{N_c} \sum_{i \in \text{completed}} T_i$ | Primary Candidate | `PAPER-CONSISTENT RECONSTRUCTION` |
| **A2** | Mean All-Task Latency ($T_{\text{tol}}$ Penalty) | $\bar{T}_{\text{all}} = \frac{1}{N_g} \left(\sum_{i \in \text{completed}} T_i + N_f \cdot T_{\text{tol}}\right)$ | Primary Candidate | `PAPER-CONSISTENT RECONSTRUCTION` |
| **A3** | Per-Vehicle Mean Latency | $\bar{T}_{\text{veh}} = \frac{1}{V} \sum_{v=1}^V \bar{T}_v$ | Primary Candidate | `PAPER-CONSISTENT RECONSTRUCTION` |
| **A4** | Data-Size-Weighted Latency | $\bar{T}_{\text{weighted}} = \frac{\sum_i d_i T_i}{\sum_i d_i}$ | Diagnostic Metric | `IMPLEMENTATION CHOICE` |
| **A5** | Cumulative Task Latency | $T_{\text{cum}} = \sum_{i \in \text{completed}} T_i$ | Diagnostic Metric | `IMPLEMENTATION CHOICE` |
| **A6** | Makespan | $T_{\text{makespan}} = \max_i t_{i, \text{completion}} - t_{\text{start}}$ | Diagnostic Metric | `IMPLEMENTATION CHOICE` |
| **A7** | Energy per Completed Task | $\bar{E}_{\text{task}} = \frac{E_{\text{total}}}{N_{\text{completed}}}$ | Primary Candidate | `PAPER-CONSISTENT RECONSTRUCTION` |

---

## 6. Summary of Forensic Audit Outcomes

1. **Environment Integrity**: Verified Python 3.11.9, PyTorch 2.12.1, SUMO 1.27.1, and locked 7 core environment SHA-256 hashes with **0 diff**.
2. **Bibliographic Grounding**: Target manuscript (Du et al., IEEE TMC 2026, DOI: `10.1109/TMC.2025.3631820`), Ref [33] (Guo et al., IEEE TVT 2025), and Ref [34] (Zhai et al., IEEE TVT 2024) confirmed.
3. **DDQN Formulation**: Locked to Ref [34] Double Q-learning ($y_t = r_t + \gamma(1-d_t) Q_{\theta^-}(s_{t+1}, \arg\max_{a'} Q_\theta(s_{t+1}, a'))$), 3-layer MLP backbone, Huber loss, $C=100$ target sync, $10{,}000$ replay capacity, $\epsilon$-decay $1.0 \to 0.05$.
4. **QRMP-DQN Hard Gate Resolution**: Excluded from primary matrix due to unresolvable domain mismatch between [33]'s STAR-RIS continuous parameterization and Du et al.'s discrete offloading environment.
5. **Primary Factorial Matrix**: Formally locked to the **Conditional Two-Algorithm Primary Matrix (60 planned replications: CoTOP vs. DDQN across 2 geometries $\times$ 3 workloads $\times$ 5 seeds)**.
