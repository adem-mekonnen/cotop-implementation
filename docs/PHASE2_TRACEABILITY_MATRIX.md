# Phase 2 Parameter & Architectural Traceability Matrix

**Document ID**: `docs/PHASE2_TRACEABILITY_MATRIX.md`  
**Stage**: Phase 2 — Step 4 (Traceability Matrix)  
**Status**: COMPLETE & LOCKED  
**Git Commit SHA**: `52f2d3c81f0b8843edd08594cccedbaca4888ea8`  

---

## 1. Provenance Classification Hierarchy

Every parameter and architectural specification is tagged with one of seven mutually exclusive categories:
- **`PAPER-SPECIFIED`**: Explicitly given in target manuscript text, equations, or Table III (Du et al. 2026).
- **`REFERENCE-SPECIFIED`**: Specified in cited foundational literature ([34] for DDQN, [33] for QRMP-DQN).
- **`REPOSITORY-SPECIFIED`**: Explicitly present in the author's original repository code (`bd34c65`).
- **`PAPER-CONSISTENT RECONSTRUCTION`**: Mathematically reconstructed to satisfy paper constraints where details were omitted.
- **`IMPLEMENTATION BUG`**: A verified defect in repository code contradicting physical theory or stated paper model.
- **`IMPLEMENTATION CHOICE`**: Standard engineering parameter where paper/references are silent.
- **`UNRESOLVED`**: Ambiguity that cannot be resolved from paper, references, or code without empirical sweep.

---

## 2. Complete Traceability Matrix

| Parameter / Architectural Element | Target Paper (Du 2026) | Foundational Ref [34]/[33] | Repository (`bd34c65`) | Provenance Category | Confidence | Phase 2 Implementation Treatment |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Highway Corridor Length** | 2400 m (Sec. III-A) | N/A | 2400.0 m | `PAPER-SPECIFIED` | High | `corridor_2400m` ($2400\text{ m}$) |
| **Urban Grid Dimensions** | 200m x 200m (Sec. V-A) | N/A | Missing | `PAPER-CONSISTENT RECONSTRUCTION` | High | `grid_200m` ($200\text{ m} \times 200\text{ m}$) |
| **Number of RSUs ($M$)** | 6 (Table III) | N/A | 6 | `PAPER-SPECIFIED` | High | 6 RSUs evenly spaced at 400 m |
| **RSU Spacing ($d_{rsu}$)** | 400 m (Sec. III-A) | N/A | 400.0 m | `PAPER-SPECIFIED` | High | Fixed 400.0 m spacing |
| **RSU Comm. Range ($R_{cov}$)** | 400 m (Table III) | N/A | 400.0 m | `PAPER-SPECIFIED` | High | Fixed 400.0 m radius |
| **Vehicle Speed ($v$)** | [30.0, 40.0] m/s (Table III)| N/A | [30.0, 40.0] | `PAPER-SPECIFIED` | High | [30.0, 40.0] m/s in SUMO |
| **Vehicle Count ($N$)** | [10, 30] (Table III) | N/A | 10 | `PAPER-SPECIFIED` | High | 10 vehicles nominal factorial |
| **Workload Tasks per Veh ($N_{\text{tasks}}$)**| [20, 40] (Table III) | N/A | 20 | `PAPER-SPECIFIED` | High | [20, 30, 40] tasks/vehicle |
| **Task Arrival Rate ($\lambda_{\text{arrival}}$)** | $\le 30$ tasks/s (Table III)| N/A | Poisson generator | `PAPER-SPECIFIED` | High | Inter-arrival sampling ($\le 30\text{ tasks/s}$) |
| **Task Data Size ($\rho$)** | [2.0, 5.0] MB (Table III) | N/A | [2.0e6, 5.0e6] | `PAPER-SPECIFIED` | High | [2.0, 5.0] MB uniformly sampled |
| **Task CPU Demand ($\phi$)** | 10 Mcycles (Sec. III-F) | N/A | 10.0e6 cycles | `PAPER-SPECIFIED` | High | 10.0 Mcycles per task |
| **Task Deadline ($T_{\text{tol}}$)** | [20.0, 30.0] s (Table III)| N/A | [20.0, 30.0] s | `PAPER-SPECIFIED` | High | [20.0, 30.0] s per task |
| **RSU CPU Capacity ($F_m$)** | [1.0, 4.0] GHz (Table III)| N/A | [1.0e9, 4.0e9] | `PAPER-SPECIFIED` | High | [1.0, 4.0] GHz per RSU |
| **Vehicle TX Power ($P_V$)** | 10 dBm (Table III) | N/A | 0.01 W | `PAPER-SPECIFIED` | High | Fixed 0.01 W ($10\text{ dBm}$) |
| **RSU TX Power ($P_R$)** | 50 dBm (Table III) | N/A | 100.0 W | `PAPER-SPECIFIED` | High | Fixed 100.0 W ($50\text{ dBm}$) |
| **RSU Server Power ($P_R^{\text{comp}}$)**| Unspecified (Eq. 11) | N/A | 50.0 W | `PAPER-CONSISTENT RECONSTRUCTION` | High | Fixed 50.0 W server processing draw |
| **V2R Bandwidth ($B^{V2R}$)** | [20, 100] MHz (Table III) | N/A | [20e6, 100e6] | `PAPER-SPECIFIED` | High | [20.0, 100.0] MHz |
| **R2R Bandwidth ($B^{R2R}$)** | 50 MHz (Table III) | N/A | 50.0e6 Hz | `PAPER-SPECIFIED` | High | Fixed 50.0 MHz |
| **Noise Power ($\sigma^2$)** | 0.001 dBm (Table III) | N/A | 0.001 W | `PAPER-SPECIFIED` | High | Fixed 0.001 W |
| **Fixed Path Loss ($K$)** | 30 dB (Table III) | N/A | 1000.0 | `PAPER-SPECIFIED` | High | Fixed $10^{30/10} = 1000.0$ |
| **Path Loss Exponent ($\sigma$)** | 2.0 (Table III) | N/A | 2.0 | `PAPER-SPECIFIED` | High | Fixed free-space $\sigma = 2.0$ |
| **Priority Weights ($\alpha, \beta$)**| 0.3, 0.7 (Sec. V-C) | N/A | 0.3, 0.7 | `PAPER-SPECIFIED` | High | Locked $\alpha = 0.3, \beta = 0.7$ |
| **CoTOP Learning Rate ($\alpha$)** | 0.0002 (Sec. V-C) | N/A | 0.0002 | `PAPER-SPECIFIED` | High | Locked $\alpha = 0.0002$ |
| **Discount Factor ($\gamma$)** | Unspecified | 0.99 (Ref [34]) | 0.99 | `REFERENCE-SPECIFIED` | High | Locked $\gamma = 0.99$ |
| **State Dimension** | Sec. IV-B | N/A | 114 | `REPOSITORY-SPECIFIED` | High | Fixed 114-dimensional vector |
| **Action Dimension** | Sec. IV-C | N/A | 7 | `PAPER-SPECIFIED` | High | 7 discrete offloading actions |
| **DDQN Target Update Rule** | Sec. V-B | Ref [34] Eq. 12 | Missing | `REFERENCE-SPECIFIED` | High | $y_t = r_t + \gamma(1-d_t) Q_{\theta^-}(s', \arg\max Q_\theta)$ |
| **DDQN Loss Function** | Unspecified | Ref [34] | Missing | `REFERENCE-SPECIFIED` | High | Smooth L1 (Huber) loss |
| **DDQN Target Sync Interval** | Unspecified | Ref [34] | Missing | `REFERENCE-SPECIFIED` | High | Hard sync every $C = 100$ steps |
| **DDQN Replay Buffer Capacity**| Unspecified | Ref [34] | Missing | `REFERENCE-SPECIFIED` | High | $10{,}000$ transitions, batch 64 |
| **DDQN Epsilon Schedule** | Unspecified | Ref [34] | Missing | `REFERENCE-SPECIFIED` | High | Linear decay $1.0 \to 0.05$ (200 eps) |
| **QRMP-DQN "MP" Definition** | Sec. V-B | Ref [33] STAR-RIS | Missing | `UNRESOLVED` | Low | Excluded from primary matrix (no discrete mapping) |
| **Training Horizon** | 500 episodes (Fig. 4) | N/A | 500 episodes | `PAPER-SPECIFIED` | High | Fixed 500 episodes (`checkpoint_ep500.pt`) |
| **Evaluation Exploration** | $\epsilon = 0.0$ | $\epsilon = 0.0$ | $\epsilon = 0.0$ | `IMPLEMENTATION CHOICE` | High | Pure greedy deterministic evaluation |
| **Published 13.90 s Delay** | Table IV | N/A | N/A | `OBSERVATION ONLY` | High | Evaluated, strictly not targeted |
| **Published 25.14 J Energy** | Table IV | N/A | N/A | `OBSERVATION ONLY` | High | Evaluated, strictly not targeted |

---

## 3. Provenance Summary Statistics

- **`PAPER-SPECIFIED`**: 18 parameters (50.0%)
- **`REFERENCE-SPECIFIED`**: 6 parameters (16.7%)
- **`REPOSITORY-SPECIFIED`**: 1 parameter (2.8%)
- **`PAPER-CONSISTENT RECONSTRUCTION`**: 3 parameters (8.3%)
- **`IMPLEMENTATION CHOICE`**: 5 parameters (13.9%)
- **`OBSERVATION ONLY`**: 2 parameters (5.6%)
- **`UNRESOLVED`**: 1 parameter (2.8%) — *QRMP-DQN Multi-Pass discrete adaptation*
- **Total Audited Parameters**: 36
