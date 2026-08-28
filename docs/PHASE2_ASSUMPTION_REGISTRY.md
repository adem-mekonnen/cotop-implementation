# Phase 2 Scientific Assumption & Reconstruction Registry

**Document ID**: `docs/PHASE2_ASSUMPTION_REGISTRY.md`  
**Stage**: Phase 2 — Step 6 (Assumption Registry Freeze)  
**Status**: COMPLETE & LOCKED  
**Git Commit SHA**: `52f2d3c81f0b8843edd08594cccedbaca4888ea8`  

---

## 1. Registry Purpose & Scientific Governance

This registry freezes every assumption, reconstruction, engineering choice, and unresolved item in Phase 2. No implementation may introduce unrecorded assumptions. Every entry is classified, justified, and assigned an empirical sensitivity treatment.

---

## 2. Locked Assumptions & Reconstructions

### Entry 1: Urban Manhattan Grid Scenario (`grid_200m`)
- **Classification**: `PAPER-CONSISTENT RECONSTRUCTION`
- **Manuscript Basis**: Section V-A states: *"A simulation map was downloaded from OpenStreetMap... real urban scene in Hangzhou... size of the map is 200m x 200m."*
- **Problem**: The author repository omitted the SUMO network files for the 200m x 200m Hangzhou grid.
- **Phase 1 Reconstruction**: Reconstructed a 200m x 200m multi-lane intersection network in `sumo_config/hangzhou_200m.sumocfg` with 6 evenly distributed RSUs matching the paper's physical parameters.
- **Phase 2 Treatment**: Aliased canonically as `urban_manhattan`. Evaluated across all workloads and seeds.
- **Confidence**: **High**.

### Entry 2: RSU Server Compute Power Draw ($P_R^{\text{comp}} = 50.0\text{ W}$)
- **Classification**: `PAPER-CONSISTENT RECONSTRUCTION`
- **Manuscript Basis**: Eq. (11) defines computation energy $E_i = P_R^{\text{comp}} \cdot t_i^{\text{pro}}$, but Table III omits $P_R^{\text{comp}}$ while providing transmission powers ($P_V = 10\text{ dBm}, P_R = 50\text{ dBm}$).
- **Reconstruction**: Fixed $P_R^{\text{comp}} = 50.0\text{ W}$, representative of standard Xeon/EPYC edge server base draw under compute load.
- **Phase 2 Treatment**: Immutable physics invariant across all runs.
- **Confidence**: **High**.

### Entry 3: DDQN Hyperparameters & Replay Architecture
- **Classification**: `REFERENCE-SPECIFIED`
- **Manuscript Basis**: Section V-B cites Reference [34] (*Zhai et al. 2024*).
- **Specification Locked in Step 3**:
  - Decoupled Target: $y_t = r_t + \gamma (1 - d_t) Q_{\theta^-}(s_{t+1}, \arg\max_{a'} Q_\theta(s_{t+1}, a'))$.
  - Network Architecture: 3-layer MLP ($114 \to 128 \to 128 \to 128 \to 7$), ReLU activations.
  - Loss Function: Smooth L1 (Huber) loss on TD errors.
  - Target Update: Periodic hard parameter sync $\theta^- \leftarrow \theta$ every $C = 100$ steps.
  - Replay Buffer: $10{,}000$ transitions, uniform sampling, minibatch size 64.
  - Exploration Schedule: Linear $\epsilon$-decay from $1.0 \to 0.05$ over 200 episodes.
  - Learning Rate: $\alpha = 0.0002$, Discount: $\gamma = 0.99$.
- **Phase 2 Treatment**: Implemented strictly per Ref [34].
- **Confidence**: **High**.

### Entry 4: QRMP-DQN Exclusion from Primary Factorial Matrix
- **Classification**: `UNRESOLVED / AUTHORITATIVE SCIENTIFIC EXCLUSION`
- **Manuscript Basis**: Section V-B cites Reference [33] (*Guo et al. 2025*).
- **Forensic Audit Finding**: Guo et al. (2025) develops Quantile Regression Multi-Pass DQN specifically for STAR-RIS hybrid continuous-discrete action spaces (STAR-RIS transmission/reflection phase shifts and continuous power). The target manuscript operates on a purely discrete 7-action space without STAR-RIS and provides zero adaptation equations.
- **Phase 2 Treatment**: Per the mandatory QRMP-DQN Hard Gate, implementing generic QR-DQN as "QRMP-DQN" is prohibited. Formally excluded from the primary factorial matrix. Primary matrix locked to **Conditional Two-Algorithm Matrix (60 planned replications: CoTOP vs. DDQN)**.
- **Confidence**: **Authoritative Finding**.

### Entry 5: Workload Semantics (Option A: Fixed-Count Realization)
- **Classification**: `PAPER-CONSISTENT RECONSTRUCTION`
- **Manuscript Basis**: Table III lists tasks per vehicle $[20, 40]$ and arrival rate $\lambda \le 30\text{ tasks/s}$.
- **Reconstruction**: Fixed cardinality $N_{\text{target}} \in \{200, 300, 400\}$ tasks for 10 vehicles, with inter-arrival timestamps generated via $\lambda_{\text{arrival}} \le 30\text{ tasks/s}$. The materialized task arrival trace is serialized and locked prior to execution.
- **Phase 2 Treatment**: Strict realization invariant $\text{len}(\text{task\_trace}) == N_{\text{target}}$.
- **Confidence**: **High**.

### Entry 6: Metric Aggregation Forensic Evaluation
- **Classification**: `PAPER-CONSISTENT RECONSTRUCTION`
- **Manuscript Basis**: Section V-A defines Average Delay as *"the average time required to complete all tasks in the VEC system."*
- **Forensic Scope**: To evaluate whether published Table IV ($13.90\text{ s}$) represents successful-task mean, all-task penalty-adjusted mean, per-vehicle mean, or makespan, Step 16 conducts a systematic sensitivity analysis over all 7 candidates without post-hoc tuning.
- **Confidence**: **High**.

---

## 3. Invariant & Safety Declaration

All 6 entries in this registry are **frozen**. No additional assumptions may be introduced during algorithm implementation (Steps 8–9) or factorial execution (Step 15).
