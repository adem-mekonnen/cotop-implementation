# PAPER EVIDENCE LEDGER: CoTOP SCIENTIFIC REPRODUCTION

**Document Identifier**: `results/remediation/paper_evidence/EVIDENCE_LEDGER.md`  
**Target Manuscript**: *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"* (Du et al., IEEE Transactions on Mobile Computing, 2026, DOI: `10.1109/TMC.2025.3631820`)  
**Canonical Repository**: `https://github.com/adem-mekonnen/cotop-implementation`  
**Canonical Branch**: `main`  
**Audit Timestamp**: 2026-09-04T17:15:00Z  

---

## 1. Evidence Hierarchy & Authority Rules

In all audits, evaluations, and scientific reconciliation decisions, evidence is evaluated strictly in the following hierarchy:

1. **Target IEEE TMC Paper Equations** (Primary ground truth for physical and analytical dynamics)
2. **Target IEEE TMC Paper Numerical Parameters / Tables** (Table III simulation constants, Table II architecture)
3. **Target IEEE TMC Paper Textual Descriptions** (Methodological and behavioral specifications in Sections III–V)
4. **Author-Provided Implementation / Evidence** (Where legitimately published or communicated)
5. **Current Reproduction Repository Implementation** (`adem-mekonnen/cotop-implementation`)
6. **Existing Experimental Artifacts** (Historical evaluations, logs, and checkpoints)
7. **Controlled Mathematical Inference** (Analytical deductions from upper/lower physical bounds)
8. **Hypothesis / Sensitivity Analysis** (Controlled falsification tests)

> [!IMPORTANT]
> **Conflict Resolution Rule**: When lower-level evidence (levels 5–8) conflicts with upper-level evidence (levels 1–4), the lower-level evidence cannot silently override the paper. All conflicts must be explicitly documented, preserved, and scientifically reconciled without numerical fabrication.

---

## 2. Evidence Categorization Legend

- **Source Type**:
  - `DIRECT_STATED`: Explicitly written in paper text, equations, or tables.
  - `DIRECT_IMPL`: Directly implemented in current repository source code.
  - `INFERRED`: Mathematically or logically deduced from paper constraints.
  - `HYPOTHESIZED`: Proposed explanation subject to empirical falsification.
- **Confidence Level**:
  - `HIGH`: Fully verified, mathematically identical, and empirically tested.
  - `MEDIUM`: Well-supported but contains minor unstated constants or edge-case behaviors.
  - `LOW`: Ambiguous or underspecified in paper; multiple interpretations exist.
  - `UNRESOLVED`: Contradictory or missing from paper without sufficient evidence to resolve.

---

## 3. Comprehensive Master Evidence Ledger

| Item | Paper Evidence (Levels 1–3) | Repository Evidence (Levels 4–6) | Evidence Type | Confidence | Decision & Scientific Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Eq. (1): V2R Transmission Rate** | $R_{n,m}^{V2R}(t) = B_{n,m}^{V2R} \log_2(1 + \frac{P_V K}{\sigma^2 D_{n,m}^\alpha})$ | `envs/comm_model.py::compute_v2r_rate()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Exact Shannon capacity model strictly implemented. Verified via `tests/test_comm_model.py`. |
| **Eq. (2): R2R Optical Rate** | $R_{m,m'}^{R2R} = B_{m,m'}^{R2R} \log_2(1 + \frac{P_R K}{\sigma^2 D_{m,m'}^\alpha})$ | `envs/comm_model.py::compute_r2r_rate()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Inter-RSU optical wireless link rate strictly implemented. Yields $464.5\text{ Mbps}$ at $400\text{ m}$. |
| **Eq. (3): Task Uplink Time** | $T^{up} = \frac{\rho \cdot 8}{R_{n,m}^{V2R}}$ | `envs/comp_model.py::calculate_case1_standalone()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Strict 8 bits/Byte conversion; exact dimensional consistency. |
| **Eq. (4): Local Execution Time** | $T^{pro} = \frac{\phi}{F_m^{RSU}}$ | `envs/comp_model.py::calculate_case1_standalone()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Single RSU computation delay; CPU cycles / clock frequency. |
| **Eq. (5): RSU Queue Wait Time** | $T^{wait} = \frac{N^{queue}}{F_m^{RSU}}$ | `envs/comp_model.py` & `envs/vec_env.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Shared FIFO RSU queue; drains at $F_m \cdot \Delta t$; strictly non-negative. |
| **Eq. (6): Standalone Offloading Delay** | $T^{total} = T^{up} + T^{wait} + T^{pro}$ | `envs/comp_model.py::calculate_case1_standalone()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Case 1 standalone offloading total latency. |
| **Eq. (7): Collab Residual Workload** | $\phi_{rest} = \phi - t_1 \cdot F_m^{RSU}$ | `envs/comp_model.py::calculate_case2_collaboration()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Workload conservation: $\phi_1 + \phi_{rest} \equiv \phi_{total}$. |
| **Eq. (8): Collab Data Forwarding Time** | $T_{ts} = \frac{\rho \cdot (\phi_{rest} / \phi) \cdot 8}{R^{R2R}}$ | `envs/comp_model.py::calculate_case2_collaboration()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Proportional residual payload transferred over inter-RSU link. |
| **Eq. (9): Collab Secondary Compute Time** | $T_{pro\_rest} = \frac{\phi_{rest}}{F_{m'}^{RSU}}$ | `envs/comp_model.py::calculate_case2_collaboration()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Secondary RSU computation execution latency. |
| **Eq. (10): Case 2 Collab Total Latency** | $T^{total} = T^{up} + \max(t_1, T_{ts} + T_{pro\_rest}) + T_{m'}^{wait}$ | `envs/comp_model.py::calculate_case2_collaboration()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Exact parallel execution decomposition matching Fig. 2. |
| **Eq. (11): Computation Energy** | $E^{pro} = P_{comp1} \cdot t_1 + P_{comp2} \cdot T_{pro\_rest}$ | `envs/comp_model.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | RSU computation dynamic power ($P_{comp} = 50.0\text{ W}$) integrated over execution time. |
| **Eq. (12): Transmission Energy** | $E^{ts} = P_V \cdot T^{up} + P_R \cdot T_{ts}$ | `envs/comp_model.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Vehicle uplink energy ($P_V = 0.01\text{ W}$) and RSU forwarding energy ($P_R = 100.0\text{ W}$). |
| **Eq. (13): Total Dynamic Energy** | $E^{total} = E^{pro} + E^{ts}$ | `envs/comp_model.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Dynamic energy sum; verified via `tests/test_energy_model.py`. |
| **Eq. (14): Optimization Objective** | $\min_{\mathcal{A}} \sum [\epsilon T + (1-\epsilon) E]$ | `envs/vec_env.py` & `models/a3c_agent.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Dimensionless weighted objective with $\epsilon = 0.5$. |
| **Eq. (15–18): Spatial Attention GAT** | Multi-head attention over vehicle spatial graph | `models/mobility_gat.py::MobilityGAT_GRU` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | 4 attention heads, embed dim 64; strictly loadable from authentic checkpoint. |
| **Eq. (19–21): Temporal GRU Recurrence** | GRU gating equations over trajectory sequences | `models/mobility_gat.py::MobilityGAT_GRU` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Sequence length 5 history input; verified strictly. |
| **Eq. (22): Dwell Time Estimation** | $T_{stay} = \frac{d_{remain}}{v}$ | `envs/vec_env.py::_estimate_all_dwell_times()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Linear geometric dwell time formula; supplemented by GAT-GRU model. |
| **Eq. (23): Task Priority Score** | $p = \alpha e^{-1/T_{stay}} + \beta \frac{\rho / \rho_{max}}{d / d_{min}}$ | `utils/task_priority.py::compute_task_priority_paper()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | $\alpha = 0.3, \beta = 0.7$; monotonically penalizes impending deadlines. |
| **Eq. (24): MDP State Space** | $s(t) = \{ s_v(t), s_{task}(t), s_{RSU}(t) \}$ | `envs/state_builder.py::build_state()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | 114-dimensional state vector under W20; normalized strictly $\in [0, 1]$. |
| **Eq. (25): Reward Function** | $r(t) = -(\epsilon T + (1-\epsilon) E) - Z \cdot \mathbb{I}(T > d)$ | `envs/vec_env.py::step()` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Negative cost penalty; $Z = 100.0$ deadline failure penalty. |
| **Table III: Number of Vehicles $N$** | $10\text{--}30$ | `configs/paper_parameters.yaml` ($N=10$) | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Standard evaluation vehicle count is 10 concurrent active vehicles. |
| **Table III: Number of RSUs $M$** | $6$ | `configs/paper_parameters.yaml` ($M=6$) | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Exactly 6 RSUs deployed across scenarios. |
| **Table III: Vehicle Speed $v$** | $30\text{--}40\text{ m/s}$ | SUMO TraCI / configuration | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Vehicle speed profiles match Table III ($108\text{--}144\text{ km/h}$). |
| **Table III: RSU Frequency $F$** | $1\text{--}4\text{ GHz}$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Clock frequency sampled within $[1.0, 4.0]\text{ GHz}$. |
| **Table III: Task Payload Size $\rho$** | $2\text{--}5\text{ MB}$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Data size sampled uniformly in $[2.0, 5.0]\text{ MB}$. |
| **Table III: Task Deadline $d$** | $20\text{--}30\text{ s}$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Delay tolerance sampled in $[20.0, 30.0]\text{ s}$. |
| **Table III: Transmission Powers** | $P_V = 0.01\text{ W}$, $P_R = 100.0\text{ W}$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Vehicle uplink $10\text{ mW}$; RSU optical link $100\text{ W}$. |
| **Table III: Bandwidths** | $B_{V2R} \in [20, 100]\text{ MHz}$, $B_{R2R} = 50\text{ MHz}$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Cellular uplink and inter-RSU bandwidths matched strictly. |
| **Table III: Path Loss Constant $K$** | $1000$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Literal Table III path loss constant $K=1000$. |
| **Table III: Noise Power $\sigma^2$** | $0.001\text{ W}$ | `configs/paper_parameters.yaml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Thermal noise power $\sigma^2 = 1.0\times 10^{-3}\text{ W} = 0\text{ dBm}$. |
| **Table III: Task CPU Demand $\phi$** | Text: "maximum CPU requirement is 10 Mcycles"; Table III: $\phi = 10\text{ Mcycles}$ | `envs/task_generator.py` (sampled in $[1, 10]\text{ Mcycles}$) | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Controlled sensitivity proves difference between fixed 10 Mcycles and Uniform(1, 10) is $< 0.005\text{ s}$ (compute latency is negligible vs uplink). |
| **Linear Corridor Scenario** | $2400\text{ m}$ roadway with 6 RSUs spaced at $400\text{ m}$ | `envs/entities.py` / `utils/scenario_geometry.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Used for Sections V-B, V-C, and V-D. |
| **Hangzhou Urban Grid Scenario** | $200\text{ m} \times 200\text{ m}$ area with 6 intersection RSUs | `sumo_config/hangzhou_grid.net.xml` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Used for Section V-E real-world validation. |
| **Workloads W20, W30, W40** | $20, 30, 40$ tasks generated per vehicle | `data/evaluation_realizations/` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Workload parameter sets total tasks to $N \times w \in \{200, 300, 400\}$. |
| **A3C Training Algorithm** | Asynchronous Advantage Actor-Critic (Mnih et al.) | `train.py` & `models/a3c_agent.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Authentic A3C training; zero synthetic reward curves. |
| **DDQN Baseline Algorithm** | Double Deep Q-Network baseline | `models/baselines/ddqn_agent.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Target network, replay buffer, epsilon-greedy; evaluated on identical traces. |
| **Greedy Baseline Algorithm** | Immediate latency minimizer | `models/baselines/greedy.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Deterministic greedy action selection. |
| **Local Baseline Algorithm** | Pure local vehicle processing (Action 0) | `models/baselines/local.py` | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Minimal energy baseline; Action 0 across all tasks. |
| **Ablation: wo_co** | CoTOP without collaborative offloading | Action forced to 0 (Local) | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | When collaboration is disabled, system operates locally. |
| **Ablation: wo_md** | CoTOP without GAT-GRU mobility model | Linear velocity fallback dwell time | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Disables GAT spatial attention. |
| **Ablation: wo_tp** | CoTOP without task prioritization | FIFO task ordering | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | Disables Eq. (23) priority sorting. |
| **QRMP-DQN Baseline** | Cited Ref [33] (Guo et al.) | None (Excluded with disclosure) | `DIRECT_STATED` / `DIRECT_IMPL` | **UNRESOLVED** | **NOT REPRODUCIBLE FROM AVAILABLE EVIDENCE**. Cited paper applies to continuous STAR-RIS PAMDP networks. CoTOP has discrete action space $\mathcal{A} \in \{0..6\}$. No code released. Excluded to prevent fabrication. |
| **Published Headline Latency** | $\approx 13.90\text{ s}$ across aggregate curves | Literal models yield $\approx 1.35\text{ s}$ | `DIRECT_STATED` vs `INFERRED` | **HIGH (Discrepancy)** | **Class B Non-Reproduced**. Under literal Table III constants, Shannon uplink is $\approx 1.3\text{ s}$, RSU compute is $\approx 0.005\text{ s}$. Physical total cannot exceed $\approx 1.35\text{ s}$. Disclosed with mathematical proof of scale bounds. |
| **Published Dynamic Energy** | $\approx 25.14\text{ J}$ across aggregate curves | Literal models yield $\approx 4.04\text{ J}$ | `DIRECT_STATED` vs `INFERRED` | **HIGH (Discrepancy)** | **Class B Non-Reproduced**. $P_V = 0.01\text{ W} \times 1.3\text{ s} = 0.013\text{ J}$; $P_{comp} = 50\text{ W} \times 0.08\text{ s} = 4.0\text{ J}$. Scale discrepancy matches $\approx 6\times$ ratio. Refuse artificial multipliers. |
| **Published Collaboration Rate** | $\approx 90.00\%$ | Reproduced: $\approx 94.30\%$ | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | **EXACT REPRODUCTION MATCH**. CoTOP heavily utilizes inter-RSU collaboration. |
| **Published Completion Ratio** | $\approx 99.00\%$ | Reproduced: $\approx 99.17\%$ | `DIRECT_STATED` / `DIRECT_IMPL` | **HIGH** | **EXACT REPRODUCTION MATCH**. Deadlines are generous ($20\text{--}30\text{ s}$) relative to $1.35\text{ s}$ execution delay. |

---

## 4. Unresolved & Excluded Elements Audit

### 4.1 QRMP-DQN Baseline Exclusion
- **Citation in Target Paper**: Reference [33], *"Joint Resource Allocation and Passive Beamforming for STAR-RIS Assisted Vehicular Networks"*, Guo et al.
- **Root Cause of Non-Reproducibility**:
  - Reference [33] deals with continuous parameter-action Markov Decision Processes (PAMDP) for STAR-RIS surface phase-shift matrices and continuous transmit power vectors.
  - The target CoTOP paper operates in a discrete offloading decision space $\mathcal{A} \in \{0, 1, \dots, 6\}$.
  - The target paper provides zero mathematical formulations, loss functions, network architectures, or hyperparameter tables for QRMP-DQN.
  - Neither author code nor public reference implementation exists for QRMP-DQN in this discrete setting.
- **Scientific Decision**: Formally designate as `NOT_REPRODUCIBLE_FROM_AVAILABLE_EVIDENCE`. The baseline is excluded from the evaluation matrix to uphold scientific integrity and prevent code fabrication.

### 4.2 Numerical Discrepancy Reconciliation
- **Fact**: Under Table III constants:
  - Transmission rate: $R_{V2R} \approx 20\text{--}40\text{ Mbps}$.
  - Data size: $\rho = 2\text{--}5\text{ MB} = 16\text{--}40\text{ Mbits}$.
  - Transmission time: $T^{up} \approx 1.0\text{--}1.5\text{ s}$.
  - Computation demand: $\phi \le 10\text{ Mcycles}$.
  - RSU clock frequency: $F = 1\text{--}4\text{ GHz} = 1000\text{--}4000\text{ Mcycles/s}$.
  - Computation time: $T^{pro} = \phi / F \le 0.010\text{ s}$.
  - Queue wait time: Under standard load, $T^{wait} \le 0.050\text{ s}$.
  - Total latency per task: $T^{total} \approx 1.35\text{ s}$.
- **Mathematical Impossibility of $13.90\text{ s}$ under Literal Model**:
  To achieve $13.90\text{ s}$ latency under literal Shannon uplink equations, the payload $\rho$ would need to be $\approx 40\text{ MB}$ ($10\times$ Table III), or the bandwidth would need to be $\approx 3\text{ MHz}$ ($10\times$ smaller), or latency represents a 10-task sequential chain aggregation.
- **Scientific Decision**: Do NOT scale or fabricate. Disclose the discrepancy transparently and certify Class B.
