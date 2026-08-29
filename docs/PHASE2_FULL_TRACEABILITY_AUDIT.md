# Phase 2 — Full Forensic Paper-to-Code Traceability Audit (Stage 2)

**Document ID**: `PHASE2_FULL_TRACEABILITY_AUDIT`  
**Target Publication**: Du et al., *"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"*, IEEE Transactions on Mobile Computing, vol. 25, no. 4, pp. 5540–5556, April 2026. DOI: `10.1109/TMC.2025.3631820`.  
**Audited Branch**: `reproduction/scientific-fidelity` at commit `2cfc822` / `1eaec8c`  
**Comparison Baselines**: Target Paper, Ref [33] (Guo et al., IEEE TVT 2025), Ref [34] (Zhai et al., IEEE TVT 2024), `main` branch (`bd34c65`), Phase 1 baseline (`52f2d3c`).

---

## 1. Classification Taxonomy

Each experimental component is classified into one of seven mutually exclusive scientific categories:

1. **`IMPLEMENTED-FAITHFULLY`**: Component matches the explicit mathematical formulations, parameter values, and execution constraints specified in the paper with zero unresolved ambiguities.
2. **`IMPLEMENTED-WITH-RECONSTRUCTION`**: Paper leaves minor implementation ambiguities, but the mechanism is reconstructed using authoritative reference literature (e.g., Refs [32], [34]) or physical first-principles, validated against unit invariant tests.
3. **`PARTIALLY-IMPLEMENTED`**: Core mechanism exists in code, but specific parameter sweeps, sub-modules, or experimental branches remain incomplete.
4. **`IMPLEMENTED-BUT-UNVERIFIED`**: Code exists in repository, but lacks systematic multi-seed verification tests against analytical toy ground truths.
5. **`MISSING`**: Component is described in the paper but has no corresponding implementation in the active reproduction branch.
6. **`UNRESOLVED`**: Paper specification contains fundamental contradictions or underspecifications that cannot be resolved without proprietary author data.
7. **`EXCLUDED`**: Component was subjected to a formal scientific audit gate and explicitly barred from execution (e.g., QRMP-DQN Ref [33] continuous-action mismatch).

---

## 2. Component-by-Component Traceability Matrix (A–AC)

### A. Environment
- **Classification**: `IMPLEMENTED-WITH-RECONSTRUCTION`
- **Paper Reference**: Section V-A, p. 5550; Table III.
- **Specification**: SUMO-based simulation with dual topology: (1) Reconstructed 200 m $\times$ 200 m urban Manhattan grid (`hangzhou_200m.sumocfg`), and (2) historical 2400 m linear corridor (`hangzhou.sumocfg`).
- **Code Location**: `envs/vec_env.py`, `sumo_config/`
- **Audit Findings**: `VECEnv` cleanly supports both geometries via `scenario_geometry="grid_200m"` and `"corridor_2400m"`. Port collisions in multi-worker parallelism are resolved via dynamic port offsets.

### B. Mobility Model
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section III-B, Equations (1)–(18), ApolloScape dataset [32].
- **Specification**: Multi-node GAT + GRU spatio-temporal vehicle trajectory prediction model estimating RSU coverage dwell time $T_{stay, v}$.
- **Code Location**: `models/gat_model.py`, `envs/vec_env.py` (`_estimate_all_dwell_times`)
- **Audit Findings**: Phase 1 repaired GAT Layer 2 Eq (18) dimensions ($F' \to 2$ coordinate delta outputs) and permutation equivariance. Dwell times are dynamically updated at each SUMO step.

### C. Vehicle Model
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section III-A, Table III.
- **Specification**: $N_V \in [10, 30]$ vehicles, transmission power $P_v = 0.1\text{ W} = 20\text{ dBm}$, local CPU frequency $f_v = 1.0\text{ GHz}$, effective switched capacitance $\kappa_v = 10^{-28}$.
- **Code Location**: `envs/entities.py`, `configs/paper_parameters.yaml`
- **Audit Findings**: Parameters match Table III exactly.

### D. RSU Topology
- **Classification**: `IMPLEMENTED-WITH-RECONSTRUCTION`
- **Paper Reference**: Section V-A, p. 5550; Section III-A.
- **Specification**: $M = 6$ RSUs evenly distributed. Coverage range $R = 100\text{ m}$ (or $300\text{ m}$ in historical corridor), computing capacity $F_m = 10.0\text{ GHz}$, computing energy factor $\kappa_m = 10^{-28}$, transmission power $P_m = 1.0\text{ W} = 30\text{ dBm}$.
- **Code Location**: `utils/scenario_geometry.py`, `envs/vec_env.py`
- **Audit Findings**: Grid topology places RSUs at $(50,50), (100,50), (150,50), (50,150), (100,150), (150,150)$ in 200 m $\times$ 200 m space.

### E. Communication Model
- **Classification**: `IMPLEMENTED-FAITHFULLY` (Phase-1 Protected)
- **Paper Reference**: Section III-C, Equations (19)–(22), Table III.
- **Specification**: V2I and I2I wireless rates computed via Shannon capacity with log-distance path loss, Rayleigh fading, and Gaussian noise $\sigma^2 = -100\text{ dBm}$.
- **Code Location**: `envs/comm_model.py` (SHA-256: `041e4106...`)
- **Audit Findings**: Strictly hash-locked and immutable (0 diff). Analytical toy values verified by `test_comm_model.py`.

### F. Computation Model
- **Classification**: `IMPLEMENTED-FAITHFULLY` (Phase-1 Protected)
- **Paper Reference**: Section III-D, Section III-E, Equations (23)–(28), Table III.
- **Specification**: Parallel subtask latency and energy accounting across 3 execution paths: local compute, direct RSU compute, and collaborative RSU-to-RSU compute.
- **Code Location**: `envs/comp_model.py` (SHA-256: `dd9f58df...`)
- **Audit Findings**: Strictly hash-locked and immutable (0 diff). Analytical multi-branch tests verified by `test_comp_model.py`.

### G. Queue Model
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section III-D, p. 5545, Equation (24).
- **Specification**: FIFO execution queue per RSU with service rate $\mu_m = F_m / \bar{\phi} \ge 100\text{ tasks/s}$, queuing delay $T_{wait, m}(t) = Q_m(t) / F_m$.
- **Code Location**: `envs/vec_env.py` (`_advance_queues`), `tests/test_queue_model.py`
- **Audit Findings**: Cross-algorithm shared queue depletion invariant verified by Test 28 and `test_12_5_queue_invariants`.

### H. Task Generation
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section III-A, Table III.
- **Specification**: Tasks per vehicle $w \in [20, 40]$, data size $\rho_i \in [2.0, 5.0]\text{ Mbits}$, CPU requirement $\phi_i \in [1.0, 10.0]\text{ Mcycles}$, tolerable delay $d_i \in [20.0, 30.0]\text{ s}$.
- **Code Location**: `envs/task_generator.py`, `envs/vec_env.py`
- **Audit Findings**: Verified by Step 12 audit. Task conservation $\Delta = 0$ holds across all 60 runs.

### I. Task Priority
- **Classification**: `IMPLEMENTED-WITH-RECONSTRUCTION`
- **Paper Reference**: Section IV-A, Equations (29)–(30).
- **Specification**: Composite priority weight $W_i = \alpha \cdot \text{norm}(1/d_i) + \beta \cdot \text{norm}(\phi_i / \rho_i)$ with default $\alpha = 0.3, \beta = 0.7$.
- **Code Location**: `utils/task_priority.py`, `envs/vec_env.py`
- **Audit Findings**: Phase 1 repaired denominator zero-safety and dual min-max scaling across active task queues.

### J. State Representation
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section IV-B, p. 5547.
- **Specification**: $s(t) = [s_{ego}, \{s_{task, i}\}_{i=1}^{N_T}, \{s_{rsu, m}\}_{m=1}^M]$. Observation dimension $D = 4 + 4 N_T + 5 M$.
- **Code Location**: `utils/state_builder.py`, `envs/vec_env.py`
- **Audit Findings**: Dimension dynamically adapts ($D=114$ for $w=20$, $D=154$ for $w=30$, $D=194$ for $w=40$). Verified by Test 20 and Step 14 Test 01.

### K. Action Space
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section IV-C, p. 5548.
- **Specification**: Discrete action space $\mathcal{A} = \{0, 1, 2, 3, 4, 5, 6\}$ (0: local compute, $1..6$: offload to RSU $m$).
- **Code Location**: `envs/vec_env.py`, `tests/test_phase2_state_action_contract.py`
- **Audit Findings**: Action space dimension $\mathcal{A} \in \{0..6\}$ strictly invariant across all algorithms and baselines.

### L. Action Feasibility & Masking
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section IV-C, Section III-D.
- **Specification**: RSU actions $a \in \{1..6\}$ are feasible iff vehicle is within communication range of at least one RSU and total delay does not exceed vehicle dwell time $T_{stay}$.
- **Code Location**: `envs/vec_env.py` (`get_action_mask`), `tests/test_action_mask_bounds.py`
- **Audit Findings**: Both CoTOP and DDQN consume identical environment-controlled feasibility masks.

### M. Reward Function
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section IV-D, Equations (31)–(32).
- **Specification**: $r(t) = - (\mu_1 T_{total} + \mu_2 E_{total} + \mu_3 \mathbb{I}(T_{total} > d_i) \cdot P_{penalty})$ with $\mu_1 = 0.5, \mu_2 = 0.5, P = 10.0$.
- **Code Location**: `envs/vec_env.py` (`_calculate_reward`), `tests/test_reward.py`
- **Audit Findings**: Verified by unit tests. Identical reward formulation applied across CoTOP and DDQN.

### N. CoTOP Architecture
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section IV-E, Fig. 3, Section IV-F.
- **Specification**: Spatial GAT trajectory feature extraction + `ActorCritic` network ($D \to 128 \to 128 \to 128 \to 7$ policy logits + $1$ value head).
- **Code Location**: `models/a3c_agent.py`, `models/gat_model.py`
- **Audit Findings**: Architecture matches paper specification.

### O. CoTOP Training
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section IV-F, Algorithm 1, Table III.
- **Specification**: Asynchronous Advantage Actor-Critic (A3C) policy gradient updates with Adam ($\text{lr} = 0.0002$), discount $\gamma = 0.99$, entropy regularization $0.01$.
- **Code Location**: `train.py`, `scripts/run_phase2_multiseed.py`
- **Audit Findings**: Fully executed across 30 multi-seed runs in Step 14.

### P. DDQN Baseline
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section V-B, p. 5550; Reference [34] (Zhai et al., IEEE TVT 2024).
- **Specification**: Double DQN with decoupled target $y_t = r_t + \gamma (1-d_t) Q_{\theta^-}(s', \arg\max Q_\theta)$, Smooth L1 loss, 10,000 FIFO replay buffer, batch size 64, 100-step target sync, $\epsilon$-decay $1.0 \to 0.05$ over 200 episodes.
- **Code Location**: `models/baselines/ddqn_agent.py`
- **Audit Findings**: Passed all 8 Step 8 unit tests, all 12 Step 9/10 cross-algorithm tests, and all 30 multi-seed training runs.

### Q. Greedy Baseline
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section V-B, p. 5550.
- **Specification**: Rule-based heuristic selecting the least loaded RSU / lowest immediate queue delay at step $t$.
- **Code Location**: `models/baselines/greedy_policy.py`, `tests/test_baselines.py`
- **Audit Findings**: Unit verified.

### R. Local Baseline
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section V-B, p. 5550.
- **Specification**: Rule-based heuristic restricting all tasks to local on-vehicle execution ($a = 0$).
- **Code Location**: `models/baselines/local_policy.py`, `tests/test_baselines.py`
- **Audit Findings**: Unit verified.

### S. QRMP-DQN Baseline
- **Classification**: `EXCLUDED`
- **Paper Reference**: Section V-B, p. 5550; Reference [33] (Guo et al., IEEE TVT 2025).
- **Forensic Audit Finding**: Ref [33] uses a continuous Multi-Pass action parameterization designed for continuous offloading fractions $[0, 1]^K$. The target paper provides zero adaptation for the discrete 7-action space.
- **Scientific Ruling**: Formally excluded from the primary factorial matrix under Step 3 Forensic Gate. Not replaced with generic QR-DQN in accordance with Rule 4.

### T. Workload Generation
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section V-A, Section V-D, Table IV, Table V.
- **Specification**: 3 workload levels ($w \in \{20, 30, 40\}$ tasks/vehicle) evaluated across 5 seeds.
- **Code Location**: `scripts/run_phase2_multiseed.py`, `envs/task_generator.py`
- **Audit Findings**: Pre-materialized into 30 locked exogenous JSON traces.

### U. Evaluation Procedure
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section V-A, p. 5550.
- **Specification**: Fixed Episode-500 checkpoint evaluation under strictly deterministic inference ($\epsilon = 0.0$, `torch.no_grad()`, `model.eval()`).
- **Code Location**: `scripts/run_phase2_multiseed.py`
- **Audit Findings**: Verified weight immutability ($H(\theta_{\text{before}}) \equiv H(\theta_{\text{after}})$) and realization immutability across all 60 runs.

### V. Aggregation
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Section V-A, Section V-D.
- **Specification**: Mean and sample standard deviation across independent paired seeds ($n=5$).
- **Code Location**: `results/phase2_algorithmic_fidelity/PHASE2_EXPERIMENT_INDEX.csv`
- **Audit Findings**: Full per-seed and aggregated statistics recorded.

### W. Random Seeds
- **Classification**: `IMPLEMENTED-FAITHFULLY`
- **Paper Reference**: Experimental protocol best practices.
- **Specification**: Master seed vector $S = [42, 43, 44, 45, 46]$ with formal mathematical separation: $S_{env} = 10000+s, S_{train} = 20000+s, S_{eval} = 30000+s$.
- **Code Location**: `scripts/run_phase2_multiseed.py`, `utils/seed.py`
- **Audit Findings**: Verified by Step 14 Test 02 and Test 07.

### X. Statistical Analysis
- **Classification**: `PARTIALLY-IMPLEMENTED`
- **Paper Reference**: Section V-D, Tables IV–V.
- **Specification**: Paired small-$n$ non-parametric inference (Wilcoxon signed-rank / exact paired permutation test, Cohen's $d_z$ effect size, bootstrap CIs).
- **Code Location**: Awaiting Step 18 execution. Dataset is ready in `PHASE2_EXPERIMENT_INDEX.csv`.

### Y. Ablation Experiments
- **Classification**: `PARTIALLY-IMPLEMENTED`
- **Paper Reference**: Section V-D, Table VI (CoTOP vs w/o MD vs w/o TP vs w/o CO).
- **Specification**: Module ablation of Mobility Detection (MD), Task Priority (TP), and Collaborative Offloading (CO).
- **Audit Findings**: Code infrastructure supports ablations (`use_mobility_model=False`, $\alpha=0, \beta=0$, no R2R offloading). Execution planned for downstream ablation stage.

### Z. Sensitivity Experiments
- **Classification**: `PARTIALLY-IMPLEMENTED`
- **Paper Reference**: Section V-C (Figs 4, 5: learning rate $\text{lr} \in [10^{-4}, 10^{-2}]$, priority weights $\alpha \in [0.1, 0.9]$), Section V-D (Fig 7: transmission rate, Fig 8: computing capacity, Fig 9: vehicle counts $N_V \in [10, 30]$).
- **Audit Findings**: Sweep parameters supported in configuration files.

### AA. Real-World / Hangzhou Experiment
- **Classification**: `IMPLEMENTED-WITH-RECONSTRUCTION`
- **Paper Reference**: Section V-E, Figs 10, 11 ($> 100$ vehicles, 8 intersections).
- **Specification**: Dense urban scenario scaled to $> 100$ vehicles across 8 RSU intersections.
- **Code Location**: `sumo_config/hangzhou.sumocfg`

### AB. Figures
- **Classification**: `PARTIALLY-IMPLEMENTED`
- **Paper Reference**: Figures 4, 5, 6, 7, 8, 9, 10, 11.
- **Audit Findings**: Raw trajectory and metric data generated in `results/phase2_algorithmic_fidelity/`. Plotting scripts to be executed in Step 15/16.

### AC. Tables
- **Classification**: `PARTIALLY-IMPLEMENTED`
- **Paper Reference**: Table III (Parameters), Table IV (Delay comparison), Table V (Completion ratio comparison), Table VI (Ablations).
- **Audit Findings**: Experimental data populated in `PHASE2_EXPERIMENT_INDEX.csv`. Final formatting into publication-grade LaTeX/Markdown tables scheduled for Step 17.

---

## 3. Summary Classification Distribution

| Classification Category | Count | Components |
|---|---|---|
| **`IMPLEMENTED-FAITHFULLY`** | 18 | B, C, E, F, G, H, J, K, L, M, N, O, P, Q, R, T, U, W |
| **`IMPLEMENTED-WITH-RECONSTRUCTION`** | 4 | A, D, I, AA |
| **`PARTIALLY-IMPLEMENTED`** | 6 | X, Y, Z, AB, AC, V |
| **`IMPLEMENTED-BUT-UNVERIFIED`** | 0 | None |
| **`MISSING`** | 0 | None |
| **`UNRESOLVED`** | 0 | None |
| **`EXCLUDED`** | 1 | S (QRMP-DQN Ref [33]) |
| **Total Components Audited** | **29** | **A through AC** |
