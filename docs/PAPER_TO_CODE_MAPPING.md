# 📖 CoTOP: Paper-to-Code Mapping & Traceability Matrix

This document serves as the **Authoritative Traceability Matrix** for the CoTOP implementation. It maps the mathematical definitions and algorithms provided in the IEEE TMC 2026 research paper directly to the source code in this repository.

---

## 1. System Model (Section III)

| Component | Paper Reference | Implementation File | Key Function/Logic | Verified Units |
| :--- | :--- | :--- | :--- | :--- |
| **V2R Data Rate** | Eq. (1) | `envs/comm_model.py` | `compute_v2r_rate()` | $\text{bps} = \text{Hz} \times \log_2(1 + \text{SINR})$ |
| **R2R Data Rate** | Eq. (2) | `envs/comm_model.py` | `compute_r2r_rate()` | $\text{bps} = \text{Hz} \times \log_2(1 + \text{SINR})$ |
| **Upload Delay** | Eq. (3) | `envs/comp_model.py` | `t_trans` in `calculate_case1_standalone` | $\text{Seconds} = \text{bits} / \text{bps}$ |
| **Processing Delay** | Eq. (4) | `envs/comp_model.py` | `t_comp` calculation using $F_m^{RSU}$ | $\text{Seconds} = \text{cycles} / \text{Hz}$ |
| **Queue Wait Delay** | Eq. (5) | `envs/comp_model.py` | `t_wait` using $N^{queue}$ | $\text{Seconds} = \text{cycles} / \text{Hz}$ |
| **Standalone Delay** | Eq. (6) | `envs/comp_model.py` | `calculate_case1_standalone()` | $T^{up} + T^{pro} + T^{wait}$ |
| **Collab. Remaining Size** | Eq. (7) | `envs/comp_model.py` | `phi_rest = task_cpu_phi - (rsu1_cpu_f * t1)` | $\text{CPU Cycles}$ |
| **R2R Trans. Delay** | Eq. (8) | `envs/comp_model.py` | `t2_inter_rsu` using $w_{R2R}$ | $\text{Seconds} = \text{bits} / \text{bps}$ |
| **Succ. Comp. Delay** | Eq. (9) | `envs/comp_model.py` | `t3_comp2` using $F_{m'}^{RSU}$ | $\text{Seconds} = \text{cycles} / \text{Hz}$ |
| **Collab. Total Delay**| Eq. (10) | `envs/comp_model.py` | `calculate_case2_collaboration()` | $T^{up} + \max(t_1, t_2 + t_3) + T^{wait}$ |
| **Comp. Energy** | Eq. (11) | `envs/comp_model.py` | $E_{comp} = \text{Time} \times P_{compute}$ | $\text{Joules} = \text{Seconds} \times \text{Watts}$ |
| **Trans. Energy** | Eq. (12) | `envs/comp_model.py` | $E_{trans} = \text{Time} \times P_{tx}$ | $\text{Joules} = \text{Seconds} \times \text{Watts}$ |

---

## 2. Mobility & Prioritization (Section IV-B, C)

| Component | Paper Reference | Implementation File | Key Function/Logic | Output & Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Coord. Expansion** | Eq. (15) | `models/mobility_gat.py` | `self.coordinate_expansion_mlp` | Expands $(x, y) \in \mathbb{R}^2 \rightarrow \mathbb{R}^{64}$ |
| **Attention Scores** | Eq. (16) | `models/mobility_gat.py` | `GATConv` (Softmax attention) | Pairwise vehicle spatial weights $\alpha_{u,v}$ |
| **Spatial Aggregation**| Eq. (17-18) | `models/mobility_gat.py` | Dual-layer GAT (4 heads) | Hierarchical multi-vehicle spatial aggregation |
| **Temporal Encoder** | Eq. (19) | `models/mobility_gat.py` | `self.encoder_gru` | Historical trajectory embedding $h_T$ |
| **Trajectory Pred.** | Eq. (20-21) | `models/mobility_gat.py` | `self.decoder_gru` autoregression | Multi-step future $(x, y)$ coordinate prediction |
| **Prediction Loss** | Eq. (22) | `train_mobility.py` | `nn.MSELoss()` | MSE loss over future trajectory points |
| **Task Priority** | Eq. (23) | `utils/task_priority.py` | `compute_task_priority()` | $P_i = \alpha e^{-1/T^{stay}} + \beta \frac{\rho_{n,i}}{d_{n,i}}$ |

---

## 3. DRL Architecture & Algorithm (Section IV-D, E)

| Component | Paper Reference | Implementation File | Key Function/Logic | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **State Space $s(t)$** | Eq. (24) | `envs/state_builder.py` | `build_state()` | $[s^v, s^{task}, s^{RSU}]$ normalized to $[0, 1]$ |
| **Reward Function** | Eq. (25) | `envs/vec_env.py` | Step reward calculation | $-(\epsilon T + (1-\epsilon)E)$ if $T \le d$ else $-Z$ |
| **Actor Loss** | Eq. (26) | `train.py` | Policy gradient with Advantage | $L_\pi(\theta) = -\log \pi(a\|s) \cdot (R - V(s))$ |
| **Cumulative Return** | Eq. (27) | `train.py` | Discounted return accumulator | $R_t = \sum \gamma^i r_{t+i}$ |
| **Critic Loss** | Eq. (28) | `train.py` | MSE loss | $L_V(\theta_v) = (R_t - V(s))^2$ |
| **CoTOP Algorithm** | Algorithm 1 | `train.py` & `envs/vec_env.py` | Parallel A3C training loop | Multiprocessing with `SharedAdam` |

---

## 4. Simulation Parameters (Section V-A, Table III)

All parameters from **Table III** are strictly centralized in `configs/paper_parameters.yaml`:

| Parameter | Paper Value | Config Key | Implementation SI Unit |
| :--- | :--- | :--- | :--- |
| Vehicle Count | 10–30 | `num_vehicles_range` | Integer count |
| Number of RSUs | 6 | `num_rsus` | Integer count |
| Vehicle Speed | 30–40 m/s | `vehicle_speed_range` | $\text{m/s}$ |
| RSU CPU Capacity | 1–4 Gcycles/s | `rsu_cpu_capacity_range`| $\text{Hz}$ ($10^9$ to $4\times 10^9$) |
| Number of Tasks | 20–40 | `num_tasks_per_vehicle_range` | Integer count |
| Task Size | 2–5 MB | `task_size_range` | Bytes ($2\times 10^6$ to $5\times 10^6$) |
| Task Tolerance Time | 20–30 s | `task_deadline_range` | Seconds |
| RSU Comm. Range | 400 m | `rsu_comm_range` | Meters |
| Vehicle TX Power | 10 dBm | `tx_power_vehicle` | Watts ($0.01\text{ W} = 10^{(10-30)/10}$) |
| RSU TX Power | 50 dBm | `tx_power_rsu` | Watts ($100.0\text{ W} = 10^{(50-30)/10}$) |
| V2R Bandwidth | 20–100 MHz | `bandwidth_v2r_range` | $\text{Hz}$ ($20\times 10^6$ to $100\times 10^6$) |
| R2R Bandwidth | 50 MHz | `bandwidth_r2r` | $\text{Hz}$ ($50\times 10^6$) |
| Noise Power | 0.001 dBm | `noise_power` | Watts ($0.001\text{ W}$) |
| Fixed Loss $K$ | 30 dB | `fixed_loss_k` | Dimensionless ratio ($10^{30/10} = 1000.0$) |
| Path Loss Exponent | 2 | `path_loss_factor` | Dimensionless exponent ($\sigma = 2.0$) |
| Priority Weights $\alpha, \beta$ | 0.3, 0.7 | `alpha`, `beta` | Dimensionless weights ($\alpha + \beta = 1.0$) |

---

## 5. Unit Conversion Standards

To eliminate dimensional ambiguity and ensure reproducibility, the codebase adheres to strict internal SI units:

1. **Time:** Seconds ($\text{s}$).
2. **Data:** Bytes ($\text{B}$) internally; explicitly multiplied by $8$ to bits ($\text{b}$) for communication equations.
3. **CPU Frequency & Capacity:** $\text{Hz}$ ($\text{cycles/s}$).
4. **CPU Demand:** Raw $\text{Cycles}$ ($\text{Mcycles} \times 10^6$).
5. **Power:** Watts ($\text{W}$). Converted from $\text{dBm}$ via $P(\text{W}) = 10^{(P_{\text{dBm}} - 30)/10}$.
6. **Energy:** Joules ($\text{J}$) ($\text{Power } (\text{W}) \times \text{Time } (\text{s})$).
7. **Distance:** Meters ($\text{m}$).

---

## 6. Scientific Decision Log

All modeling decisions, structural interpretations, and boundary conditions are documented in [docs/IMPLEMENTATION_DECISIONS.md](file:///d:/cotop-implementation/docs/IMPLEMENTATION_DECISIONS.md) and audited in [docs/REPRODUCTION_AUDIT.md](file:///d:/cotop-implementation/docs/REPRODUCTION_AUDIT.md).
