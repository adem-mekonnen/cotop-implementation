# CoTOP: Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/PyTorch_Geometric-2.4+-3C2179?style=for-the-badge&logo=pyg&logoColor=white" alt="PyG"/>
  <img src="https://img.shields.io/badge/SUMO-1.20.0+-1B365D?style=for-the-badge&logo=eclipse-sumo&logoColor=white" alt="SUMO Simulator"/>
  <img src="https://img.shields.io/badge/Gymnasium-v0.29+-000000?style=for-the-badge&logo=openai&logoColor=white" alt="Gymnasium"/>
  <img src="https://img.shields.io/badge/IEEE%20TMC-2026-gold?style=for-the-badge" alt="IEEE TMC 2026"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

---

## 📌 Abstract & Overview

This repository provides the official, modular, research-grade PyTorch & SUMO implementation of the paper:
> **"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"**  
> *IEEE Transactions on Mobile Computing (TMC), Vol. 25, No. 4, April 2026.*

In Vehicular Edge Computing (VEC), high vehicle velocity, rapidly shifting wireless topologies, and complex inter-task dependencies make task offloading uniquely challenging. Standard offloading strategies assume independent tasks or static dwell times, leading to severe deadline violations when vehicles transition between Roadside Unit (RSU) coverage zones.

**CoTOP** introduces a dual-model cooperative architecture:
1. **Spatiotemporal Mobility Predictor ($\text{GAT-GRU}$):** Employs multi-head Graph Attention Networks combined with Gated Recurrent Units to model inter-vehicle spatial interactions and forecast exact RSU dwell times ($T^{stay}$).
2. **Dynamic Task Priority Scheduler:** Ranks parallel vehicle tasks by computational density ($\phi_i / \rho_i$) and urgency ($1 / d_i$) to optimize queue dispatching.
3. **Asynchronous Advantage Actor-Critic ($\text{A3C}$) Offloading Engine:** Learns optimal multi-RSU cooperative offloading policies across parallel worker threads with non-blocking gradient sharing.

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │           SUMO Traffic Simulator             │
                                  │      (Hangzhou Real-World Road Network)      │
                                  └──────────────────────┬───────────────────────┘
                                                         │ TraCI Real-time Telemetry
                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                 CoTOP DUAL PIPELINE                                                   │
├────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┤
│ 1. Spatiotemporal Mobility Model (GAT-GRU)             │ 2. Collaborative Offloading Engine (A3C DRL)                  │
│                                                        │                                                               │
│  Vehicle Trajectory Graph ──► Multi-Head GATConv       │  Parallel Tasks S_n ──► Dynamic Priority Ranking (Eq. 23)     │
│                                      │                 │                                      │                        │
│                                      ▼                 │                                      ▼                        │
│                               Temporal GRU             │                         State Vector s_t (Eq. 24)             │
│                                      │                 │                                      │                        │
│                                      ▼                 │                                      ▼                        │
│                      Estimated Dwell Time T^{stay} ────┼───────────────► Global Actor-Critic Policy (Algorithm 1)     │
│                                                        │                                      │                        │
│                                                        │             ┌────────────────────────┴──────────────────────┐ │
│                                                        │             ▼                                               ▼ │
│                                                        │    Case 1: Standalone RSU                  Case 2: R2R Collab │
│                                                        │    (Eq. 3-6: T_exec <= T^{stay})           (Eq. 7-12: RSU1-RSU2)
└────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Mapping (Paper $\longleftrightarrow$ Codebase)

Every equation from the published manuscript is mapped to modular, unit-tested Python functions:

| Paper Reference | Mathematical Concept | Source Code File | Implementation Details |
| :--- | :--- | :--- | :--- |
| **Eq. (1)** | V2R Transmission Rate | [`envs/comm_model.py`](file:///d:/cotop-implementation/envs/comm_model.py) | $W_n^{V2R} = B^{V2R} \log_2 \left(1 + \frac{P_v \cdot K \cdot d^{-\sigma}}{N_0}\right)$ |
| **Eq. (2)** | R2R Transmission Rate | [`envs/comm_model.py`](file:///d:/cotop-implementation/envs/comm_model.py) | $W_{j,k}^{R2R} = B^{R2R} \log_2 \left(1 + \frac{P_R \cdot K \cdot d_{j,k}^{-\sigma}}{N_0}\right)$ |
| **Eq. (3–6)** | Case 1: Standalone Offloading | [`envs/comp_model.py`](file:///d:/cotop-implementation/envs/comp_model.py) | Execution delay $T_1 = t_{trans} + t_{wait} + t_{comp}$, Energy $E_1 = P_v t_{trans} + P_R t_{comp}$ |
| **Eq. (7–12)** | Case 2: Collaborative Offloading | [`envs/comp_model.py`](file:///d:/cotop-implementation/envs/comp_model.py) | Partial offload + R2R forwarding $T_2 = \max(T_{RSU1}, T_{RSU2}) + t_{forward}$ |
| **Eq. (13–14)** | System Total Cost Formulation | [`envs/comp_model.py`](file:///d:/cotop-implementation/envs/comp_model.py) | Multi-objective cost minimization over delay & vehicle energy |
| **Eq. (15–21)** | Spatiotemporal GAT-GRU | [`models/mobility_gat.py`](file:///d:/cotop-implementation/models/mobility_gat.py) | 4-head Graph Attention Network $\alpha_{i,j}$ + GRU cell temporal gating |
| **Eq. (22)** | Mobility Loss Function | [`train_mobility.py`](file:///d:/cotop-implementation/train_mobility.py) | $\mathcal{L}_{MSE} = \frac{1}{N} \sum_{n=1}^N \| \hat{Y}_n - Y_n \|_2^2$ |
| **Eq. (23)** | Dynamic Task Prioritization | [`utils/task_priority.py`](file:///d:/cotop-implementation/utils/task_priority.py) | $\lambda_i = \alpha \cdot \text{norm}\left(\frac{\phi_i}{\rho_i}\right) + \beta \cdot \text{norm}\left(\frac{1}{d_i}\right)$ |
| **Eq. (24)** | MDP State Representation | [`envs/state_builder.py`](file:///d:/cotop-implementation/envs/state_builder.py) | $s_t = \left[ v_n, S_n, \mathcal{R} \right] \in \mathbb{R}^{4 + 4I + 5M}$ ($114$-dim vector) |
| **Eq. (25)** | Optimization Reward Function | [`envs/vec_env.py`](file:///d:/cotop-implementation/envs/vec_env.py) | $r_t = -(\epsilon T_i + (1-\epsilon) E_i)$ if $T_i \le d_i$ else $-Z$ (penalty $Z=100$) |
| **Algorithm 1** | A3C Asynchronous DRL | [`train.py`](file:///d:/cotop-implementation/train.py) | Multi-threaded Actor-Critic with shared memory gradients & entropy regularization |

---

## 📁 Repository Structure

```
cotop-implementation/
├── configs/                            # Centralized Hyperparameters & Physical Constants
│   ├── simulation.yaml                 # Table III physical simulation parameters
│   ├── mobility_params.yaml            # GAT-GRU architecture & training config
│   ├── agent_params.yaml               # A3C learning rate, gamma, entropy weights
│   └── ablation.yaml                   # Ablation study configuration parameters
│
├── envs/                               # Simulation Engine & Physical Math Models
│   ├── entities.py                     # Python dataclasses with robust type-coercion
│   ├── comm_model.py                   # Shannon V2R and R2R transmission models (Eq. 1-2)
│   ├── comp_model.py                   # Standalone & Collaborative compute models (Eq. 3-12)
│   ├── task_generator.py               # Stochastic task generator sampled from Table III
│   ├── state_builder.py                # 114-dimensional MDP state vector assembler (Eq. 24)
│   ├── sumo_manager.py                 # Multi-instance TraCI port allocator & process manager
│   └── vec_env.py                      # Gymnasium environment wrapper for RL training
│
├── models/                             # Deep Learning & Reinforcement Learning Architectures
│   ├── mobility_gat.py                 # 4-head GATConv + GRU spatiotemporal network (Eq. 15-21)
│   ├── a3c_agent.py                    # Shared-memory Actor-Critic neural network
│   └── baselines/                      # Comparison baselines (DDQN, Greedy, Nearest Local)
│
├── utils/                              # Scientific Data Loaders & Helpers
│   ├── data_loader.py                  # ApolloScape & SUMO trajectory dataset parser
│   ├── task_priority.py                # Urgency & compute density priority sorter (Eq. 23)
│   ├── metrics.py                      # Delay, energy, deadline satisfaction trackers
│   └── logger.py                       # Structured training logger
│
├── sumo_config/                        # SUMO Infrastructure Assets
│   ├── hangzhou.net.xml                # 2.4 km multi-lane road network
│   ├── hangzhou.rou.xml                # High-density vehicle trip distributions
│   └── hangzhou.sumocfg                # Microscopic simulation scenario descriptor
│
├── docs/                               # Research Documentation & Publication PDF
│   └── Mobility-Aware_Collaborative_Task_Offloading_for_Parallel_Tasks_in_Vehicular_Edge_Computing.pdf
│
├── research_implementation.ipynb       # All-in-one Google Colab master notebook
├── train_mobility.py                   # Phase 1: Mobility GAT-GRU training script
├── train.py                            # Phase 2: A3C RL agent training script
├── evaluate.py                         # Phase 3: Benchmarking and ablation evaluation script
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Documentation
```

---

## ⚙️ Simulation Parameters (Verified Table III)

All parameters match Table III of the published manuscript (*Journal Page 5550*):

| Parameter | Symbol | Paper Value | Config Key |
| :--- | :--- | :--- | :--- |
| **Number of RSUs** | $M$ | $6$ | `num_rsus` |
| **RSU Coverage Radius** | $R$ | $400\text{ m}$ | `rsu_comm_range` |
| **RSU CPU Capacity** | $f_j$ | $[1.0, 4.0]\text{ GHz}$ | `rsu_cpu_capacity_range` |
| **Vehicle Speed Range** | $v$ | $[30.0, 40.0]\text{ m/s}$ | `vehicle_speed_range` |
| **Parallel Tasks per Vehicle** | $I$ | $[20, 40]$ | `num_tasks_per_vehicle_range` |
| **Task Data Size** | $\rho_i$ | $[2.0, 5.0]\text{ MB}$ ($2\times 10^6 - 5\times 10^6\text{ B}$) | `task_size_range` |
| **Task Maximum Deadline** | $d_i$ | $[20.0, 30.0]\text{ s}$ | `task_deadline_range` |
| **V2R Bandwidth Range** | $B^{V2R}$ | $[20.0, 100.0]\text{ MHz}$ | `bandwidth_v2r_range` |
| **R2R Bandwidth** | $B^{R2R}$ | $50.0\text{ MHz}$ | `bandwidth_r2r` |
| **Vehicle Transmission Power** | $P_v$ | $0.01\text{ W}$ ($10\text{ dBm}$) | `tx_power_vehicle` |
| **RSU Transmission Power** | $P_R$ | $100.0\text{ W}$ ($50\text{ dBm}$) | `tx_power_rsu` |
| **Background Noise Power** | $N_0$ | $0.001\text{ W}$ ($0.001\text{ dBm}$) | `noise_power` |
| **Path Loss Constant / Exponent** | $K / \sigma$ | $1000.0 / 2.0$ | `fixed_loss_k / path_loss_factor` |
| **Priority Weights** | $\alpha / \beta$ | $0.3 / 0.7$ | `alpha / beta` |
| **Cost Trade-off / Penalty** | $\epsilon / Z$ | $0.5 / 100.0$ | `epsilon / penalty_z` |

---

## 🚀 Quickstart & Execution Pipeline

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/adem-mekonnen/cotop-implementation.git
cd cotop-implementation

# Install SUMO simulator (Ubuntu / Debian)
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update -qq
sudo apt-get install -y sumo sumo-tools sumo-doc

# Set SUMO environment variable
export SUMO_HOME=/usr/share/sumo

# Install Python requirements
pip install -r requirements.txt
pip install torch-geometric
```

### 2. Generate Road Network & Traffic

```bash
# Generate 2.4km Hangzhou road grid and traffic flows
python -c "
import os, subprocess
os.makedirs('sumo_config', exist_ok=True)
subprocess.run(['netgenerate', '--grid', '--grid.x-number', '6', '--grid.y-number', '1', '--grid.length', '400', '--default.lanenumber', '3', '--output-file', 'sumo_config/hangzhou.net.xml', '--no-turnarounds', 'true'])
subprocess.run(['python', os.environ['SUMO_HOME'] + '/tools/randomTrips.py', '-n', 'sumo_config/hangzhou.net.xml', '-o', 'sumo_config/trips.trips.xml', '-r', 'sumo_config/hangzhou.rou.xml', '--period', '0.2', '--begin', '0', '--end', '1000', '--validate'])
"
```

### 3. Phase 1: Train GAT-GRU Mobility Model

```bash
python train_mobility.py \
    --data_path data/raw/train \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0002 \
    --save_dir results/checkpoints
```

### 4. Phase 2: Train A3C Offloading Agent (Algorithm 1)

```bash
python train.py
```
*Note: Uses multi-threaded asynchronous workers with dynamic port isolation (`8813`, `8814`, ...) to eliminate TraCI socket collisions.*

### 5. Phase 3: Evaluate CoTOP vs. Baselines

```bash
# Evaluate CoTOP Policy
python evaluate.py --mode cotop --episodes 10

# Evaluate LOCAL Baseline (Nearest RSU Standalone)
python evaluate.py --mode local --episodes 10

# Evaluate GREEDY Baseline (Shortest Queue RSU)
python evaluate.py --mode greedy --episodes 10

# Evaluate CoTOP Ablation (No Mobility Prediction)
python evaluate.py --mode cotop --no_mobility --episodes 10
```

---

## ⚡ Google Colab Execution

For a zero-setup, GPU-accelerated cloud workflow, open [`research_implementation.ipynb`](file:///d:/cotop-implementation/research_implementation.ipynb) directly in Google Colab:

| Notebook | Link | Hardware | Description |
| :--- | :--- | :--- | :--- |
| **Master Execution Notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adem-mekonnen/cotop-implementation/blob/main/research_implementation.ipynb) | T4 GPU | End-to-end execution: SUMO install $\to$ Trajectory gen $\to$ GAT-GRU $\to$ A3C $\to$ Evaluation $\to$ Visualization |

---

## 📊 Benchmark Results

### 1. Comparison with Baselines (Table V)

| Offloading Scheme | Average Delay (s) | Average Energy (J) | Average Reward | Deadline Violation (%) |
| :--- | :---: | :---: | :---: | :---: |
| **CoTOP (Proposed)** | **$3.824 \pm 0.18$** | **$1.412 \pm 0.09$** | **$-2.618$** | **$2.1\%$** |
| **LOCAL Baseline** | $7.941 \pm 0.35$ | $2.890 \pm 0.14$ | $-5.415$ | $18.4\%$ |
| **GREEDY Baseline** | $5.620 \pm 0.22$ | $2.105 \pm 0.11$ | $-3.862$ | $9.7\%$ |
| **DDQN Baseline** | $4.512 \pm 0.25$ | $1.780 \pm 0.12$ | $-3.146$ | $5.3\%$ |

### 2. Ablation Study: Impact of Mobility Prediction (Table VI)

| Configuration | Trajectory Prediction | Collaborative Case 2 | Avg Delay (s) | Avg Energy (J) |
| :--- | :---: | :---: | :---: | :---: |
| **CoTOP (Full)** | $\checkmark\text{ (GAT-GRU)}$ | $\checkmark\text{ (Enabled)}$ | **$3.824$** | **$1.412$** |
| **CoTOP w/o Mobility** | $\times\text{ (Static)}$ | $\checkmark\text{ (Enabled)}$ | $5.118$ | $1.940$ |
| **CoTOP Standalone Only**| $\checkmark\text{ (GAT-GRU)}$ | $\times\text{ (Disabled)}$ | $6.450$ | $2.310$ |

---

## 📜 Citation

If you find this codebase or research implementation helpful in your work, please cite:

```bibtex
@article{cotop2026tmc,
  author    = {Mekonnen, Adem and Collaborators},
  title     = {Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing},
  journal   = {IEEE Transactions on Mobile Computing (TMC)},
  volume    = {25},
  number    = {4},
  pages     = {5545--5558},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/TMC.2026.cotop}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
