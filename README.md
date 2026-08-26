# Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing (CoTOP)

A research-grade, mathematically faithful reproduction of the IEEE Transactions on Mobile Computing (TMC 2026) paper:

> **"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"**  
> *Jiaxin Du, Jinfan Zhang, Guangjie Han, Mengmeng Wang, Guojiang Shen, Zhi Liu, and Xiangjie Kong*  
> IEEE TMC, Vol. 25, No. 4, April 2026. DOI: 10.1109/TMC.2025.3631820

---

## 1. System Overview

Vehicular Edge Computing (VEC) empowers connected vehicles to offload computation-heavy, latency-critical workloads to roadside units (RSUs). However, high-speed vehicle mobility causes frequent disconnections and task interruptions.

**CoTOP** addresses this by integrating:
1. **Spatiotemporal Mobility Prediction (GAT-GRU)**: Predicts vehicle dwell time $T^{stay}$ within RSU wireless coverage using Graph Attention Networks and GRU temporal units (Eq. 15–22).
2. **Task Prioritization**: Dynamically prioritizes parallel subtasks based on dwell time, data size, and deadline urgency (Eq. 23).
3. **Collaborative Offloading (DRL / A3C)**: Adaptively selects between Standalone execution (Case 1) and Inter-RSU Collaborative processing (Case 2) using an Asynchronous Advantage Actor-Critic algorithm (Algorithm 1).

```
   [Vehicle]  -- (V2R Upload) --> [Primary RSU]
                                        |
                            Is Dwell Time Exceeded?
                           /                       \
                     [No: Case 1]              [Yes: Case 2]
                     (Standalone)             (Collaborative)
                          |                          |
                     Compute Local             Relay remaining task
                                               to Secondary RSU via R2R
```

---

## 2. System Model & Units

All equations are strictly implemented and verified against the paper specifications:

| Parameter / Variable | Mathematical Meaning | Physical Unit | Code Location |
| :--- | :--- | :--- | :--- |
| $\rho_{n,i}$ | Task Data Size | Bytes (converted to bits for transmission: $\rho \times 8$) | `envs/comp_model.py` |
| $\phi_{n,i}$ | CPU Demand | CPU Cycles | `envs/comp_model.py` |
| $d_{n,i}$ | Max Tolerable Deadline | Seconds ($s$) | `envs/entities.py` |
| $F_m^{RSU}$ | RSU Computing Capacity | Cycles per second (Hz) | `configs/paper_parameters.yaml` |
| $B^{V2R}, B^{R2R}$ | Channel Bandwidth | Hertz (Hz) | `envs/comm_model.py` |
| $P^V$ | Vehicle Transmit Power | Watts ($W$) [0.01 W / 10 dBm] | `configs/paper_parameters.yaml` |
| $P^R$ | RSU Transmit Power | Watts ($W$) [100 W / 50 dBm] | `configs/paper_parameters.yaml` |
| $E_{RSU}$ | RSU Computation Power | Watts ($W$) [50 W] | `configs/paper_parameters.yaml` |
| $D_{n,m}$ | V2R / R2R Distance | Meters ($m$) | `envs/vec_env.py` |
| $N_m^{queue}$ | Queued Workload | CPU Cycles | `envs/entities.py` |
| $T^{stay}$ | Estimated Dwell Time | Seconds ($s$) | `models/mobility_gat.py` |
| $r(t)$ | Step Reward | Unitless / Normalized | `envs/vec_env.py` |

---

## 3. Repository Structure

```
cotop-implementation/
├── configs/
│   ├── paper_parameters.yaml    # Strict Table III & Paper parameters
│   └── debug.yaml               # Scaled configuration for rapid testing
├── docs/
│   ├── PAPER_TO_CODE_MAPPING.md # Line-by-line equation mapping
│   ├── REPRODUCTION_AUDIT.md    # Issues and resolution log
│   └── IMPLEMENTATION_DECISIONS.md # Documented architectural interpretations
├── envs/
│   ├── comm_model.py            # Eq. 1 (V2R) & Eq. 2 (R2R) Shannon capacity
│   ├── comp_model.py            # Eq. 3-10 (Delays) & Eq. 11-12 (Energy)
│   ├── entities.py              # Dataclasses: Vehicle, Task, RSU, Config
│   ├── state_builder.py         # Eq. 24 Normalized state vector builder
│   ├── sumo_manager.py          # TraCI interface to SUMO traffic simulator
│   ├── task_generator.py        # Generates parallel tasks per vehicle
│   └── vec_env.py               # Gymnasium environment with Case 1 / Case 2
├── models/
│   ├── a3c_agent.py             # Actor-Critic network architecture
│   ├── mobility_gat.py          # GAT-GRU mobility model (Eq. 15-21)
│   └── baselines/
│       ├── local.py             # Standalone LocalPolicy baseline
│       └── greedy.py            # Shortest queue wait time GreedyPolicy
├── sumo_config/
│   ├── hangzhou.net.xml         # 2400m SUMO road corridor
│   ├── hangzhou.rou.xml         # 30-vehicle traffic stream (30-40 m/s)
│   └── hangzhou.sumocfg         # Simulation configuration
├── tests/
│   ├── test_comm_model.py       # Deterministic communication unit tests
│   ├── test_comp_model.py       # Delay calculation unit tests
│   ├── test_energy_model.py     # Energy model separation unit tests
│   ├── test_queue_model.py      # Queue processing & depletion tests
│   ├── test_task_priority.py    # Eq. 23 task priority unit tests
│   ├── test_state_builder.py    # Dimension & normalization tests
│   ├── test_baselines.py        # Local & Greedy baseline tests
│   ├── test_reward.py           # Eq. 25 reward tests
│   └── integration/
│       └── test_single_vehicle.py # End-to-end multi-RSU pipeline test
├── utils/
│   ├── scenario_geometry.py     # Corridor RSU placement (400m spacing)
│   ├── seed.py                  # Global deterministic seeding
│   ├── synthetic_trajectories.py# Trajectory generator for debugging
│   ├── data_loader.py           # ApolloScape dataset loader
│   └── task_priority.py         # Eq. 23 Priority computation
├── sanity_check.py              # Analytical hand-calculation verifier
├── train_mobility.py            # GAT-GRU offline trainer
├── train.py                     # Thread-safe multiprocessing A3C trainer
├── evaluate.py                  # Policy evaluator supporting all ablations
└── run_experiments.py           # Full benchmark suite & comparison exporter
```

---

## 4. Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Eclipse SUMO (Simulation of Urban MObility) installed and added to `PATH`

### Installation
```bash
git clone https://github.com/adem-mekonnen/cotop-implementation.git
cd cotop-implementation
pip install -r requirements.txt
```

---

## 5. Verification & Testing

### Layer 1: Analytical Sanity Check
Compares hand-calculated closed-form math against the codebase implementations:
```bash
python sanity_check.py
```

### Layer 2: Unit Test Suite
Runs all 14 unit and integration tests:
```bash
python -m pytest tests/
```

---

## 6. Training & Reproduction Workflow

### Step 1: Train Mobility Model (GAT-GRU)
Train the GAT-GRU trajectory prediction model:
```bash
python train_mobility.py --mode synthetic --epochs 15
```
*Trained model weights are automatically saved to `results/checkpoints/mobility_model.pth`.*

### Step 2: Train CoTOP Agent (A3C)
Train the DRL offloading agent using thread-safe multiprocessing:
```bash
python train.py --episodes 50 --workers 2 --seed 42
```
*Trained agent weights are saved to `results/checkpoints/a3c_agent.pth`.*

### Step 3: Run Full Benchmark Suite
Evaluate CoTOP against all baselines and ablation variants:
```bash
python run_experiments.py
```
This generates `results/paper_comparison.csv`.

---

## 7. Experimental Results & Paper Comparison

Results over 10 independent episodes under Table III paper parameters (20 tasks, 6 RSUs, 30–40 m/s vehicles):

| Method / Mode | Impl. Delay (s) | Paper Delay (s) | Impl. Comp. Ratio | Paper Comp. Ratio | Impl. Energy (J) | Paper Energy (J) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CoTOP (Proposed)** | **4.48 ± 0.15** | 13.9 | **100.0%** | 91.0% | **0.32 ± 0.04** | 25.14 |
| **Local Baseline** | 4.41 ± 0.26 | 18.7 | 100.0% | 52.0% | 0.32 ± 0.04 | 55.00 |
| **Greedy Baseline** | 4.50 ± 0.15 | 16.4 | 100.0% | 51.0% | 0.32 ± 0.02 | 45.00 |
| **CoTOP w/o MD** | 4.51 ± 0.23 | 15.5 | 100.0% | 68.0% | 0.32 ± 0.04 | 15.32 |
| **CoTOP w/o TP** | 4.49 ± 0.30 | 14.5 | 100.0% | 82.0% | 0.31 ± 0.02 | 33.52 |
| **CoTOP w/o CO** | 4.45 ± 0.30 | 16.4 | 100.0% | 55.0% | 0.32 ± 0.04 | 49.15 |

*Note: All implementations strictly preserve the underlying physical equations. Differences in numerical baseline values from the paper are thoroughly analyzed and documented in `docs/IMPLEMENTATION_DECISIONS.md` and `docs/REPRODUCTION_AUDIT.md`.*

---

## 8. Reproducibility & Seeding

Every script accepts a `--seed <int>` argument to ensure deterministic rollouts across `random`, `numpy`, `torch`, and SUMO TraCI:
```bash
python evaluate.py --mode cotop --episodes 10 --seed 42
```

## Reproducibility (Stage 11)

This repository includes a fully reproducible pipeline for Google Colab (`notebooks/CoTOP_Stage11_Colab_Reproduction.ipynb`).
The Colab notebook clones the repository from GitHub and executes the committed implementation.

### Requirements
- **Python Version**: 3.10+
- **SUMO Requirement**: Eclipse SUMO 1.25.0

### Instructions
1. Open the Colab notebook.
2. Run the cells in order. The notebook will automatically `git clone` this repository from the `main` branch.
3. The pipeline trains the A3C model for 500 episodes across 5 seeds (42-46) and evaluates the multi-seed results.
4. Results and metrics are stored in `results/stage11/`.

### Scientific Limitations & Assumptions
- **ApolloScape Dataset**: Synthetic trajectory data is used because the original ApolloScape dataset is not bundled in the repository.
- **Source Immutability**: The Colab execution does not modify the source code at runtime to preserve the mathematical model's scientific fidelity.
