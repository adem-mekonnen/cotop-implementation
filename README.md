# CoTOP: Mobility-Aware Collaborative Task Offloading for Parallel Tasks

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![SUMO](https://img.shields.io/badge/SUMO-1.21.0+-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official modular implementation of the research paper:  
**"Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing"**  
*Published in IEEE Transactions on Mobile Computing (TMC), Vol. 25, No. 4, April 2026.*

---

## 📖 Paper Overview
CoTOP addresses the challenges of vehicle mobility and parallel task dependencies in Vehicular Edge Computing (VEC). It integrates a **Graph Attention Network (GAT)** for mobility prediction and an **Asynchronous Advantage Actor-Critic (A3C)** DRL agent for optimized collaborative offloading.

### Key Contributions:
- **Mobility Awareness:** GAT-GRU model to estimate vehicle dwell time ($T^{stay}$).
- **Parallel Task Processing:** Handling multiple simultaneous tasks per vehicle.
- **Dynamic Prioritization:** A task sequencing algorithm based on urgency and data size.

---

## 📂 Implementation Mapping

| Component | Files | Mathematical Reference |
| :--- | :--- | :--- |
| **Physics Engine** | `envs/comm_model.py`, `envs/comp_model.py` | Shannon Formula (**Eq 1-2**), Delay/Energy (**Eq 3-12**) |
| **Mobility Model** | `models/mobility_gat.py` | GAT-GRU Architecture (**Eq 15-22**) |
| **Task Priority** | `utils/task_priority.py` | Priority Weighting (**Eq 23**) |
| **State Assembly** | `envs/state_builder.py` | State Vector Construction (**Eq 24**) |
| **RL Environment** | `envs/vec_env.py`, `envs/sumo_manager.py` | Gymnasium MDP Wrapper |
| **Offloading Brain** | `models/a3c_agent.py`, `train.py` | A3C DRL Algorithm (**Algorithm 1**) |

---

## ⚙️ Simulation Parameters (Verified Table III)
The simulation parameters are synchronized with **Journal Page 5550**:

- **RSUs:** 6 units with 400m communication range.
- **Bandwidth:** V2R: [20, 100] Mbps | R2R: 50 MHz.
- **Power (Watts):** Vehicle: 0.01W (10 dBm) | RSU: 100W (50 dBm).
- **Noise Power:** 0.001W (0.001 dBm).
- **Path Loss:** Fixed Loss ($K$) = 1000 | Exponent ($\sigma$) = 2.
- 

---

