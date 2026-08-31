# PHASE 2 EXPERIMENTAL PROTOCOL RECONSTRUCTION

## 1. Overview
This document reconstructs the experimental protocol for the target paper ("Mobility-Aware Collaborative Task Offloading for Parallel Tasks in Vehicular Edge Computing") line-by-line. Every parameter is classified into one of the following categories:
- **PAPER-SPECIFIED**: Explicitly stated in the paper text, tables, or equations.
- **REFERENCE-SPECIFIED**: Derived from a cited reference in the paper.
- **REPOSITORY-SPECIFIED**: Found in the author's reference code.
- **RECONSTRUCTED**: Derived logically through scientific forensic analysis to bridge gaps.
- **ASSUMED**: Filled in due to missing specification and lack of direct evidence.
- **UNKNOWN**: Cannot be determined.

---

## 2. Experimental Protocol Matrix

| # | Paper Quantity | Section/Table/Eq | Implementation File | Current Value | Required Value | Evidence Type | Confidence | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Number of vehicles | Table III | `configs/paper_parameters.yaml` | `[10, 30]` | `[10, 30]` | Text | High | PAPER-SPECIFIED |
| 2 | Number of RSUs | Table III | `configs/paper_parameters.yaml` | `6` | `6` | Text | High | PAPER-SPECIFIED |
| 3 | RSU placement | Sec IV (Hangzhou) | `utils/scenario_geometry.py` | `grid_200m` | Hangzhou | Map | High | RECONSTRUCTED |
| 4 | Road/grid geometry | Sec IV | `sumo_config/hangzhou.sumocfg` | Hangzhou 200m | Hangzhou 200m | Map | High | RECONSTRUCTED |
| 5 | Vehicle mobility | Sec III-D | `envs/sumo_manager.py` | SUMO TraCI | SUMO TraCI | Text | High | PAPER-SPECIFIED |
| 6 | Communication range | Table III | `configs/paper_parameters.yaml` | `400.0` m | `400.0` m | Text | High | PAPER-SPECIFIED |
| 7 | Task generation | Sec III-A | `envs/task_generator.py` | Parallel | Parallel | Text | High | PAPER-SPECIFIED |
| 8 | $I$ tasks per veh | Table III | `configs/paper_parameters.yaml` | `[20, 40]` | `[20, 40]` | Text | High | PAPER-SPECIFIED |
| 9 | Workload sizes ($\rho$) | Table III | `configs/paper_parameters.yaml` | `[2.0e6, 5.0e6]` | `[2M, 5M]` | Text | High | PAPER-SPECIFIED |
| 10 | Arrival intensity | Unspecified | `envs/vec_env.py` | Once per veh | N/A | Heuristic | Low | ASSUMED |
| 11 | Task CPU cycles ($\phi$) | Sec III-F | `configs/paper_parameters.yaml` | Max `10.0` Mcycles | Max `10.0` Mcycles | Text | High | PAPER-SPECIFIED |
| 12 | Task input data | Table III | `configs/paper_parameters.yaml` | `[2.0e6, 5.0e6]` | `[2M, 5M]` | Text | High | PAPER-SPECIFIED |
| 13 | Task output data | Unspecified | `envs/comp_model.py` | Ignored | Ignored | Literature | Med | ASSUMED |
| 14 | Deadlines ($d$) | Table III | `configs/paper_parameters.yaml` | `[20.0, 30.0]` s | `[20, 30]` s | Text | High | PAPER-SPECIFIED |
| 15 | Local execution | Baselines | `models/baselines/local.py` | Executed Locally | Executed Locally | Text | High | PAPER-SPECIFIED |
| 16 | RSU standalone | Eq 1-3 | `envs/comp_model.py` | Case 1 | Case 1 | Eq | High | PAPER-SPECIFIED |
| 17 | Collaborative exec | Eq 4-7 | `envs/comp_model.py` | Case 2 | Case 2 | Eq | High | PAPER-SPECIFIED |
| 18 | RSU queue model | Sec III-B | `envs/entities.py` | FIFO Capacity | FIFO Capacity | Text | High | RECONSTRUCTED |
| 19 | Comm delay | Eq 8-9 | `envs/comm_model.py` | Shannon-Hartley | Shannon-Hartley | Eq | High | PAPER-SPECIFIED |
| 20 | Comp delay | Eq 10 | `envs/comp_model.py` | CPU / Freq | CPU / Freq | Eq | High | PAPER-SPECIFIED |
| 21 | Energy model | Eq 11 | `envs/comp_model.py` | Power * Time | Power * Time | Eq | High | PAPER-SPECIFIED |
| 22 | Priority Eq. 23 | Eq 23 | `utils/task_priority.py` | `paper_literal` | Literal | Eq | High | PAPER-SPECIFIED |
| 23 | Coverage failure | Eq 25 | `envs/vec_env.py` | `completion_position`| Within Range | Eq | High | RECONSTRUCTED |
| 24 | State vector | Sec IV-A | `envs/state_builder.py` | 4+4*I+5*R | Paper defined | Text | Med | RECONSTRUCTED |
| 25 | Action vector | Sec IV-B | `envs/vec_env.py` | 1 + RSU Count | 1 + RSU Count | Text | High | PAPER-SPECIFIED |
| 26 | Action masking | Implicit | `envs/vec_env.py` | Strict Masking | Valid only | Heuristic | High | RECONSTRUCTED |
| 27 | CoTOP architecture | Sec IV-D | `models/a3c_agent.py` | A3C | A3C | Text | High | PAPER-SPECIFIED |
| 28 | GAT architecture | Sec IV-D | `models/mobility_gat.py` | GAT for Pos | GAT for Pos | Text | High | PAPER-SPECIFIED |
| 29 | GRU architecture | Sec IV-D | `models/mobility_gat.py` | GRU sequence | GRU sequence | Text | High | PAPER-SPECIFIED |
| 30 | Actor-critic arch | Sec IV-D | `models/a3c_agent.py` | Actor-Critic | Actor-Critic | Text | High | PAPER-SPECIFIED |
| 31 | Training episodes | Unspecified | `train.py` | `50` | N/A | Heuristic | Low | ASSUMED |
| 32 | Learning rate | Unspecified | `train.py` | `0.0002` | N/A | Heuristic | Low | ASSUMED |
| 33 | Discount factor ($\gamma$) | Unspecified | `train.py` | `0.99` | N/A | Standard | Med | ASSUMED |
| 34 | Entropy coeff | Unspecified | `train.py` | `0.01` | N/A | Standard | Med | ASSUMED |
| 35 | Exploration | Sec IV-D | `train.py` | Categorical | Softmax/Cat | Text | Med | RECONSTRUCTED |
| 36 | Optimizer | Unspecified | `train.py` | SharedAdam | N/A | Standard | Med | ASSUMED |
| 37 | Replay/batch | Sec IV-D | `train.py` | On-policy (None)| N/A | Text | High | PAPER-SPECIFIED |
| 38 | Evaluation protocol | Sec V | `evaluate.py` | Avg Metrics | Avg Metrics | Text | Med | RECONSTRUCTED |
| 39 | Random seeds | Unspecified | `utils/seed.py` | `42` | N/A | Standard | Low | ASSUMED |
| 40 | Aggregation method | Sec V | `scripts/run_phase2_aggregation_audit.py` | Veh Aggregate | Veh Aggregate | Audit | High | RECONSTRUCTED |

---

## 3. Protocol Lock

The experimental parameters listed above establish the strict boundary for the Phase 2 reproduction.
**Rule:** No implementation code mapping to `PAPER-SPECIFIED` or `RECONSTRUCTED` components shall be modified unless a mathematical error in the implementation itself is proven. `ASSUMED` parameters may be systematically ablated if required, but currently stand as the official reproduction settings.

**Final Decision**: The protocol is LOCKED. All training and testing must adhere to these exact constraints.
