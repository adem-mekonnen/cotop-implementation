# PHASE 2 CoTOP ALGORITHMIC FIDELITY AUDIT

## 1. Executive Summary
This document provides a mathematical and algorithmic trace of the CoTOP repository implementation against the reference paper. We evaluate the core RL loop, Mobility GAT-GRU, Actor-Critic structures, action space constraints, and reward functions. 

**FINAL FIDELITY RATING: PASS** (Following the correction of an Action Masking anomaly in `train.py` and `evaluate.py`).

## 2. Mathematical Component Trace

| Paper Equation / Component | Mathematical Meaning | Repository Function | Source File | Exact Implementation | Tensor Shape | Fidelity Classification |
|---|---|---|---|---|---|---|
| **Environment Reset** | Initialize simulation | `VECEnv.reset()` | `envs/vec_env.py` | TraCI init, sets `sim_time=0`, resets queues | N/A | **PASS** |
| **State Construction** | $S = \{T_i, S_v, S_r\}$ | `build_state()` | `envs/state_builder.py` | Flattens 4(task) + 4I(veh tasks) + 5R(rsu) | `(4 + 4I + 5R,)` | **PASS** |
| **GAT Expansion (Eq 15)** | Transform coords to embed | `coordinate_expansion_mlp` | `models/mobility_gat.py` | 2-layer MLP (Input->64->64) | `(N, 64)` | **PASS** |
| **GAT Layer 1 (Eq 17)** | Spatial multi-head attention | `GATConv(concat=True)` | `models/mobility_gat.py` | 4 heads, `out=16`, concatenated to 64 | `(N, 64)` | **PASS** |
| **GAT Layer 2 (Eq 18)** | Mean head aggregation | `GATConv(concat=False)` | `models/mobility_gat.py` | 4 heads, averaged across heads | `(N, 64)` | **PASS** |
| **GRU Encoder (Eq 19)** | Temporal sequence encoding | `encoder_gru` | `models/mobility_gat.py` | `nn.GRU` (hidden=64) | `(N, T, 64)` | **PASS** |
| **Actor Output (Eq 20)** | Policy over $R$ RSUs + Local | `ActorCritic.actor_head` | `models/a3c_agent.py` | `nn.Linear(hidden, num_actions)` | `(num_actions,)` | **PASS** |
| **Critic Output** | State Value $V(s)$ | `ActorCritic.critic_head` | `models/a3c_agent.py` | `nn.Linear(hidden, 1)` | `(1,)` | **PASS** |
| **Action Masking** | Prevent invalid offloading | `get_action_mask()` | `envs/vec_env.py` | Boolean array masking unavailable RSUs | `(num_actions,)` | **CONDITIONAL PASS** (Fixed) |
| **Eq 25 (Reward)** | Weighted cost + penalty | `_calculate_reward()` | `envs/vec_env.py` | `-(eps*delay + (1-eps)*energy) - Z` | Scalar | **PASS** |
| **Discounting** | $\gamma$ | `gamma=0.99` | `train.py` | Cumulative reverse iteration | Scalar | **PASS** |
| **Advantage Estimation** | $A = R - V(s)$ | `worker_process` | `train.py` | `returns - values.detach()` | Scalar/Batch | **PASS** |
| **Policy Update (A3C)** | $\nabla \log \pi \times A$ | `actor_loss` | `train.py` | `-(log_probs * advantages).mean()` | Scalar | **PASS** |
| **Entropy Regularization** | $H(\pi)$ | `entropy` | `train.py` | `-(probs * log(probs)).mean()` | Scalar | **PASS** (Fixed mask) |
| **Critic Update** | L2 Loss on Returns | `critic_loss` | `train.py` | `F.mse_loss(values, returns)` | Scalar | **PASS** |
| **Optimizer** | Parameter update | `SharedAdam` | `train.py` | Asynchronous param sharing | N/A | **PASS** |
| **Episode termination** | End of episode | `done = terminated` | `vec_env.py` | Terminated when task lists empty | N/A | **PASS** |
| **Vehicle departure** | Leave SUMO bounds | `_advance_sumo_time_slot` | `vec_env.py` | Vehicles cleaned up, tasks failed | N/A | **PASS** |
| **Task completion** | Eq 24 / Task End | `step()` | `vec_env.py` | `task.completed = True` if delay $\le d$ | N/A | **PASS** |
| **Task failure** | Exceeds deadline | `step()` | `vec_env.py` | Evaluated against `max_delay_d` | N/A | **PASS** |
| **Queue state** | RSU Queues | `RSU.update()` | `entities.py` | FIFO subtraction by CPU speed | N/A | **PASS** |
| **Parallel task exec** | Eq 1 / 4 | `task_generator.py` | `task_generator.py` | I tasks dispatched synchronously | N/A | **PASS** |
| **Multi-veh concurrency** | SUMO | `_advance_sumo_time_slot`| `vec_env.py` | Multiple vehicles active in `active_vehicles` | N/A | **PASS** |

## 3. Discrepancy Log & Resolutions

### Discrepancy 1: Action Masking Bypass
- **Description**: The paper dictates that only feasible RSUs (within range or existing) can be selected. The environment provides `get_action_mask()` which accurately determines this feasibility constraint. However, prior to this audit, `train.py` directly sampled from the softmax of the raw `ActorCritic` logits without applying this mask. Similarly, `evaluate.py` directly applied `argmax`.
- **Classification**: Implementation Error.
- **Minimal Fix**: 
  - In `train.py`, we now fetch the mask and apply `policy_logits[~mask] = -1e9` prior to softmax (both for the sampling logic and the entropy calculation).
  - In `evaluate.py`, we apply `logits[~mask] = -1e9` prior to `argmax`.
- **Test Coverage**: Added `test_phase2_action_masking_integration.py` containing `test_training_action_masking` and `test_evaluation_action_masking` to verify mathematical isolation of invalid actions. Tests pass.
- **Resolution**: Implemented and verified.

## 4. Conclusion
With the Action Masking fix fully integrated and tested, the CoTOP mathematical and algorithmic fidelity strictly mirrors the formulations presented in the target paper. The repository is mathematically ready for phase 2 full factorial evaluation.
