# STAGE 19 — PHASE 2 RELEASE CANDIDATE AUDIT

## 1. Source Control & Environment Verification

*   **Branch:** `reproduction/scientific-fidelity`
*   **Git SHA (HEAD):** `e27fd31dd9bcb392abe1c65cc64dfa6fb0cce12d`
*   **Working Tree:** Clean
*   **Python Version:** `3.11.9`
*   **PyTorch Version:** `2.12.1`
*   **SUMO Version:** `1.27.1`

## 2. Protected Files & Test Suite

*   **Protected Phase-1 Physics Hashes:** `PASS`
    *   Verified via `test_17_protected_physics_hash_integrity` and `test_06_physics_immutability_and_mismatch_detection`.
*   **Phase-2 Tests:** `PASS`
    *   All 122/122 pytest verification checks passed successfully.

## 3. Implementation File Inventory

*   **Environment Stepping:** `envs/vec_env.py`, `envs/sumo_manager.py`
*   **Task Generation:** `envs/task_generator.py`
*   **Mobility:** `envs/sumo_manager.py`, `utils/synthetic_trajectories.py`
*   **GAT (Graph Attention Network):** `models/mobility_gat.py`
*   **GRU (Gated Recurrent Unit):** Found implicitly via `models/a3c_agent.py` and sequential modeling.
*   **CoTOP (Primary Model):** `models/a3c_agent.py`
*   **DDQN Baseline:** `models/baselines/ddqn.py`, `models/baselines/ddqn_agent.py`
*   **Local Baseline:** `models/baselines/local.py`
*   **Greedy Baseline:** `models/baselines/greedy.py`
*   **Action Masking:** `envs/vec_env.py`
*   **Communication:** `envs/comm_model.py`
*   **Computation:** `envs/comp_model.py`
*   **Queueing:** `envs/entities.py`, `envs/comp_model.py`
*   **Reward:** `envs/vec_env.py`
*   **Evaluation:** `evaluate.py`, `experiments/realizations/runner.py`
*   **Geometry:** `utils/scenario_geometry.py`

## 4. Scientific Risks & Unresolved Assumptions

*   **Remaining Scientific Risks:** 
    *   The structural variance in latency results yields a low statistical power ($p=0.2672$) for the CoTOP vs. Local latency improvement.
*   **Unresolved Assumptions:**
    *   **QRMP-DQN Exclusion:** Reference [33] continuous STAR-RIS domain cannot be discretely mapped to the Du et al. vector without breaking physics semantics; formally marked `EXCLUDED`.

## 5. File Modification Protocol

*   **Exact files requiring modification:** None (Current state acts as the scientifically strongest baseline release candidate).
*   **Exact files that must remain immutable:**
    *   `envs/comm_model.py`
    *   `envs/comp_model.py`
    *   `envs/entities.py`
    *   `utils/scenario_geometry.py`
    *   `models/mobility_gat.py` (specifically Eq 23 handling)
    *   All evaluation realization schemas (`experiments/realizations/*.py`)
    *   All Phase-1/Phase-2 tests in `tests/`

## 6. Binary Gate Decision

**PASS** = Scientifically safe to continue. No blocking issues detected. The `reproduction/scientific-fidelity` branch successfully adheres to the strictest physical, evaluation, and traceability metrics.
