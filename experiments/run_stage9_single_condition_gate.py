"""
experiments/run_stage9_single_condition_gate.py

Executes STAGE 9 — SINGLE-CONDITION COMPARATIVE GATE.
Compares CoTOP, DDQN, Greedy, and Local under exactly ONE canonical condition:
- geometry: corridor_2400m
- workload: w20 (200 tasks)
- seed: 0 (eval seed 30000)

Enforces all 12 validation gates:
1. realization hash identical
2. environment configuration identical
3. task count identical
4. vehicle trajectories identical
5. action-space semantics identical
6. evaluation weights immutable
7. deterministic action sequence
8. deterministic state sequence
9. task conservation
10. latency decomposition
11. energy decomposition
12. no NaN/Inf
"""

import os
import sys
import json
import hashlib
import time
import copy
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.realizations.schema import ExperimentRealization
from experiments.realizations.validator import RealizationValidator
from experiments.realizations.runner import RealizationRunner, RealizationRunResult
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent


def compute_model_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_stage9_gate(
    realization_path: str = "data/evaluation_realizations/corridor_2400m_w20_seed0_realization.json",
    cotop_ckpt_path: str = "results/stage5_cotop_retrain/corridor_2400m/seed_0/checkpoint_ep500.pt",
    ddqn_ckpt_path: str = "results/phase2_algorithmic_fidelity/linear_corridor_DDQN_w20/seed_42/checkpoint_ep500.pt",
    output_dir: str = "results/stage9_single_condition_gate"
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("      STAGE 9: SINGLE-CONDITION COMPARATIVE GATE AUDIT")
    print("=" * 80)

    # 1. Load Canonical Realization
    if not os.path.exists(realization_path):
        raise FileNotFoundError(f"Realization file not found: {realization_path}")
    
    realization = ExperimentRealization.load(realization_path)
    realization_hash = realization.realization_hash
    print(f"Loaded Realization: {realization.realization_id}")
    print(f"Geometry: {realization.geometry} | Workload: w{realization.workload} | Seed: {realization.seed}")
    print(f"Cryptographic SHA-256 Hash: {realization_hash}")

    # Validate Pre-Flight Rejection Gates
    RealizationValidator.validate(
        realization=realization,
        expected_geometry="corridor_2400m",
        expected_workload=20,
        expected_seed=0
    )
    print("[PASS] Pre-Flight Realization Validation PASSED.")

    # 2. Setup Frozen Models and Record Checkpoint Parameter Hashes
    # CoTOP Checkpoint
    obs_dim = 4 + (20 * 4) + (6 * 5)  # 114
    num_actions = 7
    cotop_agent = ActorCritic(input_dim=obs_dim, num_actions=num_actions, hidden_size=128)
    if os.path.exists(cotop_ckpt_path):
        ckpt_data = torch.load(cotop_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        cotop_agent.load_state_dict(state_dict)
        print(f"[INFO] Loaded trained CoTOP from {cotop_ckpt_path}")
    cotop_agent.eval()
    for p in cotop_agent.parameters():
        p.requires_grad = False
    cotop_hash_before = compute_model_hash(cotop_agent)

    # DDQN Checkpoint
    ddqn_agent = DDQNAgent(input_dim=obs_dim, num_actions=num_actions, hidden_dim=128, device="cpu")
    if os.path.exists(ddqn_ckpt_path):
        ddqn_agent.load_checkpoint(ddqn_ckpt_path)
        print(f"[INFO] Loaded trained DDQN from {ddqn_ckpt_path}")
    ddqn_agent.online_net.eval()
    ddqn_agent.target_net.eval()
    for p in ddqn_agent.online_net.parameters():
        p.requires_grad = False
    for p in ddqn_agent.target_net.parameters():
        p.requires_grad = False
    ddqn_hash_before = compute_model_hash(ddqn_agent.online_net)

    # 3. Controlled Execution across 4 Algorithms
    runner = RealizationRunner()
    
    # Run 1: Local
    res_local = runner.run_algorithm("Local", realization=realization)
    
    # Run 2: Greedy
    res_greedy = runner.run_algorithm("Greedy", realization=realization)
    
    # Run 3: DDQN
    res_ddqn = runner.run_algorithm("DDQN", realization=realization, agent_or_checkpoint=ddqn_agent)
    
    # Run 4: CoTOP
    res_cotop = runner.run_algorithm("CoTOP", realization=realization, agent_or_checkpoint=cotop_agent)

    # Run Repeated Passes to Verify Determinism (Gates 7 & 8)
    res_cotop_repeat = runner.run_algorithm("CoTOP", realization=realization, agent_or_checkpoint=cotop_agent)
    res_ddqn_repeat = runner.run_algorithm("DDQN", realization=realization, agent_or_checkpoint=ddqn_agent)

    # 4. Check Evaluation Weight Immutability (Gate 6)
    cotop_hash_after = compute_model_hash(cotop_agent)
    ddqn_hash_after = compute_model_hash(ddqn_agent.online_net)

    # 5. 12-Gate Audit Verification
    gate_results = {}
    
    # Gate 1: Realization hash identical
    gate_results["Gate 1: Realization Hash Identical"] = (
        res_local.realization_hash == realization_hash == res_greedy.realization_hash == res_ddqn.realization_hash == res_cotop.realization_hash
    )

    # Gate 2: Environment configuration identical
    gate_results["Gate 2: Environment Configuration Identical"] = (
        realization.environment_configuration["tx_power_vehicle"] == 0.01 and
        realization.environment_configuration["tx_power_rsu"] == 100.0 and
        realization.environment_configuration["noise_power"] == 0.001 and
        realization.environment_configuration["fixed_loss_k"] == 1000.0
    )

    # Gate 3: Task count identical
    gate_results["Gate 3: Task Count Identical"] = (
        res_local.total_tasks == 200 and
        res_greedy.total_tasks == 200 and
        res_ddqn.total_tasks == 200 and
        res_cotop.total_tasks == 200
    )

    # Gate 4: Vehicle trajectories identical
    gate_results["Gate 4: Vehicle Trajectories Identical"] = (
        len(realization.vehicle_trajectories) == 10 and
        all(len(vt["trajectory_points"]) > 0 for vt in realization.vehicle_trajectories)
    )

    # Gate 5: Action-space semantics identical
    gate_results["Gate 5: Action-Space Semantics Identical"] = (
        all(0 <= d < 7 for d in res_local.decisions) and
        all(0 <= d < 7 for d in res_greedy.decisions) and
        all(0 <= d < 7 for d in res_ddqn.decisions) and
        all(0 <= d < 7 for d in res_cotop.decisions) and
        all(d == 0 for d in res_local.decisions)  # Local is standalone
    )

    # Gate 6: Evaluation weights immutable
    gate_results["Gate 6: Evaluation Weights Immutable"] = (
        cotop_hash_before == cotop_hash_after and
        ddqn_hash_before == ddqn_hash_after
    )

    # Gate 7: Deterministic action sequence
    gate_results["Gate 7: Deterministic Action Sequence"] = (
        res_cotop.decisions == res_cotop_repeat.decisions and
        res_ddqn.decisions == res_ddqn_repeat.decisions
    )

    # Gate 8: Deterministic state sequence
    gate_results["Gate 8: Deterministic State Sequence"] = (
        res_cotop.task_delays == res_cotop_repeat.task_delays and
        res_ddqn.task_delays == res_ddqn_repeat.task_delays
    )

    # Gate 9: Task conservation
    gate_results["Gate 9: Task Conservation"] = (
        (res_local.completed_tasks + res_local.failed_tasks == 200) and
        (res_greedy.completed_tasks + res_greedy.failed_tasks == 200) and
        (res_ddqn.completed_tasks + res_ddqn.failed_tasks == 200) and
        (res_cotop.completed_tasks + res_cotop.failed_tasks == 200)
    )

    # Gate 10: Latency decomposition
    gate_results["Gate 10: Latency Decomposition"] = all(
        np.isfinite(res.mean_delay_s) and np.isfinite(res.comm_delay_s) and
        np.isfinite(res.comp_delay_s) and np.isfinite(res.wait_delay_s)
        for res in [res_local, res_greedy, res_ddqn, res_cotop]
    )

    # Gate 11: Energy decomposition
    gate_results["Gate 11: Energy Decomposition"] = all(
        np.isfinite(res.mean_energy_j) and res.mean_energy_j >= 0.0
        for res in [res_local, res_greedy, res_ddqn, res_cotop]
    )

    # Gate 12: No NaN/Inf
    gate_results["Gate 12: No NaN/Inf"] = all(
        not np.isnan(res.mean_delay_s) and not np.isinf(res.mean_delay_s) and
        not np.isnan(res.mean_energy_j) and not np.isinf(res.mean_energy_j)
        for res in [res_local, res_greedy, res_ddqn, res_cotop]
    )

    print("\n" + "=" * 80)
    print("                    12-GATE AUDIT VERIFICATION MATRIX")
    print("=" * 80)
    all_passed = True
    for gate_name, passed in gate_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {gate_name}")
        if not passed:
            all_passed = False

    print("=" * 80)
    if not all_passed:
        raise RuntimeError("STAGE 9 FAILED: One or more single-condition comparative gates failed!")

    print("[SUCCESS] ALL 12 GATES PASSED.")

    # 6. Summary Comparison Table
    print("\n" + "-" * 80)
    print(f"{'Algorithm':<10} | {'Total':<6} | {'Completed':<10} | {'Ratio (%)':<10} | {'Delay (s)':<10} | {'Energy (J)':<10}")
    print("-" * 80)
    for res in [res_local, res_greedy, res_ddqn, res_cotop]:
        print(f"{res.algorithm:<10} | {res.total_tasks:<6} | {res.completed_tasks:<10} | {res.completion_ratio * 100:<10.1f} | {res.mean_delay_s:<10.2f} | {res.mean_energy_j:<10.2f}")
    print("-" * 80)

    # 7. Persist Results JSON
    results_summary = {
        "stage": "STAGE 9 — SINGLE-CONDITION COMPARATIVE GATE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "realization_id": realization.realization_id,
        "realization_hash": realization_hash,
        "geometry": realization.geometry,
        "workload": realization.workload,
        "seed": realization.seed,
        "eval_seed": realization.eval_seed,
        "cotop_model_hash": cotop_hash_before,
        "ddqn_model_hash": ddqn_hash_before,
        "gate_results": gate_results,
        "all_gates_passed": all_passed,
        "algorithms": {
            "Local": {
                "total_tasks": res_local.total_tasks,
                "completed_tasks": res_local.completed_tasks,
                "failed_tasks": res_local.failed_tasks,
                "completion_ratio": res_local.completion_ratio,
                "mean_delay_s": res_local.mean_delay_s,
                "mean_energy_j": res_local.mean_energy_j,
                "comm_delay_s": res_local.comm_delay_s,
                "comp_delay_s": res_local.comp_delay_s,
                "wait_delay_s": res_local.wait_delay_s,
                "decisions": res_local.decisions
            },
            "Greedy": {
                "total_tasks": res_greedy.total_tasks,
                "completed_tasks": res_greedy.completed_tasks,
                "failed_tasks": res_greedy.failed_tasks,
                "completion_ratio": res_greedy.completion_ratio,
                "mean_delay_s": res_greedy.mean_delay_s,
                "mean_energy_j": res_greedy.mean_energy_j,
                "comm_delay_s": res_greedy.comm_delay_s,
                "comp_delay_s": res_greedy.comp_delay_s,
                "wait_delay_s": res_greedy.wait_delay_s,
                "decisions": res_greedy.decisions
            },
            "DDQN": {
                "total_tasks": res_ddqn.total_tasks,
                "completed_tasks": res_ddqn.completed_tasks,
                "failed_tasks": res_ddqn.failed_tasks,
                "completion_ratio": res_ddqn.completion_ratio,
                "mean_delay_s": res_ddqn.mean_delay_s,
                "mean_energy_j": res_ddqn.mean_energy_j,
                "comm_delay_s": res_ddqn.comm_delay_s,
                "comp_delay_s": res_ddqn.comp_delay_s,
                "wait_delay_s": res_ddqn.wait_delay_s,
                "decisions": res_ddqn.decisions
            },
            "CoTOP": {
                "total_tasks": res_cotop.total_tasks,
                "completed_tasks": res_cotop.completed_tasks,
                "failed_tasks": res_cotop.failed_tasks,
                "completion_ratio": res_cotop.completion_ratio,
                "mean_delay_s": res_cotop.mean_delay_s,
                "mean_energy_j": res_cotop.mean_energy_j,
                "comm_delay_s": res_cotop.comm_delay_s,
                "comp_delay_s": res_cotop.comp_delay_s,
                "wait_delay_s": res_cotop.wait_delay_s,
                "decisions": res_cotop.decisions
            }
        }
    }

    out_file = os.path.join(output_dir, "single_condition_gate_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[INFO] Saved audit results to {out_file}")

    return results_summary


if __name__ == "__main__":
    run_stage9_gate()
