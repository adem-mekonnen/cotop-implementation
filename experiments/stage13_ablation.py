"""
experiments/stage13_ablation.py

Executes STAGE 13 — CoTOP ABLATION REPRODUCTION.
Evaluates the 4 ablation conditions:
1. Full CoTOP
2. Mobility detection disabled (w/o MD)
3. Task priority disabled (w/o TP)
4. Collaboration disabled (w/o CO)

Across:
- Geometries: grid_200m, corridor_2400m
- Workloads: w20, w30, w40
- Seeds: 0, 1, 2, 3, 4

Uses identical pre-materialized evaluation realizations from data/evaluation_realizations/.
Generates:
- results/phase2_algorithmic_fidelity/table6_ablation.csv
"""

import os
import sys
import csv
import json
import time
import hashlib
from typing import Dict, List, Any

import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.realizations.schema import ExperimentRealization
from experiments.realizations.runner import RealizationRunner
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from models.a3c_agent import ActorCritic


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_ablation_realization(
    realization: ExperimentRealization,
    checkpoint_path: str,
    ablation_mode: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes a single ablation condition over the realization.
    ablation_mode: 'full', 'wo_md', 'wo_tp', 'wo_co'
    """
    obs_dim = 4 + (realization.workload * 4) + (len(realization.rsu_configurations) * 5)
    num_actions = len(realization.rsu_configurations) + 1

    # Load CoTOP model
    model = ActorCritic(input_dim=obs_dim, num_actions=num_actions, hidden_size=128)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Setup RSUs
    rsus = []
    for r_cfg in realization.rsu_configurations:
        rsus.append({
            "id": r_cfg["rsu_id"],
            "loc": np.array(r_cfg["location"]),
            "cpu_f": float(r_cfg["cpu_capacity_f"]),
            "q_cycles": float(r_cfg["initial_queued_cycles"]),
            "p_tx": float(r_cfg["transmission_power_P_R"]),
            "b_v2r": float(r_cfg["bandwidth_v2r"]),
            "b_r2r": float(r_cfg["bandwidth_r2r"]),
            "range": float(r_cfg["comm_range"])
        })

    veh_traj_map = {vt["vehicle_id"]: vt for vt in realization.vehicle_trajectories}
    mob_map = {ms["vehicle_id"]: ms for ms in realization.mobility_states}
    map_scale = 200.0 if realization.geometry in ["grid_200m", "urban_manhattan"] else 2400.0

    # Determine task processing order
    tasks_to_process = list(realization.tasks)
    if ablation_mode == "wo_tp":
        # Disable task priority: process in arrival/generation FIFO order
        tasks_to_process.sort(key=lambda t: t["task_id"])
    else:
        # Full, wo_md, wo_co use the realization priority order
        tasks_to_process.sort(key=lambda t: t.get("priority_weight", 0.0), reverse=True)

    task_delays = []
    task_energies = []
    decisions = []
    completed_count = 0
    failed_count = 0
    collab_count = 0

    env_cfg = realization.environment_configuration
    p_v = float(env_cfg.get("tx_power_vehicle", 0.01))
    p_r = float(env_cfg.get("tx_power_rsu", 100.0))
    p_comp = 50.0  # W
    noise_power = float(env_cfg.get("noise_power", 0.001))
    fixed_loss_k = float(env_cfg.get("fixed_loss_k", 1000.0))
    path_loss_factor = float(env_cfg.get("path_loss_factor", 2.0))

    for task_dict in tasks_to_process:
        v_id = task_dict["vehicle_id"]
        gen_time = task_dict["generation_timestamp"]
        size_rho = task_dict["size_rho"]
        cpu_phi = task_dict["cpu_phi"]
        max_delay = task_dict["max_delay_d"]
        p_weight = 0.0 if ablation_mode == "wo_tp" else task_dict["priority_weight"]

        # Vehicle position
        vt = veh_traj_map[v_id]
        points = vt["trajectory_points"]
        closest_pt = min(points, key=lambda pt: abs(pt["timestamp"] - gen_time))
        v_pos = np.array([closest_pt["x"], closest_pt["y"]])
        v_speed = float(closest_pt["speed"])

        # Primary RSU
        dists = [np.linalg.norm(v_pos - r["loc"]) for r in rsus]
        primary_rsu_idx = int(np.argmin(dists))
        primary_dist = dists[primary_rsu_idx]

        # Dwell time
        if ablation_mode == "wo_md":
            # Disabled mobility detection: dwell time is unknown (0.0)
            t_stay = 0.0
        else:
            dwell_dict = mob_map[v_id]["predicted_dwell_time_per_rsu"]
            t_stay = dwell_dict.get(str(primary_rsu_idx), 10.0)

        # Build State Vector
        s_ego = [
            v_pos[0] / map_scale,
            v_pos[1] / map_scale,
            v_speed / 40.0,
            min(t_stay / 100.0, 1.0)
        ]
        s_tasks = [
            size_rho / 5.0e6,
            cpu_phi / 10.0e6,
            max_delay / 30.0,
            p_weight
        ] * realization.workload
        s_rsus = []
        for r in rsus:
            s_rsus.extend([
                r["loc"][0] / map_scale,
                r["loc"][1] / map_scale,
                r["cpu_f"] / 4.0e9,
                min(r["q_cycles"] / 1.0e9, 1.0),
                r["p_tx"] / 100.0
            ])
        state_vec = np.array(s_ego + s_tasks + s_rsus, dtype=np.float32)

        # Action Feasibility Mask
        action_mask = np.zeros(num_actions, dtype=bool)
        action_mask[0] = True

        if ablation_mode != "wo_co":
            for r_idx, r in enumerate(rsus):
                if dists[r_idx] <= r["range"] * 2.0:
                    action_mask[r_idx + 1] = True
        # Note: if ablation_mode == 'wo_co', action_mask remains strictly [True, False, False, ...]

        # Action Selection
        with torch.no_grad():
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(state_t)
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
            logits = torch.where(mask_t, logits.squeeze(0), torch.tensor(-1e9, device=device))
            action = torch.argmax(logits).item()

        decisions.append(int(action))

        # Execution Physics
        rsu1 = rsus[primary_rsu_idx]
        v2r_rate = compute_v2r_rate(
            distance=primary_dist,
            bandwidth_B=rsu1["b_v2r"],
            power_P_V=p_v,
            noise_power=noise_power,
            fixed_loss_k=fixed_loss_k,
            path_loss_factor=path_loss_factor
        )
        t_wait_rsu1 = rsu1["q_cycles"] / rsu1["cpu_f"] if rsu1["cpu_f"] > 0 else 0.0

        if action == 0 or action - 1 == primary_rsu_idx:
            # Case 1: Standalone
            t_total, e_total = calculate_case1_standalone(
                task_size_rho=size_rho,
                task_cpu_phi=cpu_phi,
                w_v2r=v2r_rate,
                rsu_cpu_f=rsu1["cpu_f"],
                power_v=p_v,
                compute_power_rsu=p_comp,
                t_wait=t_wait_rsu1
            )
            rsu1["q_cycles"] += cpu_phi
        else:
            # Case 2: Collaboration
            collab_count += 1
            sec_rsu_idx = action - 1
            rsu2 = rsus[sec_rsu_idx]
            r2r_dist = np.linalg.norm(rsu1["loc"] - rsu2["loc"])
            r2r_rate = compute_r2r_rate(
                distance=r2r_dist,
                bandwidth_B=rsu1["b_r2r"],
                power_P_R=p_r,
                noise_power=noise_power,
                fixed_loss_k=fixed_loss_k,
                path_loss_factor=path_loss_factor
            )
            t_wait_rsu2 = rsu2["q_cycles"] / rsu2["cpu_f"] if rsu2["cpu_f"] > 0 else 0.0

            # Under wo_md, dwell lookahead t1 is 0.0, so entire task is transferred
            dwell_t1 = 0.0 if ablation_mode == "wo_md" else t_stay

            t_total, e_total = calculate_case2_collaboration(
                task_size_rho=size_rho,
                task_cpu_phi=cpu_phi,
                w_v2r=v2r_rate,
                w_r2r=r2r_rate,
                rsu1_cpu_f=rsu1["cpu_f"],
                rsu2_cpu_f=rsu2["cpu_f"],
                t1_dwell_time=dwell_t1,
                power_v=p_v,
                tx_power_rsu1=p_r,
                compute_power_rsu1=p_comp,
                compute_power_rsu2=p_comp,
                t_wait=t_wait_rsu2
            )
            phi_1 = min(rsu1["cpu_f"] * dwell_t1, cpu_phi)
            phi_rest = max(cpu_phi - phi_1, 0.0)
            rsu1["q_cycles"] += phi_1
            rsu2["q_cycles"] += phi_rest

        task_delays.append(t_total)
        task_energies.append(e_total)

        if t_total <= max_delay:
            completed_count += 1
        else:
            failed_count += 1

    total_tasks = len(tasks_to_process)
    completion_ratio = completed_count / total_tasks if total_tasks > 0 else 1.0
    collab_ratio = collab_count / total_tasks if total_tasks > 0 else 0.0

    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_count,
        "failed_tasks": failed_count,
        "completion_ratio": round(completion_ratio, 4),
        "collab_ratio": round(collab_ratio, 4),
        "mean_delay": round(float(np.mean(task_delays)), 4),
        "std_delay": round(float(np.std(task_delays)), 4),
        "mean_energy": round(float(np.mean(task_energies)), 4),
        "std_energy": round(float(np.std(task_energies)), 4),
        "decisions": decisions,
        "task_delays": task_delays,
        "task_energies": task_energies
    }


def run_all_ablations():
    print("=" * 80)
    print("      STAGE 13: CoTOP ABLATION EXPERIMENT REPRODUCTION")
    print("=" * 80)

    geometries = ["grid_200m", "corridor_2400m"]
    workloads = [20, 30, 40]
    seeds = [0, 1, 2, 3, 4]
    
    ablation_conditions = [
        ("Full CoTOP", "full", "None (Baseline Full Architecture)"),
        ("w/o MD", "wo_md", "Mobility Detection Disabled (s_ego[3]=0.0, dwell_t1=0.0)"),
        ("w/o TP", "wo_tp", "Task Priority Disabled (FIFO task order, p_weight=0.0)"),
        ("w/o CO", "wo_co", "Collaborative Offloading Disabled (action_mask[1:]=False, Case 1 Only)")
    ]

    all_records = []
    seen_cells = set()

    for geom in geometries:
        for w in workloads:
            for s in seeds:
                realization_file = os.path.join("data", "evaluation_realizations", f"{geom}_w{w}_seed{s}_realization.json")
                if not os.path.exists(realization_file):
                    raise FileNotFoundError(f"Missing realization: {realization_file}")

                realization = ExperimentRealization.load(realization_file)
                realization_hash = realization.realization_hash

                ckpt_path = os.path.join("results", "phase2_algorithmic_fidelity", geom, "CoTOP", f"w{w}", f"seed_{s}", "checkpoint_ep500.pt")
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Missing CoTOP checkpoint: {ckpt_path}")

                for cond_name, cond_mode, code_path_desc in ablation_conditions:
                    cell_key = (cond_name, geom, w, s)
                    if cell_key in seen_cells:
                        raise ValueError(f"Duplicate cell: {cell_key}")
                    seen_cells.add(cell_key)

                    # Evaluate pass 1 & pass 2 for determinism
                    res1 = run_ablation_realization(realization, ckpt_path, cond_mode)
                    res2 = run_ablation_realization(realization, ckpt_path, cond_mode)

                    # Invariant Assertions
                    n_gen = w * 10
                    assert res1["total_tasks"] == n_gen, f"Task count mismatch: {res1['total_tasks']} vs {n_gen}"
                    assert res1["completed_tasks"] + res1["failed_tasks"] == n_gen, "Task conservation violated"
                    assert res1["decisions"] == res2["decisions"], "Non-deterministic decisions"
                    assert res1["task_delays"] == res2["task_delays"], "Non-deterministic delays"
                    assert np.isfinite(res1["mean_delay"]) and np.isfinite(res1["mean_energy"]), "NaN/Inf detected"

                    record = {
                        "ablation_condition": cond_name,
                        "geometry": geom,
                        "workload": w,
                        "seed": s,
                        "realization_hash": realization_hash,
                        "mean_delay": res1["mean_delay"],
                        "std_delay": res1["std_delay"],
                        "mean_energy": res1["mean_energy"],
                        "std_energy": res1["std_energy"],
                        "completion_ratio": res1["completion_ratio"],
                        "collab_ratio": res1["collab_ratio"],
                        "N_generated": n_gen,
                        "N_completed": res1["completed_tasks"],
                        "N_failed": res1["failed_tasks"],
                        "disabled_code_path": code_path_desc,
                        "invariants_passed": True
                    }
                    all_records.append(record)
                    print(f"[{cond_name:12s} | {geom:13s} | w{w:2d} | Seed {s}] Delay: {record['mean_delay']:.4f}s | Energy: {record['mean_energy']:.4f}J | Collab: {record['collab_ratio']*100:.1f}%")

    # Cardinality check
    expected_total = len(geometries) * len(workloads) * len(seeds) * len(ablation_conditions)
    assert len(all_records) == expected_total == 120, f"Expected {expected_total} rows, got {len(all_records)}"

    # Save results/phase2_algorithmic_fidelity/table6_ablation.csv
    out_csv = os.path.join("results", "phase2_algorithmic_fidelity", "table6_ablation.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "ablation_condition",
        "geometry",
        "workload",
        "seed",
        "realization_hash",
        "mean_delay",
        "std_delay",
        "mean_energy",
        "std_energy",
        "completion_ratio",
        "collab_ratio",
        "N_generated",
        "N_completed",
        "N_failed",
        "disabled_code_path",
        "invariants_passed"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 80)
    print(f"[COMPLETE] Table 6 Ablation CSV generated successfully ({len(all_records)} rows).")
    print(f"File: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_ablations()
