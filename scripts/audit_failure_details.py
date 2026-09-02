#!/usr/bin/env python3
"""
scripts/audit_failure_details.py
Deep audit of every failed task in realization_corridor_2400m_w20_seed42.json.
Extracts vehicle positions, arrival times, deadlines, delays, coverage boundaries, and predicates.
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from envs.vec_env import get_euclidean_distance
from utils.scenario_geometry import get_rsu_positions

def main():
    print("=" * 75)
    print("   PHASE 3 — DETAILED AUDIT OF THE 7 FAILED TASKS")
    print("=" * 75)

    scenario = "corridor_2400m"
    workload = 20
    seed = 42
    
    with open(os.path.join(root_dir, "configs/paper_parameters.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["num_tasks_per_vehicle_range"] = [workload, workload]
    config = SimulationConfig(**cfg_dict)
    
    realization_path = os.path.join(root_dir, "data", "evaluation_realizations", f"realization_{scenario}_w{workload}_seed{seed}.json")
    
    # Run AlwaysLocal
    env_local = FrozenVECEnv(config=config, realization_path=realization_path)
    env_local.reset()
    
    rsu_positions = get_rsu_positions(config.num_rsus, None, scenario_mode=scenario)
    print(f"RSU Positions in {scenario}:")
    for idx, pos in enumerate(rsu_positions):
        print(f"  RSU {idx}: {pos} (Coverage: [{pos[0]-200:.1f}, {pos[0]+200:.1f}])")
        
    failed_task_records = []
    all_local_records = []
    
    task_idx = 0
    while len(env_local.pending_tasks) > 0:
        curr_veh, curr_task = env_local.pending_tasks[0]
        v_id = curr_veh.v_id
        task_id = curr_task.task_id
        task_size = curr_task.size_rho
        task_cpu = curr_task.cpu_phi
        deadline = curr_task.max_delay_d
        arrival_pos = tuple(curr_veh.pos)
        speed = curr_veh.speed
        sim_time_arr = env_local.sim_time
        
        target_rsu = min(env_local.rsus, key=lambda r: get_euclidean_distance(curr_veh.pos, r.location))
        target_rsu_id = target_rsu.rsu_id
        target_rsu_loc = target_rsu.location
        
        # Distance to primary RSU at arrival
        dist_arr = get_euclidean_distance(arrival_pos, target_rsu_loc)
        
        # Calculate coverage exit time for primary RSU
        # Vehicle moves in +x direction: x(t) = pos[0] + speed * t
        # Exit occurs when x(t) > target_rsu_loc[0] + 200.0
        rsu_exit_x = target_rsu_loc[0] + config.rsu_comm_range
        dist_to_exit = rsu_exit_x - arrival_pos[0]
        time_to_exit = dist_to_exit / speed if speed > 0 else float('inf')
        
        obs, reward, terminated, truncated, info = env_local.step(0)
        
        delay = info["delay"]
        energy = info["energy"]
        comm_delay = info["comm_delay"]
        comp_delay = info["comp_delay"]
        wait_delay = info["wait_delay"]
        completed = info["completed"]
        fail_deadline = info["fail_deadline"]
        fail_coverage = info["fail_coverage"]
        fail_reason = info["failure_reason"]
        
        completion_pos = (arrival_pos[0] + speed * delay, arrival_pos[1])
        dist_at_completion = get_euclidean_distance(completion_pos, target_rsu_loc)
        
        # Deadline slack
        deadline_slack = deadline - delay
        # Coverage slack (time inside coverage after completion)
        coverage_time_slack = time_to_exit - delay
        
        # Explicit classification
        if not completed:
            if fail_coverage and not fail_deadline:
                classified_reason = "COVERAGE_EXIT"
            elif fail_deadline and not fail_coverage:
                classified_reason = "DEADLINE_MISS"
            elif fail_deadline and fail_coverage:
                classified_reason = "DUAL_VIOLATION"
            else:
                classified_reason = "OTHER"
        else:
            classified_reason = "COMPLETED"
            
        record = {
            "task_idx": task_idx,
            "task_id": task_id,
            "vehicle_id": v_id,
            "arrival_time_s": sim_time_arr,
            "speed_m_s": speed,
            "arrival_pos_x": arrival_pos[0],
            "arrival_pos_y": arrival_pos[1],
            "primary_rsu": target_rsu_id,
            "primary_rsu_x": target_rsu_loc[0],
            "dist_at_arrival_m": dist_arr,
            "time_to_rsu_exit_s": time_to_exit,
            "task_size_mb": task_size,
            "task_cpu_gcycles": task_cpu,
            "deadline_s": deadline,
            "v2r_delay_s": comm_delay,
            "compute_delay_s": comp_delay,
            "total_delay_s": delay,
            "completion_pos_x": completion_pos[0],
            "dist_at_completion_m": dist_at_completion,
            "rsu_comm_range_m": config.rsu_comm_range,
            "deadline_slack_s": deadline_slack,
            "coverage_slack_s": coverage_time_slack,
            "fail_deadline": fail_deadline,
            "fail_coverage": fail_coverage,
            "completed": completed,
            "failure_reason": fail_reason,
            "classified_reason": classified_reason
        }
        all_local_records.append(record)
        if not completed:
            failed_task_records.append(record)
        task_idx += 1
        
    env_local.close()
    
    df_failed = pd.DataFrame(failed_task_records)
    df_all = pd.DataFrame(all_local_records)
    
    print(f"\nTotal tasks: {len(df_all)} | Completed: {len(df_all) - len(df_failed)} | Failed: {len(df_failed)}")
    print("\n" + "=" * 75)
    print("   FAILED TASKS AUDIT TABLE")
    print("=" * 75)
    for idx, r in df_failed.iterrows():
        print(f"Task {r['task_id']} (Veh {r['vehicle_id']}):")
        print(f"  Arrival Pos: x={r['arrival_pos_x']:.1f}m | Speed: {r['speed_m_s']:.1f} m/s | RSU: {r['primary_rsu']} (x={r['primary_rsu_x']:.1f}m)")
        print(f"  Delay: {r['total_delay_s']:.3f}s (Comm: {r['v2r_delay_s']:.3f}s, Comp: {r['compute_delay_s']:.3f}s) | Deadline: {r['deadline_s']:.3f}s (Slack: {r['deadline_slack_s']:.3f}s)")
        print(f"  Time to RSU Exit: {r['time_to_rsu_exit_s']:.3f}s vs Delay: {r['total_delay_s']:.3f}s (Coverage Slack: {r['coverage_slack_s']:.3f}s)")
        print(f"  Pos at Completion: x={r['completion_pos_x']:.1f}m | Dist to RSU: {r['dist_at_completion_m']:.1f}m (Range: {r['rsu_comm_range_m']:.1f}m)")
        print(f"  Fail Deadline: {r['fail_deadline']} | Fail Coverage: {r['fail_coverage']} | Classification: {r['classified_reason']}")
        print("-" * 75)

    # Save failure trace
    out_dir = os.path.join(root_dir, "results", "remediation", "completion_failure_audit")
    os.makedirs(out_dir, exist_ok=True)
    df_failed.to_csv(os.path.join(out_dir, "failure_trace.csv"), index=False)
    print(f"Saved failure trace to {os.path.join(out_dir, 'failure_trace.csv')}")

if __name__ == "__main__":
    main()
