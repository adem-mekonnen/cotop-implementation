#!/usr/bin/env python3
"""
scripts/audit_local_distributions.py
Computes distribution metrics (mean, median, p50, p95, max) for Local execution.
"""

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from envs.vec_env import get_euclidean_distance

def main():
    scenario = "corridor_2400m"
    workload = 20
    seed = 42
    
    with open(os.path.join(root_dir, "configs/paper_parameters.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["num_tasks_per_vehicle_range"] = [workload, workload]
    config = SimulationConfig(**cfg_dict)
    
    realization_path = os.path.join(root_dir, "data", "evaluation_realizations", f"realization_{scenario}_w{workload}_seed{seed}.json")
    env = FrozenVECEnv(config=config, realization_path=realization_path)
    env.reset()
    
    records = []
    while len(env.pending_tasks) > 0:
        curr_veh, curr_task = env.pending_tasks[0]
        obs, reward, terminated, truncated, info = env.step(0)
        records.append({
            "task_id": curr_task.task_id,
            "size_rho_mb": curr_task.size_rho,
            "cpu_phi_gcycles": curr_task.cpu_phi,
            "deadline_d_s": curr_task.max_delay_d,
            "v2r_delay_s": info["comm_delay"],
            "comp_delay_s": info["comp_delay"],
            "wait_delay_s": info["wait_delay"],
            "total_delay_s": info["delay"],
            "energy_j": info["energy"],
            "deadline_slack_s": curr_task.max_delay_d - info["delay"],
            "completed": info["completed"],
            "fail_deadline": info["fail_deadline"],
            "fail_coverage": info["fail_coverage"],
            "failure_reason": info["failure_reason"]
        })
    env.close()
    
    df = pd.DataFrame(records)
    
    metrics = [
        ("cpu_phi_gcycles", "Task CPU (Gcycles)"),
        ("v2r_delay_s", "V2R Transmission Delay (s)"),
        ("comp_delay_s", "Computation Delay (s)"),
        ("wait_delay_s", "Queue Wait Delay (s)"),
        ("total_delay_s", "Total Delay (s)"),
        ("deadline_d_s", "Deadline (s)"),
        ("deadline_slack_s", "Deadline Slack (s)"),
        ("energy_j", "Energy Consumption (J)")
    ]
    
    print("=" * 80)
    print("   LOCAL EXECUTION METRIC DISTRIBUTIONS (200 Tasks)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Mean':<8} | {'Median':<8} | {'P50':<8} | {'P95':<8} | {'Max':<8}")
    print("-" * 80)
    
    stats_dict = {}
    for col, name in metrics:
        vals = df[col].values
        mean_val = float(np.mean(vals))
        med_val = float(np.median(vals))
        p50_val = float(np.percentile(vals, 50))
        p95_val = float(np.percentile(vals, 95))
        max_val = float(np.max(vals))
        print(f"{name:<30} | {mean_val:8.4f} | {med_val:8.4f} | {p50_val:8.4f} | {p95_val:8.4f} | {max_val:8.4f}")
        stats_dict[col] = {
            "mean": mean_val, "median": med_val, "p50": p50_val, "p95": p95_val, "max": max_val
        }
        
    print("-" * 80)
    total_tasks = len(df)
    completed_tasks = df["completed"].sum()
    failed_tasks = total_tasks - completed_tasks
    comp_ratio = completed_tasks / total_tasks
    fail_ratio = failed_tasks / total_tasks
    print(f"Total Tasks: {total_tasks} | Completed: {completed_tasks} ({comp_ratio*100:.2f}%) | Failed: {failed_tasks} ({fail_ratio*100:.2f}%)")
    print(f"Coverage Failures: {df['fail_coverage'].sum()} | Deadline Failures: {df['fail_deadline'].sum()}")
    print("=" * 80)
    
    # Save to completion_summary.json
    out_dir = os.path.join(root_dir, "results", "remediation", "completion_failure_audit")
    os.makedirs(out_dir, exist_ok=True)
    
    summary_data = {
        "scenario": scenario,
        "workload": workload,
        "seed": seed,
        "total_tasks": total_tasks,
        "completed_tasks": int(completed_tasks),
        "failed_tasks": int(failed_tasks),
        "completion_ratio": float(comp_ratio),
        "failure_ratio": float(fail_ratio),
        "failure_reason_counts": {
            "COVERAGE_EXIT": int(df["fail_coverage"].sum()),
            "DEADLINE_MISS": int(df["fail_deadline"].sum()),
            "OTHER": 0
        },
        "distribution_statistics": stats_dict
    }
    with open(os.path.join(out_dir, "completion_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved completion summary to {os.path.join(out_dir, 'completion_summary.json')}")

if __name__ == "__main__":
    main()
