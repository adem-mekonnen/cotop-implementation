import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import torch

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy
from utils.seed import set_seed


def run_smoke_evaluation(vehicle_counts=[2, 5, 10, 30], seed=42, output_csv="results/multivehicle_contention/smoke_experiment_results.csv"):
    set_seed(seed)
    
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = SimulationConfig(**cfg_dict)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # Load CoTOP model if checkpoint exists
    cotop_model = None
    ckpt_path = "results/checkpoints/a3c_agent.pth"
    if os.path.exists(ckpt_path):
        try:
            cotop_model = ActorCritic(114, 7)
            cotop_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            cotop_model.eval()
            print(f"[INFO] Loaded CoTOP model from {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint ({e}), evaluating CoTOP with initialized policy")
            cotop_model = None

    results = []
    policies = [("local", LocalPolicy(config)), ("greedy", GreedyPolicy(config))]
    if cotop_model is not None:
        policies.insert(0, ("cotop", cotop_model))

    print("=" * 85)
    print("      CoTOP MULTI-VEHICLE CONTENTION & SCALABILITY SMOKE EXPERIMENT")
    print("=" * 85)
    print(f"Vehicle Scales: {vehicle_counts} | Seed: {seed}")
    print("-" * 85)

    base_port = 8900
    for n_veh in vehicle_counts:
        for pol_name, pol_obj in policies:
            port = base_port
            base_port += 1
            
            env = VECEnv(config=config, port=port, seed=seed, max_vehicles=n_veh)
            obs, _ = env.reset(seed=seed, options={"max_vehicles": n_veh})
            
            delays = []
            comm_delays = []
            comp_delays = []
            wait_delays = []
            energies = []
            queue_snapshots = []
            completed_tasks = 0
            total_tasks = 0
            active_veh_set = set()
            
            done = False
            while not done:
                if pol_name == "cotop":
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = pol_obj(obs_t)
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    action = pol_obj.select_action(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_tasks += 1
                delays.append(info.get("delay", 0.0))
                comm_delays.append(info.get("comm_delay", 0.0))
                comp_delays.append(info.get("comp_delay", 0.0))
                wait_delays.append(info.get("wait_delay", 0.0))
                energies.append(info.get("energy", 0.0))
                if "rsu_queues" in info:
                    queue_snapshots.append(info["rsu_queues"])
                if info.get("completed", False):
                    completed_tasks += 1
                if "v_id" in info:
                    active_veh_set.add(info["v_id"])

            env.close()

            # Compute statistics
            avg_delay = float(np.mean(delays)) if delays else 0.0
            avg_comm = float(np.mean(comm_delays)) if comm_delays else 0.0
            avg_comp = float(np.mean(comp_delays)) if comp_delays else 0.0
            avg_wait = float(np.mean(wait_delays)) if wait_delays else 0.0
            avg_energy = float(np.mean(energies)) if energies else 0.0
            comp_ratio = (completed_tasks / max(total_tasks, 1)) * 100.0

            # Queue statistics (in Mcycles)
            if queue_snapshots:
                q_arr = np.array(queue_snapshots)
                avg_q_mcycles = float(np.mean(q_arr) / 1.0e6)
                max_q_mcycles = float(np.max(q_arr) / 1.0e6)
            else:
                avg_q_mcycles = 0.0
                max_q_mcycles = 0.0

            res = {
                "n_vehicles_target": n_veh,
                "n_vehicles_observed": len(active_veh_set),
                "policy": pol_name,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "completion_rate_pct": round(comp_ratio, 2),
                "avg_total_delay_s": round(avg_delay, 4),
                "avg_comm_delay_s": round(avg_comm, 4),
                "avg_comp_delay_s": round(avg_comp, 4),
                "avg_wait_delay_s": round(avg_wait, 4),
                "avg_energy_J": round(avg_energy, 4),
                "avg_rsu_queue_Mcycles": round(avg_q_mcycles, 2),
                "max_rsu_queue_Mcycles": round(max_q_mcycles, 2),
            }
            results.append(res)

            print(f"[N={n_veh:02d} | {pol_name.upper():6s}] Tasks: {total_tasks:3d} | Comp: {comp_ratio:5.1f}% | Delay: {avg_delay:6.3f}s (Wait: {avg_wait:6.3f}s) | Energy: {avg_energy:6.3f}J | MaxQ: {max_q_mcycles:6.1f}Mcyc")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("=" * 85)
    print(f"[SUCCESS] Smoke experiment results saved to: {output_csv}")
    print("=" * 85)
    return df


if __name__ == "__main__":
    run_smoke_evaluation()
