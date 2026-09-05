#!/usr/bin/env python3
"""
scripts/materialize_training_realizations.py
Generates dedicated frozen training realizations strictly separated from the
60 canonical evaluation realizations.
"""

import os
import sys
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from utils.realization import generate_realization, save_realization

def materialize_training_realizations(output_dir: str = "data/training_realizations"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open("configs/paper_parameters.yaml", "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    configs = [
        ("corridor_2400m", 20, 101),
        ("corridor_2400m", 20, 102),
        ("corridor_2400m", 30, 101),
        ("corridor_2400m", 30, 102),
        ("corridor_2400m", 40, 101),
        ("grid_200m", 20, 101),
        ("grid_200m", 20, 102),
        ("grid_200m", 30, 101),
        ("grid_200m", 30, 102),
        ("grid_200m", 40, 101),
    ]
    
    port_base = 9200
    manifest = []
    
    for geom, wl, seed in configs:
        out_file = f"realization_{geom}_w{wl}_seed{seed}.json"
        out_path = os.path.join(output_dir, out_file)
        
        cfg_copy = dict(config_data)
        cfg_copy["num_tasks_per_vehicle_range"] = [wl, wl]
        sim_cfg = SimulationConfig(**cfg_copy)
        
        print(f"Materializing training trace: {geom} | w{wl} | seed {seed}...")
        env = VECEnv(
            config=sim_cfg,
            port=port_base,
            scenario_geometry=geom,
            seed=seed,
            max_vehicles=10
        )
        realization = generate_realization(env)
        env.close()
        
        save_realization(realization, out_path)
        manifest.append({
            "filename": out_file,
            "geometry": geom,
            "workload": wl,
            "seed": seed,
            "hash": realization["hash"]
        })
        port_base += 1
        
    print(f"[SUCCESS] Materialized {len(manifest)} dedicated training traces in '{output_dir}'.")
    return manifest

if __name__ == "__main__":
    materialize_training_realizations()
