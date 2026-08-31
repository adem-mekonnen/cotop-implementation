import json
import hashlib
import time
import os
import copy
from typing import Dict, Any

def get_git_sha():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"

def generate_realization(env_instance) -> Dict[str, Any]:
    """
    Generates a canonical realization trace by advancing the environment's SUMO instance
    through the entire max_sim_time and recording all exogenous vehicle positions and generated tasks.
    """
    realization_id = f"realization_{env_instance.scenario_geometry}_{env_instance.seed}"
    
    trace = {}
    generated_tasks = {}
    
    # We must ensure sumo is started
    if not env_instance.sumo_started:
        env_instance.sumo_manager.start_simulation()
        env_instance.sumo_started = True
    else:
        env_instance.sumo_manager.reload_simulation()

    # Reset internal tracking manually so we capture from time 0
    env_instance.sim_time = 0.0
    env_instance.active_vehicles = {}
    env_instance.vehicle_tasks = {}
    env_instance.generated_vehicle_ids = set()
    env_instance.pending_tasks = []
    
    # Capture trace for max_sim_time
    max_steps = int(env_instance.max_sim_time)
    for step in range(max_steps + 1):
        env_instance.sumo_manager.step()
        vehicles_data = env_instance.sumo_manager.get_vehicle_data()
        
        # Save step data
        step_data = {}
        for v_id, v_data in vehicles_data.items():
            step_data[v_id] = {
                "pos": [float(v_data['pos'][0]), float(v_data['pos'][1])],
                "speed": float(v_data['speed'])
            }
            
            # If new vehicle, generate tasks and record
            if v_id not in env_instance.active_vehicles:
                if len(env_instance.generated_vehicle_ids) < env_instance.max_vehicles:
                    env_instance.generated_vehicle_ids.add(v_id)
                    tasks = env_instance.task_gen.generate_tasks_for_vehicle(v_id)
                    task_dicts = []
                    for t in tasks:
                        task_dicts.append({
                            "task_id": t.task_id,
                            "size_rho": float(t.size_rho),
                            "cpu_phi": float(t.cpu_phi),
                            "max_delay_d": float(t.max_delay_d)
                        })
                    generated_tasks[v_id] = task_dicts
                    env_instance.active_vehicles[v_id] = True # Mark as seen
                    
        trace[str(int(env_instance.sim_time))] = step_data
        env_instance.sim_time += 1.0

    # Build the realization object
    config_dict = env_instance.config.__dict__.copy()
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode('utf-8')).hexdigest()
    
    realization = {
        "realization_id": realization_id,
        "geometry": env_instance.scenario_geometry,
        "workload": "I_" + str(env_instance.config.num_tasks_per_vehicle_range[0]),
        "seed": env_instance.seed,
        "config_hash": config_hash,
        "git_sha": get_git_sha(),
        "creation_timestamp": str(time.time()),
        "schema_version": "1.0",
        "vehicle_trace": trace,
        "task_trace": generated_tasks
    }
    
    return realization

def save_realization(realization: Dict[str, Any], output_path: str):
    """Saves the realization with an integrated SHA-256 hash to prevent tampering."""
    # Ensure no pre-existing hash is included in the computation
    data_to_hash = copy.deepcopy(realization)
    if "hash" in data_to_hash:
        del data_to_hash["hash"]
        
    json_str = json.dumps(data_to_hash, sort_keys=True)
    realization_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    realization["hash"] = realization_hash
    
    with open(output_path, 'w') as f:
        json.dump(realization, f, indent=2)

def load_realization(filepath: str) -> Dict[str, Any]:
    """Loads a realization and strictly verifies its SHA-256 hash."""
    with open(filepath, 'r') as f:
        realization = json.load(f)
        
    if "hash" not in realization:
        raise ValueError("Realization file is missing a cryptographic hash.")
        
    expected_hash = realization["hash"]
    
    data_to_hash = copy.deepcopy(realization)
    del data_to_hash["hash"]
    json_str = json.dumps(data_to_hash, sort_keys=True)
    actual_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    if expected_hash != actual_hash:
        raise ValueError(f"Realization hash mismatch! The file has been tampered with or corrupted. Expected {expected_hash}, got {actual_hash}")
        
    return realization
