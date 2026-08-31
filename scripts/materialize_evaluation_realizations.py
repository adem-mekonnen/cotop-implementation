import os
import argparse
import yaml
from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from utils.realization import generate_realization, save_realization

def main():
    parser = argparse.ArgumentParser(description="Materialize Frozen Evaluation Realizations")
    parser.add_argument("--output_dir", type=str, default="data/evaluation_realizations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    parser.add_argument("--geometries", type=str, nargs="+", default=["corridor_2400m", "grid_200m"])
    parser.add_argument("--workloads", type=int, nargs="+", default=[20, 30, 40])
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    
    port_base = 9990
    
    for geom in args.geometries:
        for workload in args.workloads:
            for seed in args.seeds:
                print(f"Materializing realization for Geometry: {geom}, Workload: w{workload}, Seed: {seed}")
                
                config_data["num_tasks_per_vehicle_range"] = [workload, workload]
                config = SimulationConfig(**config_data)
                
                env = VECEnv(
                    config=config,
                    port=port_base,
                    scenario_geometry=geom,
                    seed=seed,
                    max_vehicles=10
                )
                
                realization = generate_realization(env)
                env.close()
                
                output_path = os.path.join(args.output_dir, f"realization_{geom}_w{workload}_{seed}.json")
                save_realization(realization, output_path)
                print(f"  -> Saved to {output_path} with hash: {realization['hash']}")
                
                port_base += 1

if __name__ == "__main__":
    main()
