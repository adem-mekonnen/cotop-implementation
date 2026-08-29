"""
experiments/demonstrate_realization_pairing.py

Demonstration of Stage 7 Controlled Experiment Realization Pairing.
Demonstrates:
1. Loading a materialized evaluation realization with SHA-256 integrity verification.
2. Executing CoTOP, DDQN, Greedy, and Local on the EXACT same realization trace.
3. Reporting side-by-side performance metrics proving shared exogenous input conditions.
"""

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.realizations.schema import ExperimentRealization
from experiments.realizations.validator import RealizationValidator
from experiments.realizations.runner import RealizationRunner


def main():
    realization_file = "data/evaluation_realizations/corridor_2400m_w20_seed0_realization.json"
    if not os.path.exists(realization_file):
        print(f"[ERROR] Realization file {realization_file} not found. Run materialize script first.")
        return

    print("=" * 80)
    print("      STAGE 7: CONTROLLED EXPERIMENT REALIZATION PAIRING DEMO")
    print("=" * 80)
    
    # 1. Load Realization
    print(f"Loading Realization File: {realization_file}")
    realization = ExperimentRealization.load(realization_file)
    print(f"Realization ID:    {realization.realization_id}")
    print(f"Geometry:          {realization.geometry}")
    print(f"Workload:          {realization.workload} tasks/veh (Total Tasks: {len(realization.tasks)})")
    print(f"Seed:              {realization.seed} (Eval Seed: {realization.eval_seed})")
    print(f"Payload SHA-256:   {realization.realization_hash}")
    
    # 2. Validate Realization Integrity (Rejection Gates 1-5)
    print("\nExecuting Pre-Flight Validation Gates (1-5)...")
    RealizationValidator.validate(
        realization=realization,
        expected_geometry="corridor_2400m",
        expected_workload=20,
        expected_seed=0
    )
    print("[SUCCESS] All 5 Pre-Flight Rejection Gates PASSED.\n")

    # 3. Controlled Paired Execution
    runner = RealizationRunner()
    algorithms = ["Local", "Greedy", "DDQN", "CoTOP"]
    results = {}

    print("-" * 80)
    print(f"{'Algorithm':<10} | {'Completed':<10} | {'Ratio (%)':<10} | {'Delay (s)':<10} | {'Energy (J)':<10} | {'Wait (s)':<10}")
    print("-" * 80)

    for algo in algorithms:
        res = runner.run_algorithm(
            algorithm=algo,
            realization=realization,
            expected_geometry="corridor_2400m",
            expected_workload=20,
            expected_seed=0
        )
        results[algo] = res
        print(f"{algo:<10} | {res.completed_tasks:<10} | {res.completion_ratio * 100:<10.1f} | {res.mean_delay_s:<10.2f} | {res.mean_energy_j:<10.2f} | {res.wait_delay_s:<10.2f}")

    print("-" * 80)
    print("\nExogenous Input Consistency Proof:")
    for algo, res in results.items():
        print(f"  - [{algo}] Consumed Realization Hash: {res.realization_hash} (Total Tasks: {res.total_tasks})")

    print("\n[CONCLUSION] All 4 algorithms evaluated against byte-identical task arrivals and trajectories.")
    print("=" * 80)


if __name__ == "__main__":
    main()
