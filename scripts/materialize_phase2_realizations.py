"""
scripts/materialize_phase2_realizations.py

CLI tool to pre-materialize canonical Phase-2 evaluation realizations.
Persists complete task timestamps, task characteristics, vehicle trajectories,
mobility states, initial conditions, RSU configs, workload configs, seed, and geometry.
Generates cryptographic SHA-256 checksums for every realization.
"""

import os
import sys
import argparse
import json
import csv
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.realizations.generator import RealizationGenerator
from experiments.realizations.validator import RealizationValidator


def main():
    parser = argparse.ArgumentParser(description="Materialize Phase 2 Evaluation Realizations")
    parser.add_argument("--geometries", nargs="+", default=["corridor_2400m", "grid_200m"])
    parser.add_argument("--workloads", nargs="+", type=int, default=[20, 30, 40])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", type=str, default="data/evaluation_realizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generator = RealizationGenerator()

    manifest_records = []
    print("=" * 70)
    print("   STAGE 7: MATERIALIZING CONTROLLED EXPERIMENT REALIZATIONS   ")
    print(f"   Geometries: {args.geometries} | Workloads: {args.workloads} | Seeds: {args.seeds}")
    print(f"   Destination Directory: {args.output_dir}")
    print("=" * 70)

    for geom in args.geometries:
        for w in args.workloads:
            for s in args.seeds:
                realization = generator.generate_realization(
                    geometry=geom,
                    workload=w,
                    seed=s,
                    eval_seed_offset=30000,
                    num_vehicles=10
                )
                
                filename = f"{geom}_w{w}_seed{s}_realization.json"
                filepath = os.path.join(args.output_dir, filename)
                
                realization_hash = realization.save(filepath)
                
                # Immediate self-validation
                assert RealizationValidator.validate(realization, expected_geometry=geom, expected_workload=w, expected_seed=s)
                
                manifest_records.append({
                    "realization_id": realization.realization_id,
                    "geometry": geom,
                    "workload": w,
                    "seed": s,
                    "eval_seed": realization.eval_seed,
                    "total_tasks": len(realization.tasks),
                    "total_vehicles": len(realization.vehicle_trajectories),
                    "filename": filename,
                    "filepath": filepath,
                    "realization_hash": realization_hash,
                    "created_at": realization.created_at
                })
                print(f"[{geom} | w{w} | Seed {s}] -> {filename} (SHA-256: {realization_hash[:12]}...)")

    # Save manifest JSON and CSV index
    manifest_json_path = os.path.join(args.output_dir, "REALIZATION_MANIFEST.json")
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "STAGE 7 — CONTROLLED EXPERIMENT REALIZATION SYSTEM",
            "total_realizations": len(manifest_records),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "realizations": manifest_records
        }, f, indent=2)

    manifest_csv_path = os.path.join(args.output_dir, "REALIZATION_INDEX.csv")
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_records[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_records)

    print("-" * 70)
    print(f"[SUCCESS] Materialized {len(manifest_records)} realizations.")
    print(f"Manifest JSON: {manifest_json_path}")
    print(f"Manifest CSV:  {manifest_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
