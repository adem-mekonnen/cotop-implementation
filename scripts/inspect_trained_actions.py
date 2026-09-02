#!/usr/bin/env python3
"""
scripts/inspect_trained_actions.py
Inspect action distributions across existing evaluation runs.
"""

import os
import glob
import pandas as pd
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    print("=== SEARCHING FOR ACTION LOGS IN RESULTS ===")
    csvs = glob.glob(os.path.join(root_dir, "results", "**", "evaluation_results.csv"), recursive=True)
    print(f"Found {len(csvs)} evaluation_results.csv files.")
    
    # Check if there are evaluation metrics or run manifests with action info
    sample_csvs = [c for c in csvs if "final" in c or "phase2" in c][:10]
    for sc in sample_csvs:
        rel = os.path.relpath(sc, root_dir)
        df = pd.read_csv(sc)
        print(f"\nFile: {rel}")
        print(f"Columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        if "action" in df.columns:
            vc = df["action"].value_counts().to_dict()
            print(f"Action value counts: {vc}")

if __name__ == "__main__":
    main()
