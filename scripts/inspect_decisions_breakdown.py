#!/usr/bin/env python3
"""
scripts/inspect_decisions_breakdown.py
Inspect decisions/actions across CoTOP, DDQN, Greedy, and Local evaluation CSVs.
"""

import os
import glob
import pandas as pd
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    csvs = glob.glob(os.path.join(root_dir, "results", "**", "evaluation_results.csv"), recursive=True)
    
    records = []
    for c in csvs:
        rel = os.path.relpath(c, root_dir)
        parts = rel.split(os.sep)
        df = pd.read_csv(c)
        col = "decision" if "decision" in df.columns else ("action" if "action" in df.columns else None)
        if col:
            val_counts = df[col].value_counts().to_dict()
            records.append({
                "path": rel,
                "total_tasks": len(df),
                "decisions": val_counts
            })
    
    print(f"Total evaluated CSVs with decisions: {len(records)}")
    
    # Check decision distributions
    for r in records[:15]:
        print(f"\n{r['path']}: {r['decisions']}")

if __name__ == "__main__":
    main()
