#!/usr/bin/env python3
"""
scripts/inspect_forensic_snapshot.py
Inspect and report all checkpoints, manifests, logs, result dirs, and figures on disk.
"""

import os
import sys
import glob
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    print("=" * 70)
    print("   FORENSIC SNAPSHOT AUDIT OF LOCAL REPOSITORY ASSETS")
    print("=" * 70)

    # 1. Search for all checkpoint files on disk
    print("\n1. SEARCHING FOR CHECKPOINTS ON DISK (*.pt, *.pth, *.ckpt, *.tar):")
    ckpt_patterns = ["**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.tar"]
    found_ckpts = []
    for pat in ckpt_patterns:
        found_ckpts.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))
    
    # Filter out virtual environments or .git
    filtered_ckpts = [f for f in found_ckpts if ".venv" not in f and ".git" not in f]
    print(f"   Found {len(filtered_ckpts)} checkpoint file(s) on disk:")
    for c in filtered_ckpts:
        rel = os.path.relpath(c, root_dir)
        sz = os.path.getsize(c)
        print(f"   - {rel} ({sz} bytes)")
    if not filtered_ckpts:
        print("   [NOTE] No .pt / .pth checkpoint files found on local disk.")

    # 2. Search for experiment manifests
    print("\n2. SEARCHING FOR EXPERIMENT MANIFESTS (*manifest*.json):")
    manifest_files = glob.glob(os.path.join(root_dir, "**/*manifest*.json"), recursive=True)
    filtered_manifests = [f for f in manifest_files if ".venv" not in f and ".git" not in f]
    print(f"   Found {len(filtered_manifests)} manifest file(s):")
    for m in sorted(filtered_manifests):
        rel = os.path.relpath(m, root_dir)
        print(f"   - {rel}")

    # 3. Search for training logs and curves (*training_curve.csv, *metrics.json, *.log)
    print("\n3. SEARCHING FOR TRAINING LOGS & CURVES:")
    curve_files = glob.glob(os.path.join(root_dir, "results/**/training_curve.csv"), recursive=True)
    metric_files = glob.glob(os.path.join(root_dir, "results/**/metrics.json"), recursive=True)
    print(f"   Found {len(curve_files)} training_curve.csv files and {len(metric_files)} metrics.json files across results/")
    for c in sorted(curve_files)[:10]:
        print(f"   - Curve: {os.path.relpath(c, root_dir)}")
    if len(curve_files) > 10:
        print(f"     ... and {len(curve_files) - 10} more training curves.")

    # 4. Result directories inventory
    print("\n4. RESULT DIRECTORIES INVENTORY (results/):")
    results_dir = os.path.join(root_dir, "results")
    if os.path.exists(results_dir):
        subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        for d in sorted(subdirs):
            p = os.path.join(results_dir, d)
            n_files = sum([len(files) for r, dirs, files in os.walk(p)])
            print(f"   - results/{d} ({n_files} files)")

    # 5. Publication figures inventory
    print("\n5. PUBLICATION FIGURES INVENTORY (publication_figures/):")
    figs_dir = os.path.join(root_dir, "publication_figures")
    if os.path.exists(figs_dir):
        figs = [f for f in os.listdir(figs_dir) if f.endswith(".png") or f.endswith(".pdf")]
        for f in sorted(figs):
            sz = os.path.getsize(os.path.join(figs_dir, f))
            print(f"   - {f} ({sz} bytes)")

    # 6. Current campaign manifests summary
    print("\n6. CURRENT CAMPAIGN MANIFESTS DETAILS:")
    for mf in ["final_experiment_manifest.json", "results/final/campaign_manifest.json", "results/final_gpu_campaign/campaign_manifest.json", "results/phase2_step21/provenance_manifest.json", "results/phase2_step20/campaign_manifest.json"]:
        p = os.path.join(root_dir, mf)
        if os.path.exists(p):
            try:
                data = json.load(open(p))
                print(f"   [{mf}]: ID={data.get('campaign_id')}, Commit={data.get('git_commit_sha', data.get('git_sha'))}, Status={data.get('status')}")
            except Exception as e:
                print(f"   [{mf}]: Error reading JSON: {e}")

    print("\n" + "=" * 70)
    print("   FORENSIC SNAPSHOT AUDIT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
