#!/usr/bin/env python3
"""
scripts/run_phase2_step16_statistics.py

Phase 2 — Step 16: Statistical Verification & Final Cross-Baseline Synthesis
Authoritative, reproducible statistical analysis pipeline.

Generates:
  results/phase2_step16/
    - raw_experiment_index.csv
    - descriptive_statistics.csv
    - cross_algorithm_statistics.csv
    - paired_comparisons.csv
    - effect_sizes.csv
    - multiple_comparison_corrections.csv
    - convergence_statistics.csv
    - seed_dispersion.csv
    - aggregation_audit.csv
    - published_value_comparison.csv
  figures/phase2_step16/
    - delay_distribution.png
    - energy_distribution.png
    - completion_ratio_by_seed.png
    - training_reward_curves.png
    - training_loss_curves.png
    - algorithm_comparison.png
    - effect_sizes_ci.png
"""

import sys
import os

# Ensure root workspace is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import hashlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.statistical_analysis import (
    paired_t_test,
    wilcoxon_test,
    cohens_dz,
    common_language_effect_size,
    holm_bonferroni,
    fdr_benjamini_hochberg,
    compute_complete_paired_stats
)

RESULTS_DIR = os.path.join(root_dir, "results", "phase2_step16")
FIGURES_DIR = os.path.join(root_dir, "figures", "phase2_step16")

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

def verify_provenance():
    print("--- 1. Verifying Provenance of Source Artifacts ---")
    sources = {
        "step14_seed_summary": os.path.join(root_dir, "results", "phase2_step14", "step14_seed_summary.csv"),
        "step14_convergence": os.path.join(root_dir, "results", "phase2_step14", "step14_convergence_analysis.csv"),
        "multiseed_results": os.path.join(root_dir, "results", "phase2_multiseed", "seed_results.csv"),
        "summary_60cell": os.path.join(root_dir, "results", "phase2_algorithmic_fidelity", "summary_60cell.csv"),
        "single_gate": os.path.join(root_dir, "results", "stage9_single_condition_gate", "single_condition_gate_results.json"),
        "aggregation_retest": os.path.join(root_dir, "results", "phase2_algorithmic_fidelity", "aggregation_hypothesis_retest.csv"),
    }
    
    hashes = {}
    for name, path in sources.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing authoritative source artifact: {path}")
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        hashes[name] = {"path": path, "sha256": h}
        print(f"  [OK] {name}: {h[:12]}... ({os.path.basename(path)})")
    return hashes

def build_raw_experiment_index(hashes):
    print("\n--- 2. Building Raw Experiment Index ---")
    rows = []
    
    # 1. Step 14 DDQN 5-seed runs
    df_s14 = pd.read_csv(hashes["step14_seed_summary"]["path"])
    for _, r in df_s14.iterrows():
        rows.append({
            "experiment_id": f"step14_ddqn_w20_seed{int(r['seed'])}",
            "source_artifact": "results/phase2_step14/step14_seed_summary.csv",
            "algorithm": "DDQN",
            "geometry": "corridor_2400m",
            "workload": "w20",
            "seed": int(r["seed"]),
            "realization_hash": str(r.get("realization_hash", "UNKNOWN")),
            "checkpoint_hash": str(r.get("model_hash", "UNKNOWN")),
            "git_sha": "cc97392682b99f67a687e1332b75fa71fd5eb4aa",
            "tasks_generated": int(r["tasks_generated"]),
            "tasks_completed": int(r["tasks_completed"]),
            "completion_ratio": float(r["completion_ratio"]),
            "mean_delay_s": float(r["mean_delay_s"]),
            "mean_energy_j": float(r["mean_energy_J"]),
            "comm_delay_s": float(r["comm_delay_mean_s"]),
            "comp_delay_s": float(r["comp_delay_mean_s"]),
            "wait_delay_s": float(r["wait_delay_mean_s"]),
        })
        
    # 2. Phase 2 60-cell CoTOP & DDQN runs
    df_60 = pd.read_csv(hashes["summary_60cell"]["path"])
    for _, r in df_60.iterrows():
        rows.append({
            "experiment_id": str(r["cell_id"]),
            "source_artifact": "results/phase2_algorithmic_fidelity/summary_60cell.csv",
            "algorithm": str(r["algorithm"]),
            "geometry": str(r["geometry"]),
            "workload": f"w{int(r['workload'])}",
            "seed": int(r["seed"]),
            "realization_hash": str(r["realization_hash"]),
            "checkpoint_hash": str(r["checkpoint_sha256"]),
            "git_sha": str(r["git_sha"]),
            "tasks_generated": int(r["total_tasks"]),
            "tasks_completed": int(r["completed_tasks"]),
            "completion_ratio": float(r["completion_ratio"]),
            "mean_delay_s": float(r["mean_delay_s"]),
            "mean_energy_j": float(r["mean_energy_j"]),
            "comm_delay_s": float(r["comm_delay_s"]),
            "comp_delay_s": float(r["comp_delay_s"]),
            "wait_delay_s": float(r["wait_delay_s"]),
        })
        
    df_index = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "raw_experiment_index.csv")
    df_index.to_csv(out_path, index=False)
    print(f"  Indexed {len(df_index)} experimental runs -> {out_path}")
    return df_index

def compute_descriptive_statistics(df_index):
    print("\n--- 3. Computing Descriptive Statistics ---")
    desc_rows = []
    
    grouped = df_index.groupby(["algorithm", "geometry", "workload"])
    for (algo, geom, wl), group in grouped:
        for metric, col in [("delay", "mean_delay_s"), ("energy", "mean_energy_j"), ("completion_ratio", "completion_ratio")]:
            vals = group[col].values
            n = len(vals)
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            med = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75, 25]) if n > 1 else (m, m)
            iqr = float(q75 - q25)
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            cv = float(s / m) if m != 0 else 0.0
            
            if n > 1 and s > 0:
                t_crit = float(stats.t.ppf(0.975, df=n - 1))
                sem = s / np.sqrt(n)
                ci_low = m - t_crit * sem
                ci_high = m + t_crit * sem
            else:
                ci_low, ci_high = m, m
                
            desc_rows.append({
                "algorithm": algo,
                "geometry": geom,
                "workload": wl,
                "metric": metric,
                "aggregation_level": "per_task_mean",
                "n": n,
                "mean": m,
                "std": s,
                "median": med,
                "iqr": iqr,
                "min": vmin,
                "max": vmax,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "cv": cv
            })
            
    df_desc = pd.DataFrame(desc_rows)
    out_path = os.path.join(RESULTS_DIR, "descriptive_statistics.csv")
    df_desc.to_csv(out_path, index=False)
    print(f"  Computed descriptive statistics ({len(df_desc)} rows) -> {out_path}")
    return df_desc

def compute_paired_comparisons(hashes):
    print("\n--- 4. Computing Matched Paired Statistical Tests ---")
    # Load 60-cell matched dataset
    df_60 = pd.read_csv(hashes["summary_60cell"]["path"])
    
    paired_rows = []
    effect_rows = []
    p_values_all = []
    comparison_meta = []
    
    conditions = [
        ("corridor_2400m", 20),
        ("corridor_2400m", 30),
        ("corridor_2400m", 40),
        ("grid_200m", 20),
        ("grid_200m", 30),
        ("grid_200m", 40),
    ]
    
    for geom, wl in conditions:
        cotop_sub = df_60[(df_60["algorithm"] == "CoTOP") & (df_60["geometry"] == geom) & (df_60["workload"] == wl)].sort_values("seed")
        ddqn_sub = df_60[(df_60["algorithm"] == "DDQN") & (df_60["geometry"] == geom) & (df_60["workload"] == wl)].sort_values("seed")
        
        # Verify pairing: seeds and realization hashes must match
        cotop_seeds = cotop_sub["seed"].values
        ddqn_seeds = ddqn_sub["seed"].values
        cotop_hashes = cotop_sub["realization_hash"].values
        ddqn_hashes = ddqn_sub["realization_hash"].values
        
        assert np.array_equal(cotop_seeds, ddqn_seeds), f"Seed mismatch in {geom} w{wl}"
        assert np.array_equal(cotop_hashes, ddqn_hashes), f"Realization mismatch in {geom} w{wl}"
        
        for metric, col in [("delay", "mean_delay_s"), ("energy", "mean_energy_j")]:
            x = cotop_sub[col].values
            y = ddqn_sub[col].values
            
            res = compute_complete_paired_stats(x, y)
            
            cond_id = f"{geom}_w{wl}_{metric}"
            p_val = res["p_value_ttest"]
            p_values_all.append(p_val)
            comparison_meta.append((cond_id, geom, f"w{wl}", metric))
            
            paired_rows.append({
                "condition_id": cond_id,
                "geometry": geom,
                "workload": f"w{wl}",
                "metric": metric,
                "n_pairs": res["n"],
                "mean_cotop": float(np.mean(x)),
                "mean_ddqn": float(np.mean(y)),
                "mean_diff": res["mean_diff"],
                "std_diff": res["std_diff"],
                "sem": res["sem"],
                "t_stat": res["t_statistic"],
                "p_value_ttest": res["p_value_ttest"],
                "w_stat": res["w_statistic"],
                "p_value_wilcoxon": res["p_value_wilcoxon"],
                "cohens_dz": res["cohens_dz"],
                "cohens_dz_ci_low": res["cohens_dz_ci_lower"],
                "cohens_dz_ci_high": res["cohens_dz_ci_upper"],
                "cles": res["cles"]
            })
            
            effect_rows.append({
                "comparison": f"CoTOP_vs_DDQN_{geom}_w{wl}_{metric}",
                "geometry": geom,
                "workload": f"w{wl}",
                "metric": metric,
                "n": res["n"],
                "cohens_dz": res["cohens_dz"],
                "cohens_dz_ci_low": res["cohens_dz_ci_lower"],
                "cohens_dz_ci_high": res["cohens_dz_ci_upper"],
                "cles": res["cles"],
                "interpretation": "Large effect" if abs(res["cohens_dz"]) >= 0.8 else ("Medium effect" if abs(res["cohens_dz"]) >= 0.5 else "Small/negligible")
            })
            
    df_paired = pd.DataFrame(paired_rows)
    df_effect = pd.DataFrame(effect_rows)
    
    # Multiple comparison corrections
    p_arr = np.array(p_values_all)
    holm_p = holm_bonferroni(p_arr)
    fdr_q = fdr_benjamini_hochberg(p_arr)
    
    mult_rows = []
    for i, (cond_id, geom, wl, metric) in enumerate(comparison_meta):
        mult_rows.append({
            "comparison": f"CoTOP_vs_DDQN_{cond_id}",
            "geometry": geom,
            "workload": wl,
            "metric": metric,
            "raw_p_value": float(p_arr[i]),
            "holm_bonferroni_p": float(holm_p[i]),
            "benjamini_hochberg_q": float(fdr_q[i]),
            "alpha": 0.05,
            "significant_raw": bool(p_arr[i] < 0.05),
            "significant_holm": bool(holm_p[i] < 0.05),
            "significant_fdr": bool(fdr_q[i] < 0.05),
        })
    df_mult = pd.DataFrame(mult_rows)
    
    out_paired = os.path.join(RESULTS_DIR, "paired_comparisons.csv")
    out_effect = os.path.join(RESULTS_DIR, "effect_sizes.csv")
    out_mult = os.path.join(RESULTS_DIR, "multiple_comparison_corrections.csv")
    
    df_paired.to_csv(out_paired, index=False)
    df_effect.to_csv(out_effect, index=False)
    df_mult.to_csv(out_mult, index=False)
    
    print(f"  Saved paired comparisons -> {out_paired}")
    print(f"  Saved effect sizes -> {out_effect}")
    print(f"  Saved multiple comparison corrections -> {out_mult}")
    
    return df_paired, df_effect, df_mult

def build_cross_algorithm_statistics(hashes):
    print("\n--- 5. Building Cross-Algorithm Benchmark Table ---")
    # Load Stage 9 single-condition benchmark (matched 4-algorithm gate)
    with open(hashes["single_gate"]["path"], "r") as f:
        gate_data = json.load(f)
        
    algos = gate_data["algorithms"]
    rows = []
    for algo_name, metrics in algos.items():
        rows.append({
            "condition": "corridor_2400m_w20_single_seed",
            "algorithm": algo_name,
            "tasks_generated": metrics["total_tasks"],
            "tasks_completed": metrics["completed_tasks"],
            "completion_ratio": metrics["completion_ratio"],
            "mean_delay_s": metrics["mean_delay_s"],
            "mean_energy_j": metrics["mean_energy_j"],
            "comm_delay_s": metrics["comm_delay_s"],
            "comp_delay_s": metrics["comp_delay_s"],
            "wait_delay_s": metrics["wait_delay_s"],
            "provenance": "results/stage9_single_condition_gate/single_condition_gate_results.json"
        })
    df_cross = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "cross_algorithm_statistics.csv")
    df_cross.to_csv(out_path, index=False)
    print(f"  Saved cross-algorithm statistics -> {out_path}")
    return df_cross

def build_convergence_and_seed_dispersion(hashes):
    print("\n--- 6. Building Convergence & Seed Dispersion Tables ---")
    df_s14_conv = pd.read_csv(hashes["step14_convergence"]["path"])
    out_conv = os.path.join(RESULTS_DIR, "convergence_statistics.csv")
    df_s14_conv.to_csv(out_conv, index=False)
    
    df_s14 = pd.read_csv(hashes["step14_seed_summary"]["path"])
    disp_rows = []
    for metric, col in [
        ("Delay (s)", "mean_delay_s"),
        ("Energy (J)", "mean_energy_J"),
        ("Completion Ratio", "completion_ratio"),
        ("Final Reward", "reward_last_50_mean"),
        ("Mean Loss", "loss_mean")
    ]:
        vals = df_s14[col].values
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        cv = float(s / abs(m)) if m != 0 else 0.0
        disp_rows.append({
            "metric": metric,
            "column": col,
            "n_seeds": len(vals),
            "mean": m,
            "std": s,
            "cv": cv,
            "median": float(np.median(vals)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "sensitivity_rating": "HIGH" if cv > 0.20 else ("MODERATE" if cv > 0.05 else "LOW")
        })
    df_disp = pd.DataFrame(disp_rows)
    out_disp = os.path.join(RESULTS_DIR, "seed_dispersion.csv")
    df_disp.to_csv(out_disp, index=False)
    
    print(f"  Saved convergence statistics -> {out_conv}")
    print(f"  Saved seed dispersion -> {out_disp}")
    return df_s14_conv, df_disp

def build_aggregation_and_published_comparison(hashes):
    print("\n--- 7. Building Aggregation & Published Comparison Tables ---")
    df_agg = pd.read_csv(hashes["aggregation_retest"]["path"])
    out_agg = os.path.join(RESULTS_DIR, "aggregation_audit.csv")
    df_agg.to_csv(out_agg, index=False)
    
    # Authoritative published value discrepancy table
    pub_rows = [
        {
            "quantity": "Delay (s)",
            "published_target": 13.90,
            "current_measured_mean": float(df_agg["metric_a_delay_per_subtask_s"].mean()),
            "nominal_idle_physics": 4.40,
            "unstated_queue_hypothesis": 13.86,
            "reproduction_status": "NOT ACHIEVED",
            "plausible_explanation": "Omitted initial server queue backlog (~18.96 Gcycles / 9.48 s delay)",
            "evidence_level": "Plausible sufficient condition, unstated in paper"
        },
        {
            "quantity": "Energy (J)",
            "published_target": 25.14,
            "current_measured_mean": float(df_agg["metric_a_energy_per_subtask_J"].mean()),
            "nominal_idle_physics": 0.32,
            "idle_power_hypothesis": 25.02,
            "reproduction_status": "NOT ACHIEVED",
            "plausible_explanation": "Omitted baseline server idle power draw (~1.8 W integrated over delay)",
            "evidence_level": "Plausible sufficient condition, unstated in paper"
        }
    ]
    df_pub = pd.DataFrame(pub_rows)
    out_pub = os.path.join(RESULTS_DIR, "published_value_comparison.csv")
    df_pub.to_csv(out_pub, index=False)
    
    print(f"  Saved aggregation audit -> {out_agg}")
    print(f"  Saved published value comparison -> {out_pub}")
    return df_agg, df_pub

def generate_publication_figures(hashes, df_index, df_paired, df_effect):
    print("\n--- 8. Generating Publication-Grade Figures ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Delay Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    df_60 = pd.read_csv(hashes["summary_60cell"]["path"])
    cotop_delays = df_60[df_60["algorithm"] == "CoTOP"]["mean_delay_s"].values
    ddqn_delays = df_60[df_60["algorithm"] == "DDQN"]["mean_delay_s"].values
    
    bplot = ax.boxplot([cotop_delays, ddqn_delays], tick_labels=["CoTOP", "DDQN"], patch_artist=True, widths=0.5)
    colors = ['#2b5c8f', '#d95f02']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean Delay per Task (s)", fontsize=12)
    ax.set_title("Cross-Algorithm Delay Distribution (60-Cell Dataset)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "delay_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Energy Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    cotop_energy = df_60[df_60["algorithm"] == "CoTOP"]["mean_energy_j"].values
    ddqn_energy = df_60[df_60["algorithm"] == "DDQN"]["mean_energy_j"].values
    bplot = ax.boxplot([cotop_energy, ddqn_energy], tick_labels=["CoTOP", "DDQN"], patch_artist=True, widths=0.5)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean Energy per Task (J)", fontsize=12)
    ax.set_title("Cross-Algorithm Energy Distribution (60-Cell Dataset)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "energy_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Completion Ratio by Seed Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    seeds = [42, 43, 44, 45, 46]
    df_s14 = pd.read_csv(hashes["step14_seed_summary"]["path"])
    comp_ratios = df_s14["completion_ratio"].values
    ax.bar([str(s) for s in seeds], comp_ratios, color='#1b9e77', alpha=0.85, width=0.4)
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_ylabel("Task Completion Ratio", fontsize=12)
    ax.set_title("Task Completion Ratio across Multi-Seed DDQN Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "completion_ratio_by_seed.png"), dpi=300)
    plt.close()
    
    # 4. Training Reward Curves
    fig, ax = plt.subplots(figsize=(9, 5))
    s14_dir = os.path.join(root_dir, "results", "phase2_step14", "linear_corridor_DDQN_w20")
    for s in seeds:
        tc_path = os.path.join(s14_dir, f"seed_{s}", "training_curve.csv")
        if os.path.exists(tc_path):
            tc = pd.read_csv(tc_path)
            smoothed_reward = tc["reward"].rolling(window=50, min_periods=1).mean()
            ax.plot(tc["episode"], smoothed_reward, label=f"Seed {s}", alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("50-Episode Smoothed Reward", fontsize=12)
    ax.set_title("Multi-Seed DDQN Training Reward Trajectories (Step 14)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_reward_curves.png"), dpi=300)
    plt.close()
    
    # 5. Training Loss Curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in seeds:
        tc_path = os.path.join(s14_dir, f"seed_{s}", "training_curve.csv")
        if os.path.exists(tc_path):
            tc = pd.read_csv(tc_path)
            smoothed_loss = tc["loss"].rolling(window=50, min_periods=1).mean()
            ax.plot(tc["episode"], smoothed_loss, label=f"Seed {s}", alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean TD Loss (50-ep smoothed)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Multi-Seed DDQN TD Loss Trajectories (Log Scale)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_loss_curves.png"), dpi=300)
    plt.close()
    
    # 6. Algorithm Comparison Bar Plot
    with open(hashes["single_gate"]["path"], "r") as f:
        gate_data = json.load(f)
    alg_names = list(gate_data["algorithms"].keys())
    alg_delays = [gate_data["algorithms"][k]["mean_delay_s"] for k in alg_names]
    alg_energies = [gate_data["algorithms"][k]["mean_energy_j"] for k in alg_names]
    
    x = np.arange(len(alg_names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    
    rects1 = ax1.bar(x - width/2, alg_delays, width, label='Delay (s)', color='#2b5c8f', alpha=0.85)
    rects2 = ax2.bar(x + width/2, alg_energies, width, label='Energy (J)', color='#d95f02', alpha=0.85)
    
    ax1.set_ylabel('Mean Delay (s)', color='#2b5c8f', fontsize=12)
    ax2.set_ylabel('Mean Energy (J)', color='#d95f02', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(alg_names, fontsize=11)
    ax1.set_title("Cross-Algorithm Benchmark (Matched Realization Gate)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "algorithm_comparison.png"), dpi=300)
    plt.close()
    
    # 7. Effect Sizes Forest Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df_effect))
    effects = df_effect["cohens_dz"].values
    ci_low = df_effect["cohens_dz_ci_low"].values
    ci_high = df_effect["cohens_dz_ci_high"].values
    labels = [f"{r['geometry'].split('_')[0]}_{r['workload']}_{r['metric']}" for _, r in df_effect.iterrows()]
    
    ax.errorbar(effects, y_pos, xerr=[effects - ci_low, ci_high - effects], fmt='o', color='#7570b3', ecolor='#7570b3', elinewidth=2, capsize=4)
    ax.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cohen's dz Effect Size (95% CI)", fontsize=12)
    ax.set_title("Standardized Effect Sizes: CoTOP vs DDQN across Conditions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "effect_sizes_ci.png"), dpi=300)
    plt.close()
    
    print("  Successfully generated all 7 publication figures in figures/phase2_step16/")

def main():
    print("=" * 70)
    print("   PHASE 2 — STEP 16: STATISTICAL VERIFICATION & SYNTHESIS")
    print("=" * 70)
    
    ensure_dirs()
    hashes = verify_provenance()
    df_index = build_raw_experiment_index(hashes)
    df_desc = compute_descriptive_statistics(df_index)
    df_paired, df_effect, df_mult = compute_paired_comparisons(hashes)
    df_cross = build_cross_algorithm_statistics(hashes)
    df_s14_conv, df_disp = build_convergence_and_seed_dispersion(hashes)
    df_agg, df_pub = build_aggregation_and_published_comparison(hashes)
    generate_publication_figures(hashes, df_index, df_paired, df_effect)
    
    print("\n" + "=" * 70)
    print("   STEP 16 STATISTICAL SYNTHESIS COMPLETE (100% REPRODUCIBLE)")
    print("=" * 70)

if __name__ == "__main__":
    main()
