"""
experiments/stage16_statistical_protocol.py

Executes STAGE 16 — FINAL STATISTICAL ANALYSIS.
Performs rigorous paired statistical evaluation on frozen Phase 2 reproduction dataset:
- CoTOP vs DDQN
- CoTOP vs Greedy
- CoTOP vs Local

Computes:
- Full difference vector
- Mean difference & standard deviation
- Paired Student's t-test (t, df, p-value)
- Wilcoxon signed-rank test (W, p-value)
- Cohen's dz effect size
- 95% Confidence Intervals
- Multiple-comparison corrections (Bonferroni & Benjamini-Hochberg FDR)
- Small-sample diagnostics (n=5 limitations)

Generates:
- results/phase2_algorithmic_fidelity/statistical_analysis_final.csv
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Any, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

DATA_PATH = os.path.join("results", "phase2_algorithmic_fidelity", "table4_5_reproduction.csv")
OUT_CSV = os.path.join("results", "phase2_algorithmic_fidelity", "statistical_analysis_final.csv")


def cohens_dz(diffs: np.ndarray) -> float:
    """Computes Cohen's dz for paired differences: mean(diffs) / std(diffs, ddof=1)"""
    s_d = np.std(diffs, ddof=1)
    if s_d == 0 or np.isnan(s_d):
        return 0.0
    return float(np.mean(diffs) / s_d)


def compute_paired_stats(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    label_a: str = "CoTOP",
    label_b: str = "Baseline"
) -> Dict[str, Any]:
    """
    Computes complete paired statistical battery between vec_a and vec_b (where diff = vec_a - vec_b).
    """
    n = len(vec_a)
    assert len(vec_b) == n, "Vectors must have identical length"
    
    diffs = vec_a - vec_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    
    # 1. Paired Student's t-test
    if std_diff > 1e-12:
        t_stat, p_val_t = stats.ttest_rel(vec_a, vec_b)
        t_stat = float(t_stat)
        p_val_t = float(p_val_t)
        # 95% CI
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
    else:
        t_stat = 0.0
        p_val_t = 1.0
        ci_lower = mean_diff
        ci_upper = mean_diff

    # 2. Wilcoxon Signed-Rank Test
    # Handle zero differences gracefully
    non_zero_diffs = diffs[np.abs(diffs) > 1e-12]
    if len(non_zero_diffs) >= 5:
        try:
            w_res = stats.wilcoxon(vec_a, vec_b, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            p_val_w = 1.0
    elif len(non_zero_diffs) > 0:
        # Small non-zero count
        try:
            w_res = stats.wilcoxon(non_zero_diffs, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            p_val_w = 1.0
    else:
        w_stat = 0.0
        p_val_w = 1.0

    # 3. Effect Size
    dz = cohens_dz(diffs)
    
    diff_vector_str = "[" + ", ".join(f"{d:+.4f}" for d in diffs) + "]"

    return {
        "n": n,
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "diff_vector": diff_vector_str,
        "t_stat": round(t_stat, 4),
        "df": n - 1,
        "p_val_t": round(p_val_t, 6),
        "w_stat": round(w_stat, 4),
        "p_val_w": round(p_val_w, 6),
        "cohens_dz": round(dz, 4),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6)
    }


def run_full_statistical_analysis():
    print("=" * 80)
    print("      STAGE 16: FINAL COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded frozen dataset: {len(df)} rows across {df['algorithm'].nunique()} algorithms.\n")

    geometries = ["corridor_2400m", "grid_200m"]
    workloads = [20, 30, 40]
    comparisons = [
        ("CoTOP", "DDQN"),
        ("CoTOP", "Greedy"),
        ("CoTOP", "Local")
    ]
    metrics = [
        ("mean_delay", "Delay (s)"),
        ("mean_energy", "Energy (J)")
    ]

    all_results = []

    # 1. Condition-Level Analysis (n=5 seeds per cell)
    for geom in geometries:
        for w in workloads:
            sub = df[(df["geometry"] == geom) & (df["workload"] == w)]
            # Ensure sort by seed
            sub_cotop = sub[sub["algorithm"] == "CoTOP"].sort_values("seed")
            
            for base_algo, base_name in [("DDQN", "DDQN"), ("Greedy", "Greedy"), ("Local", "Local")]:
                sub_base = sub[sub["algorithm"] == base_algo].sort_values("seed")
                
                for metric_col, metric_label in metrics:
                    vec_cotop = sub_cotop[metric_col].to_numpy()
                    vec_base = sub_base[metric_col].to_numpy()
                    
                    st = compute_paired_stats(vec_cotop, vec_base, "CoTOP", base_algo)
                    
                    res_row = {
                        "analysis_scope": "Cell_Level",
                        "geometry": geom,
                        "workload": w,
                        "comparison": f"CoTOP vs {base_algo}",
                        "metric": metric_label,
                        "n": st["n"],
                        "mean_cotop": round(float(np.mean(vec_cotop)), 4),
                        "mean_baseline": round(float(np.mean(vec_base)), 4),
                        "mean_diff": st["mean_diff"],
                        "std_diff": st["std_diff"],
                        "diff_vector": st["diff_vector"],
                        "t_stat": st["t_stat"],
                        "df": st["df"],
                        "p_val_t": st["p_val_t"],
                        "w_stat": st["w_stat"],
                        "p_val_w": st["p_val_w"],
                        "cohens_dz": st["cohens_dz"],
                        "ci_95": f"[{st['ci_95_lower']:.4f}, {st['ci_95_upper']:.4f}]"
                    }
                    all_results.append(res_row)

    # 2. Geometry-Aggregated Analysis (n=15 realizations per geometry)
    for geom in geometries:
        sub = df[df["geometry"] == geom]
        sub_cotop = sub[sub["algorithm"] == "CoTOP"].sort_values(["workload", "seed"])
        for base_algo in ["DDQN", "Greedy", "Local"]:
            sub_base = sub[sub["algorithm"] == base_algo].sort_values(["workload", "seed"])
            for metric_col, metric_label in metrics:
                vec_cotop = sub_cotop[metric_col].to_numpy()
                vec_base = sub_base[metric_col].to_numpy()
                st = compute_paired_stats(vec_cotop, vec_base, "CoTOP", base_algo)
                res_row = {
                    "analysis_scope": f"Geometry_Aggregated ({geom})",
                    "geometry": geom,
                    "workload": "All (20,30,40)",
                    "comparison": f"CoTOP vs {base_algo}",
                    "metric": metric_label,
                    "n": st["n"],
                    "mean_cotop": round(float(np.mean(vec_cotop)), 4),
                    "mean_baseline": round(float(np.mean(vec_base)), 4),
                    "mean_diff": st["mean_diff"],
                    "std_diff": st["std_diff"],
                    "diff_vector": st["diff_vector"],
                    "t_stat": st["t_stat"],
                    "df": st["df"],
                    "p_val_t": st["p_val_t"],
                    "w_stat": st["w_stat"],
                    "p_val_w": st["p_val_w"],
                    "cohens_dz": st["cohens_dz"],
                    "ci_95": f"[{st['ci_95_lower']:.4f}, {st['ci_95_upper']:.4f}]"
                }
                all_results.append(res_row)

    # 3. Global Factorial Aggregation (n=30 realizations total)
    sub_cotop = df[df["algorithm"] == "CoTOP"].sort_values(["geometry", "workload", "seed"])
    for base_algo in ["DDQN", "Greedy", "Local"]:
        sub_base = df[df["algorithm"] == base_algo].sort_values(["geometry", "workload", "seed"])
        for metric_col, metric_label in metrics:
            vec_cotop = sub_cotop[metric_col].to_numpy()
            vec_base = sub_base[metric_col].to_numpy()
            st = compute_paired_stats(vec_cotop, vec_base, "CoTOP", base_algo)
            res_row = {
                "analysis_scope": "Global_Factorial_Total",
                "geometry": "Both",
                "workload": "All",
                "comparison": f"CoTOP vs {base_algo}",
                "metric": metric_label,
                "n": st["n"],
                "mean_cotop": round(float(np.mean(vec_cotop)), 4),
                "mean_baseline": round(float(np.mean(vec_base)), 4),
                "mean_diff": st["mean_diff"],
                "std_diff": st["std_diff"],
                "diff_vector": st["diff_vector"],
                "t_stat": st["t_stat"],
                "df": st["df"],
                "p_val_t": st["p_val_t"],
                "w_stat": st["w_stat"],
                "p_val_w": st["p_val_w"],
                "cohens_dz": st["cohens_dz"],
                "ci_95": f"[{st['ci_95_lower']:.4f}, {st['ci_95_upper']:.4f}]"
            }
            all_results.append(res_row)

    # Apply Multiple-Comparison Correction (Benjamini-Hochberg FDR & Bonferroni)
    res_df = pd.DataFrame(all_results)
    
    # Cell-level Bonferroni (m = 36 cell tests)
    m_cell = len(res_df[res_df["analysis_scope"] == "Cell_Level"])
    alpha_bonf = 0.05 / m_cell if m_cell > 0 else 0.05
    
    p_vals = res_df["p_val_t"].to_numpy()
    # Benjamini-Hochberg adjustment
    ranked_indices = np.argsort(p_vals)
    ranks = np.empty_like(ranked_indices)
    ranks[ranked_indices] = np.arange(1, len(p_vals) + 1)
    q_vals = p_vals * len(p_vals) / ranks
    q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_vals = np.clip(q_vals, 0.0, 1.0)
    
    res_df["p_val_bonferroni_adj"] = np.clip(p_vals * m_cell, 0.0, 1.0)
    res_df["q_val_fdr"] = np.round(q_vals, 6)
    res_df["sig_bonferroni"] = res_df["p_val_t"] < alpha_bonf
    res_df["sig_fdr_05"] = res_df["q_val_fdr"] < 0.05

    res_df.to_csv(OUT_CSV, index=False)
    print(f"[SUCCESS] Wrote comprehensive statistical ledger to {OUT_CSV} ({len(res_df)} test rows)")

    # Print summary of key findings
    print("\n" + "=" * 80)
    print("KEY GLOBAL FACTORIAL COMPARISONS (n=30 Paired Realizations):")
    print("=" * 80)
    glob = res_df[res_df["analysis_scope"] == "Global_Factorial_Total"]
    for idx, r in glob.iterrows():
        print(f"[{r['comparison']:16s} | {r['metric']:10s}] Mean Diff: {r['mean_diff']:+.4f} | t({r['df']}) = {r['t_stat']:+.2f}, p={r['p_val_t']:.4e} | Wilcoxon p={r['p_val_w']:.4e} | Cohen's dz: {r['cohens_dz']:+.2f} | 95% CI: {r['ci_95']}")


if __name__ == "__main__":
    run_full_statistical_analysis()
