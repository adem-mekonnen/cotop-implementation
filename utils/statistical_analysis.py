import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Union, Optional

def paired_t_test(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Computes a two-sided paired Student's t-test with mean difference, SEM, and 95% CI.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert len(x) == len(y), "Vectors must have identical length for paired comparison"
    n = len(x)
    assert n >= 2, "Need at least 2 pairs to compute sample variance"
    
    diff = x - y
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    sem = std_diff / np.sqrt(n) if std_diff > 0 else 0.0
    
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lower = mean_diff - t_crit * sem
    ci_upper = mean_diff + t_crit * sem
    
    if std_diff > 1e-12:
        t_res = stats.ttest_rel(x, y)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
    else:
        t_stat = 0.0
        p_val = 1.0
        
    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "sem": sem,
        "t_statistic": t_stat,
        "p_value": p_val,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper
    }

def wilcoxon_test(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Computes a two-sided Wilcoxon signed-rank test for paired observations.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert len(x) == len(y), "Vectors must have identical length for paired comparison"
    diff = x - y
    
    if np.allclose(diff, 0.0):
        return {"w_statistic": 0.0, "p_value": 1.0}
        
    try:
        w_res = stats.wilcoxon(x, y, alternative='two-sided', zero_method='wilcox')
        return {"w_statistic": float(w_res.statistic), "p_value": float(w_res.pvalue)}
    except Exception:
        return {"w_statistic": float('nan'), "p_value": float('nan')}

def cohens_dz(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Computes Cohen's d_z for paired designs: d_z = mean(diff) / std(diff).
    Includes 95% confidence interval estimation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff > 1e-12:
        dz = float(mean_diff / std_diff)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        se_dz = np.sqrt(1.0 / n + (dz ** 2) / (2.0 * n))
        ci_lower = dz - t_crit * se_dz
        ci_upper = dz + t_crit * se_dz
    else:
        dz, ci_lower, ci_upper = 0.0, 0.0, 0.0
        
    return {
        "cohens_dz": dz,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper)
    }

def common_language_effect_size(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
    """
    Computes Common Language Effect Size (CLES) / Probability of Superiority:
    Pr(X > Y) + 0.5 * Pr(X == Y) for paired samples.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert len(x) == len(y), "Vectors must have identical length"
    
    diff = x - y
    greater = np.sum(diff > 1e-12)
    equal = np.sum(np.abs(diff) <= 1e-12)
    cles = (greater + 0.5 * equal) / len(diff)
    return float(cles)

def holm_bonferroni(p_values: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Applies step-down Holm-Bonferroni correction to control Family-Wise Error Rate (FWER).
    """
    p_vals = np.asarray(p_values, dtype=np.float64)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=np.float64)
        
    order = np.argsort(p_vals)
    adjusted = np.zeros(n, dtype=np.float64)
    running_max = 0.0
    for i, idx in enumerate(order):
        adj = p_vals[idx] * (n - i)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted

def fdr_benjamini_hochberg(p_values: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Applies Benjamini-Hochberg False Discovery Rate (FDR) correction.
    """
    p_vals = np.asarray(p_values, dtype=np.float64)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=np.float64)
        
    order = np.argsort(p_vals)
    reverse_order = np.argsort(order)
    sorted_p = p_vals[order]
    
    q_vals = np.zeros(n, dtype=np.float64)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        q = sorted_p[i] * n / (i + 1)
        running_min = min(running_min, q)
        q_vals[i] = min(1.0, running_min)
        
    return q_vals[reverse_order]

def compute_complete_paired_stats(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Executes complete paired statistical diagnostic pipeline.
    """
    t_res = paired_t_test(x, y)
    w_res = wilcoxon_test(x, y)
    dz_res = cohens_dz(x, y)
    cles_val = common_language_effect_size(x, y)
    
    return {
        "n": t_res["n"],
        "mean_diff": t_res["mean_diff"],
        "std_diff": t_res["std_diff"],
        "sem": t_res["sem"],
        "t_statistic": t_res["t_statistic"],
        "p_value_ttest": t_res["p_value"],
        "ci_95_lower": t_res["ci_95_lower"],
        "ci_95_upper": t_res["ci_95_upper"],
        "w_statistic": w_res["w_statistic"],
        "p_value_wilcoxon": w_res["p_value"],
        "cohens_dz": dz_res["cohens_dz"],
        "cohens_dz_ci_lower": dz_res["ci_95_lower"],
        "cohens_dz_ci_upper": dz_res["ci_95_upper"],
        "cles": cles_val
    }
