import pytest
import numpy as np
from utils.statistical_analysis import (
    paired_t_test,
    wilcoxon_test,
    cohens_dz,
    common_language_effect_size,
    holm_bonferroni,
    fdr_benjamini_hochberg,
    compute_complete_paired_stats
)

def test_paired_t_test_exact():
    x = [2.0, 3.0, 4.0, 5.0, 6.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = paired_t_test(x, y)
    assert res["n"] == 5
    assert np.isclose(res["mean_diff"], 1.0)
    assert np.isclose(res["std_diff"], 0.0)
    assert np.isclose(res["sem"], 0.0)
    assert np.isclose(res["ci_95_lower"], 1.0)
    assert np.isclose(res["ci_95_upper"], 1.0)

def test_paired_t_test_stochastic():
    np.random.seed(42)
    x = np.random.normal(10, 2, 20)
    y = x + np.random.normal(1.5, 0.5, 20)
    res = paired_t_test(y, x)
    assert res["n"] == 20
    assert res["mean_diff"] > 0
    assert res["p_value"] < 0.001
    assert res["ci_95_lower"] < res["mean_diff"] < res["ci_95_upper"]

def test_wilcoxon_test():
    x = [10, 12, 15, 18, 20]
    y = [8, 9, 11, 14, 15]
    res = wilcoxon_test(x, y)
    assert res["p_value"] < 0.1
    assert res["w_statistic"] == 0.0 or res["w_statistic"] == 15.0

def test_cohens_dz():
    x = [10, 20, 30]
    y = [5, 15, 25]
    # diff = [5, 5, 5], std = 0
    res = cohens_dz(x, y)
    assert res["cohens_dz"] == 0.0
    
    x2 = [10, 22, 35]
    y2 = [5, 15, 25]
    # diff = [5, 7, 10], mean=7.33, std=2.516
    res2 = cohens_dz(x2, y2)
    assert res2["cohens_dz"] > 2.0

def test_common_language_effect_size():
    x = [5, 6, 7, 8]
    y = [1, 2, 3, 4]
    assert np.isclose(common_language_effect_size(x, y), 1.0)
    assert np.isclose(common_language_effect_size(y, x), 0.0)
    assert np.isclose(common_language_effect_size(x, x), 0.5)

def test_holm_bonferroni():
    p_vals = [0.01, 0.04, 0.03, 0.005]
    adj = holm_bonferroni(p_vals)
    # Sorted: 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> max(0.06, 0.04) = 0.06)
    assert np.isclose(adj[3], 0.02)
    assert np.isclose(adj[0], 0.03)
    assert np.isclose(adj[2], 0.06)
    assert np.isclose(adj[1], 0.06)

def test_fdr_benjamini_hochberg():
    p_vals = [0.01, 0.04, 0.03, 0.005]
    q_vals = fdr_benjamini_hochberg(p_vals)
    assert len(q_vals) == 4
    assert np.all(q_vals <= 1.0)
    assert np.all(q_vals >= 0.0)
    # Monotonic in order
    assert q_vals[3] <= q_vals[0] <= q_vals[2] <= q_vals[1]

def test_complete_paired_stats():
    x = [2.05, 2.10, 2.02, 1.98, 2.03]
    y = [2.01, 2.04, 1.99, 1.95, 2.01]
    res = compute_complete_paired_stats(x, y)
    assert res["n"] == 5
    assert res["mean_diff"] > 0
    assert "cohens_dz" in res
    assert "p_value_ttest" in res
    assert "p_value_wilcoxon" in res
    assert "cles" in res
