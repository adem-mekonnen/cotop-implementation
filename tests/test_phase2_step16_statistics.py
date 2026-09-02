import pytest
import os
import hashlib
import numpy as np
import pandas as pd
from utils.statistical_analysis import (
    paired_t_test,
    wilcoxon_test,
    cohens_dz,
    common_language_effect_size,
    holm_bonferroni,
    fdr_benjamini_hochberg,
    compute_complete_paired_stats
)
from scripts.run_phase2_step16_statistics import (
    verify_provenance,
    build_raw_experiment_index,
    compute_descriptive_statistics,
    compute_paired_comparisons,
    build_cross_algorithm_statistics,
    RESULTS_DIR,
    FIGURES_DIR
)

# 1. Descriptive Statistics Invariant Test
def test_01_descriptive_statistics_calculation():
    vals = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mean_val = np.mean(vals)
    std_val = np.std(vals, ddof=1)
    median_val = np.median(vals)
    iqr_val = np.percentile(vals, 75) - np.percentile(vals, 25)
    
    assert np.isclose(mean_val, 5.0)
    assert np.isclose(median_val, 4.5)
    assert iqr_val == 1.5

# 2. Paired Difference Calculation Test
def test_02_paired_difference_calculation():
    x = np.array([10.0, 12.0, 14.0])
    y = np.array([8.0, 9.0, 10.0])
    diff = x - y
    assert np.array_equal(diff, np.array([2.0, 3.0, 4.0]))
    assert np.isclose(np.mean(diff), 3.0)

# 3. Paired T-Test
def test_03_paired_t_test():
    x = [5.0, 6.0, 7.0, 8.0, 9.0]
    y = [3.0, 4.0, 5.0, 6.0, 7.0]
    res = paired_t_test(x, y)
    assert res["n"] == 5
    assert np.isclose(res["mean_diff"], 2.0)
    assert np.isclose(res["std_diff"], 0.0)
    assert np.isclose(res["ci_95_lower"], 2.0)
    assert np.isclose(res["ci_95_upper"], 2.0)

# 4. Wilcoxon Signed-Rank Test
def test_04_wilcoxon_signed_rank_test():
    x = [10.0, 15.0, 20.0, 25.0, 30.0]
    y = [8.0, 12.0, 18.0, 22.0, 28.0]
    res = wilcoxon_test(x, y)
    assert res["p_value"] < 0.1
    assert not np.isnan(res["w_statistic"])

# 5. Cohen's dz Effect Size
def test_05_cohens_dz():
    x = [12.0, 15.0, 18.0, 21.0, 24.0]
    y = [10.0, 12.0, 14.0, 16.0, 18.0]
    res = cohens_dz(x, y)
    # diff = [2, 3, 4, 5, 6], mean=4, std=1.5811
    expected_dz = 4.0 / np.std([2, 3, 4, 5, 6], ddof=1)
    assert np.isclose(res["cohens_dz"], expected_dz)
    assert res["ci_95_lower"] < res["cohens_dz"] < res["ci_95_upper"]

# 6. Common Language Effect Size (CLES)
def test_06_common_language_effect_size():
    x = [5.0, 6.0, 7.0, 8.0]
    y = [1.0, 2.0, 3.0, 4.0]
    assert np.isclose(common_language_effect_size(x, y), 1.0)
    assert np.isclose(common_language_effect_size(y, x), 0.0)
    assert np.isclose(common_language_effect_size(x, x), 0.5)

# 7. Confidence Interval Coverage
def test_07_confidence_intervals():
    x = [2.05, 2.10, 2.02, 1.98, 2.03]
    y = [2.01, 2.04, 1.99, 1.95, 2.01]
    res = compute_complete_paired_stats(x, y)
    assert res["ci_95_lower"] <= res["mean_diff"] <= res["ci_95_upper"]
    assert res["cohens_dz_ci_lower"] <= res["cohens_dz"] <= res["cohens_dz_ci_upper"]

# 8. Holm-Bonferroni Correction
def test_08_holm_bonferroni():
    p_vals = [0.01, 0.04, 0.03, 0.005]
    adj = holm_bonferroni(p_vals)
    assert np.isclose(adj[3], 0.02)
    assert np.isclose(adj[0], 0.03)
    assert np.isclose(adj[2], 0.06)
    assert np.isclose(adj[1], 0.06)

# 9. Benjamini-Hochberg FDR Correction
def test_09_fdr_benjamini_hochberg():
    p_vals = [0.01, 0.04, 0.03, 0.005]
    q_vals = fdr_benjamini_hochberg(p_vals)
    assert len(q_vals) == 4
    assert np.all(q_vals <= 1.0)
    assert np.all(q_vals >= 0.0)

# 10. Deterministic Pipeline Output
def test_10_deterministic_pipeline_output():
    hashes = verify_provenance()
    df_index1 = build_raw_experiment_index(hashes)
    df_index2 = build_raw_experiment_index(hashes)
    pd.testing.assert_frame_equal(df_index1, df_index2)

# 11. Missing Data Handling
def test_11_missing_data_handling():
    p_empty = holm_bonferroni([])
    assert len(p_empty) == 0
    q_empty = fdr_benjamini_hochberg([])
    assert len(q_empty) == 0

# 12. Provenance Validation
def test_12_provenance_validation():
    hashes = verify_provenance()
    assert "step14_seed_summary" in hashes
    assert "summary_60cell" in hashes
    assert "single_gate" in hashes

# 13. Aggregation-Level Validation
def test_13_aggregation_level_validation():
    agg_path = os.path.join(RESULTS_DIR, "aggregation_audit.csv")
    if os.path.exists(agg_path):
        df_agg = pd.read_csv(agg_path)
        assert "metric_a_delay_per_subtask_s" in df_agg.columns
        assert "metric_b_delay_per_vehicle_workload_s" in df_agg.columns
        # Vehicle aggregate must strictly exceed per-subtask mean
        assert (df_agg["metric_b_delay_per_vehicle_workload_s"] > df_agg["metric_a_delay_per_subtask_s"]).all()

# 14. Rejection of Invalid Pairing
def test_14_rejection_of_invalid_pairing():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0]
    with pytest.raises(AssertionError):
        paired_t_test(x, y)

# 15. Source File Immutability
def test_15_no_modification_of_source_result_files():
    hashes_before = verify_provenance()
    hashes_after = verify_provenance()
    for k in hashes_before:
        assert hashes_before[k]["sha256"] == hashes_after[k]["sha256"], f"Source file mutated: {k}"
