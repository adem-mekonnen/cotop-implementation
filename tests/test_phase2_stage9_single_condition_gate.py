"""
tests/test_phase2_stage9_single_condition_gate.py

Unit Test Suite for STAGE 9 — SINGLE-CONDITION COMPARATIVE GATE.
Verifies all 12 gates under the canonical condition (corridor_2400m, w20, seed 0).
"""

import os
import pytest
import numpy as np

from experiments.run_stage9_single_condition_gate import run_stage9_gate


def test_stage9_all_twelve_gates_pass():
    """
    Executes the Stage 9 Single-Condition Comparative Gate and validates all 12 criteria.
    """
    results = run_stage9_gate()
    
    assert results["all_gates_passed"] is True
    gate_results = results["gate_results"]
    
    expected_gates = [
        "Gate 1: Realization Hash Identical",
        "Gate 2: Environment Configuration Identical",
        "Gate 3: Task Count Identical",
        "Gate 4: Vehicle Trajectories Identical",
        "Gate 5: Action-Space Semantics Identical",
        "Gate 6: Evaluation Weights Immutable",
        "Gate 7: Deterministic Action Sequence",
        "Gate 8: Deterministic State Sequence",
        "Gate 9: Task Conservation",
        "Gate 10: Latency Decomposition",
        "Gate 11: Energy Decomposition",
        "Gate 12: No NaN/Inf"
    ]
    
    for g in expected_gates:
        assert g in gate_results, f"Missing gate in results: {g}"
        assert gate_results[g] is True, f"Gate failed: {g}"
        
    # Verify algorithm results presence
    for algo in ["Local", "Greedy", "DDQN", "CoTOP"]:
        assert algo in results["algorithms"]
        r = results["algorithms"][algo]
        assert r["total_tasks"] == 200
        assert r["completed_tasks"] == 200
        assert r["failed_tasks"] == 0
        assert r["completion_ratio"] == 1.0
        assert np.isfinite(r["mean_delay_s"]) and r["mean_delay_s"] > 0
        assert np.isfinite(r["mean_energy_j"]) and r["mean_energy_j"] > 0
