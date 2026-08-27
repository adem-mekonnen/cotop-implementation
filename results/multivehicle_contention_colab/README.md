# Multi-Vehicle Contention Colab Reproduction Experiment

## Executive Summary
This directory contains the immutable experiment records for the **Multi-Vehicle Concurrent Contention** reproduction of CoTOP.
- **Branch**: `reproduction/multivehicle-contention`
- **Base Commit**: `bd34c65`
- **Execution Date**: 2026-08-27 11:46:55
- **Seeds Evaluated**: [0, 1, 2, 3, 4]
- **Total Training Episodes**: 250 (50 episodes/seed)
- **Total Evaluation Episodes**: 300 (100 episodes per policy)

## Key Findings
1. **Contention Emergence**: RSU queue backlog reaches **139.56 Mcycles** under multi-vehicle traffic, generating physical queue waiting delay up to **0.1329 s**.
2. **CoTOP Performance**:
   - Total Delay: **1.9849 ± 0.0253 s**
   - Total Energy: **4.0686 ± 0.7274 J**
   - Completion Rate: **100.00%**
3. **Statistical Significance**: CoTOP demonstrates verified load-balancing and collaboration semantics across all 5 random seeds.

## Directory Artifacts
- `experiment_config.json`: Complete parameter specification.
- `training_summary.csv`: Step-by-step training curves per episode and seed.
- `evaluation_episode_results.csv`: 100 paired evaluation episodes across CoTOP, Local, Greedy.
- `seed_summary.csv`: Aggregated seed-level performance metrics.
- `statistical_analysis.csv`: Paired t-test, Wilcoxon, Cohen's dz, Holm/FDR multiple-testing corrections.
- `published_vs_reproduced.csv`: Direct quantitative comparison against paper headline values.
- `queue_diagnostics.csv`: Scalability diagnostics across N in [2, 5, 10, 20, 30].
- `runtime_vehicle_diagnostics.csv`: Slot-by-slot SUMO vehicle telemetry proving true concurrency.
- `environment_validation.txt`: Immutability audit and test verification log.
