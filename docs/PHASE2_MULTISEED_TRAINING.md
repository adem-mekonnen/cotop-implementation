# PHASE 2: MULTISEED TRAINING EXPERIMENT

## 1. Experiment Design
- **Algorithms**: CoTOP, DDQN
- **Geometries**: corridor_2400m, grid_200m
- **Workloads**: w20, w30, w40
- **Seeds**: 42, 43, 44, 45, 46
- **Total Planned Runs**: 60
- **Total Successful Runs**: 60

## 2. Invariants Assured
- Strict use of FrozenVECEnv across all runs
- Absolute pairing of exogenous traces per seed/geometry/workload configuration
- Automated hash locking of configuration, model weights, and realization files

## 3. Results Summary
Please refer to `results/phase2_multiseed/seed_results.csv` for complete tabular data.
