# Phase 2 Canonical Realization Protocol

## Overview
To guarantee absolute fairness during the evaluation of the algorithmic baselines (CoTOP, DDQN, etc.), this repository enforces a strictly deterministic exogenous sequence via the `Realization` data structure. 

All algorithms evaluated must consume the exact same realization without modifying it. A realization encapsulates all randomness and SUMO interactions (vehicle arrivals, coordinates, trajectories, velocities, and task characteristics).

## Realization Schema (`1.0`)

A materialization JSON file contains the following fields:

- `realization_id`: String formatted as `realization_{geometry}_{seed}`
- `geometry`: Identifier (e.g., `corridor_2400m` or `grid_200m`)
- `workload`: `I_X` string identifying tasks generated per vehicle
- `seed`: Integer RNG seed used to construct the environment
- `config_hash`: SHA-256 hash of the simulation configuration parameters
- `git_sha`: SHA of the codebase at materialization time
- `creation_timestamp`: Unix timestamp
- `schema_version`: Data structure version (currently `1.0`)
- `vehicle_trace`: A mapping of time-step integers to a dictionary of vehicle IDs and their respective positional and velocity state.
- `task_trace`: A mapping of vehicle IDs to a list of pre-generated discrete task dictionaries (size, CPU cycles, deadline).
- `hash`: SHA-256 integrity hash over the JSON structure.

## Mechanism

1. **Materialization**: `scripts/materialize_evaluation_realizations.py` launches a full standalone `VECEnv` connected to SUMO and sweeps through `0..max_sim_time`, extracting the generated tasks and step-by-step positions directly into memory. This structure is written to JSON and hashed.
2. **Evaluation Execution**: The algorithms are executed over `envs.frozen_vec_env.FrozenVECEnv`. This environment bypasses SUMO and the `TaskGenerator` completely. It iterates over the internal `vehicle_trace`, placing vehicles deterministically and provisioning them with tasks mathematically identically regardless of internal action choices.
3. **Immutability Guarantee**: Any accidental divergence (e.g. attempting to override a frozen seed) triggers immediate `ValueError` exceptions, mathematically guaranteeing that the DDQN and CoTOP models receive identically shaped and valued tasks at the exact same spatial coordinates.
