# Phase 2 — Environment Integration Forensic Wiring Audit (Stage 3)

**Document ID**: `PHASE2_ENVIRONMENT_WIRING_AUDIT`  
**Audited Branch**: `reproduction/scientific-fidelity`  
**Scope**: Forensic tracing of Phase-1 environment parameters from CLI/configuration to VECEnv, environment state, and reinforcement learning agents across all training and evaluation runners.

---

## 1. Executive Summary

A forensic audit of parameter propagation across `train.py`, `evaluate.py`, `run_experiments.py`, and `scripts/run_phase2_multiseed.py` was conducted. All CLI and configuration parameters were verified to propagate end-to-end into `VECEnv` without silent drop or substitution:

$$\boxed{\text{CLI / YAML Config} \longrightarrow \text{Runner / Worker} \longrightarrow \text{VECEnv(\dots)} \longrightarrow \text{Environment State} \longrightarrow \text{Agent / Algorithm}}$$

Automated verification tests in [`tests/test_phase2_environment_wiring.py`](file:///d:/cotop-implementation/tests/test_phase2_environment_wiring.py) passed **6/6 (100%)**.

---

## 2. End-to-End Parameter Propagation Ledger

| Parameter Name | CLI Argument | YAML Config Key | `VECEnv` Internal Attribute | Propagation Target & Behavioral Effect | Verified In Code & Test |
|---|---|---|---|---|---|
| **`scenario_geometry`** | `--scenario_geometry` | `scenario_geometry` | `self.scenario_geometry` | Controls `sumo_manager.sumocfg_path` (`hangzhou_200m.sumocfg` vs `hangzhou.sumocfg`), `self.map_scale` ($200.0\text{ m}$ vs $2400.0\text{ m}$), and 2D grid vs 1D line RSU deployment coordinates. | **PASS** (Test 01) |
| **`use_mobility_model`** | `--use_mobility_model` / `--no_mobility_model` | `use_mobility_model` | `self.use_mobility_model` | Controls whether `MobilityGAT_GRU` is loaded and evaluated on spatial graphs for dynamic dwell time estimation ($T_{stay, v}$). | **PASS** (Test 02) |
| **`priority_mode`** | `--priority_mode` | `priority_mode` | `self.priority_mode` | Selects between `'paper_literal'` (Eq 23 literal) and `'normalized_candidate'` (dual min-max queue scaling). | **PASS** (Test 03) |
| **`coverage_mode`** | `--coverage_mode` | `coverage_mode` | `self.coverage_mode` | Selects between `'completion_position'` (Eq 25 position check) and `'continuous_required_rsus'`. | **PASS** (Test 03) |
| **`spatial_graph_radius`** | `--spatial_graph_radius` | `spatial_graph_radius` | `self.spatial_graph_radius` | Sets edge connection distance threshold ($d \le R_{graph}$) for multi-node vehicle graph construction. | **PASS** (Test 04) |
| **`max_vehicles`** | `--max_vehicles` | `max_vehicles` | `self.max_vehicles` | Bounds maximum concurrent vehicles in simulation. | **PASS** (Test 04) |
| **`workload`** | `--workload` | `num_tasks_per_vehicle_range` | `self.config.num_tasks_per_vehicle_range` | Dynamically dimensions state observation vector: $D = 4 + 4w + 5M$ ($114, 154, 194$). | **PASS** (Test 05) |
| **`seed`** | `--seed` | `seed` | `self.seed` | Initializes global RNG, numpy RNG, torch RNG, and TraCI simulation seed. | **PASS** (Test 01) |
| **`physics_hash`** | N/A (Hard-Locked) | N/A | `comm_model.py`, `comp_model.py` | Hash-locked to prevent accidental physics drift or tuning. | **PASS** (Test 06) |

---

## 3. Dual Scenario Geometry Smoke Verification

Direct execution of `test_01_scenario_geometry_wiring_and_sumo_cfg` proves that changing `scenario_geometry` alters the physical SUMO simulation and RSU geometry:

### 3.1 Urban Manhattan Grid (`grid_200m`)
- **SUMO Config File**: `sumo_config/hangzhou_200m.sumocfg`
- **Network Extents**: $200.0\text{ m} \times 200.0\text{ m}$
- **Map Scale**: $200.0\text{ m}$
- **RSU Positions (6 RSUs in $2 \times 3$ grid)**:
  $$(50.0, 50.0), (100.0, 50.0), (150.0, 50.0), (50.0, 150.0), (100.0, 150.0), (150.0, 150.0)$$

### 3.2 Linear Corridor (`corridor_2400m`)
- **SUMO Config File**: `sumo_config/hangzhou.sumocfg`
- **Network Extents**: $2400.0\text{ m}$ linear span
- **Map Scale**: $2400.0\text{ m}$
- **RSU Positions (6 RSUs in 1D arterial line)**:
  $$(200.0, 0.0), (600.0, 0.0), (1000.0, 0.0), (1400.0, 0.0), (1800.0, 0.0), (2200.0, 0.0)$$

---

## 4. Cross-Execution Physics & Configuration Mismatch Prevention

To ensure that training and evaluation cannot accidentally use mismatched physics or configurations without detection:
1. **Manifest Lock**: Every evaluation run records `comm_model_hash`, `comp_model_hash`, `evaluation_exogenous_trace_hash`, and `checkpoint_hash` in `run_manifest.json`.
2. **Dynamic Observation Dimension Check**: If an agent trained on $w=20$ ($D=114$) is evaluated on $w=30$ ($D=154$), PyTorch tensor linear transformation immediately errors (`RuntimeError: mat1 and mat2 shapes cannot be multiplied`).
3. **Weight Immutability Invariant**: Evaluator asserts $H(\theta_{\text{before}}) \equiv H(\theta_{\text{after}})$, preventing hidden fine-tuning during evaluation passes.
4. **Automated Regression Invariant**: Test 18 in `test_phase2_step14_multiseed.py` and Test 06 in `test_phase2_environment_wiring.py` verify Phase-1 physics SHA-256 hashes on every pytest run.

---

## 5. Audit Verdict & Conclusion

- **Audit Status**: **ALL PARAMETERS WIRED & VERIFIED (PASS)**
- **Test Results**: **6/6 PASS** in `tests/test_phase2_environment_wiring.py`
- **Full Suite**: **94/94 PASS across entire repository**
- **Silent Parameter Drops**: **ZERO**

*Stage 3 Environment Integration Forensic Fix is complete. Execution has stopped in compliance with instructions.*
