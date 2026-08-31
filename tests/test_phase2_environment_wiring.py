import hashlib
import os
import pytest
import yaml
import numpy as np
import torch

from envs.entities import SimulationConfig
from envs.vec_env import VECEnv
from utils.scenario_geometry import get_rsu_positions


def test_01_scenario_geometry_wiring_and_sumo_cfg():
    """
    Test 01 — Scenario Geometry Wiring
    Verify that passing scenario_geometry='grid_200m' or 'corridor_2400m'
    correctly configures map_scale, SUMO cfg file, and RSU coordinates.
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
        
    # 1. Urban Manhattan Grid (200m)
    env_grid = VECEnv(config=cfg, scenario_geometry="grid_200m", port=9101, seed=42)
    assert env_grid.scenario_geometry == "grid_200m"
    assert env_grid.map_scale == 200.0
    assert "hangzhou_200m.sumocfg" in env_grid.sumo_manager.sumocfg_path
    
    env_grid.reset(seed=42)
    # Check RSU coordinates for grid
    rsu_grid_coords = [r.location for r in env_grid.rsus]
    expected_grid_coords = get_rsu_positions(num_rsus=6, scenario_mode="grid_200m")
    assert rsu_grid_coords == expected_grid_coords
    assert max(c[0] for c in rsu_grid_coords) <= 200.0
    assert max(c[1] for c in rsu_grid_coords) <= 200.0
    env_grid.close()
    
    # 2. Linear Corridor (2400m)
    env_corr = VECEnv(config=cfg, scenario_geometry="corridor_2400m", port=9102, seed=42)
    assert env_corr.scenario_geometry == "corridor_2400m"
    assert env_corr.map_scale == 2400.0
    assert "hangzhou.sumocfg" in env_corr.sumo_manager.sumocfg_path
    
    env_corr.reset(seed=42)
    rsu_corr_coords = [r.location for r in env_corr.rsus]
    expected_corr_coords = get_rsu_positions(num_rsus=6, scenario_mode="corridor_2400m")
    assert rsu_corr_coords == expected_corr_coords
    assert max(c[0] for c in rsu_corr_coords) > 1000.0
    env_corr.close()



def test_02_mobility_model_flag_wiring():
    """
    Test 02 — Mobility Model Flag Wiring
    Verify that use_mobility_model=True initializes the GAT model,
    while use_mobility_model=False disables it.
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
        
    env_mob = VECEnv(config=cfg, use_mobility_model=True, port=9103, seed=42)
    assert env_mob.use_mobility_model is True
    # If checkpoint exists, mobility_model should be loaded
    if os.path.exists("results/checkpoints/mobility_model.pth"):
        assert env_mob.mobility_model is not None
    env_mob.close()
    
    env_no_mob = VECEnv(config=cfg, use_mobility_model=False, port=9104, seed=42)
    assert env_no_mob.use_mobility_model is False
    assert env_no_mob.mobility_model is None
    env_no_mob.close()


def test_03_priority_mode_and_coverage_mode_wiring():
    """
    Test 03 — Priority Mode and Coverage Mode Parameter Wiring
    Verify priority_mode and coverage_mode reach VECEnv attributes.
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
        
    env = VECEnv(
        config=cfg,
        priority_mode="normalized_candidate",
        coverage_mode="continuous_required_rsus",
        port=9105,
        seed=42
    )
    assert env.priority_mode == "normalized_candidate"
    assert env.coverage_mode == "continuous_required_rsus"
    env.close()


def test_04_spatial_graph_radius_and_max_vehicles_wiring():
    """
    Test 04 — Spatial Graph Radius and Max Vehicles Wiring
    Verify spatial_graph_radius and max_vehicles are stored and respected in VECEnv.
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
        
    env = VECEnv(
        config=cfg,
        spatial_graph_radius=350.0,
        max_vehicles=15,
        port=9106,
        seed=42
    )
    assert env.spatial_graph_radius == 350.0
    assert env.max_vehicles == 15
    env.close()


def test_05_workload_configuration_observation_dimension_scaling():
    """
    Test 05 — Workload Configuration Observation Dimension Scaling
    Verify that varying workload (tasks/vehicle: 20, 30, 40) strictly scales observation dimension:
    obs_dim = 4 + 4*w + 5*6.
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    for w, expected_dim in [(20, 114), (30, 154), (40, 194)]:
        cfg_dict["num_tasks_per_vehicle_range"] = [w, w]
        cfg = SimulationConfig(**cfg_dict)
        env = VECEnv(config=cfg, port=9107 + (w // 10), seed=42)
        assert env.obs_dim == expected_dim
        assert env.observation_space.shape[0] == expected_dim
        env.close()


def test_06_physics_immutability_and_mismatch_detection():
    """
    Test 06 — Cross-Execution Physics Immutability and Hash Mismatch Detection
    Verify that comm_model.py and comp_model.py SHA-256 hashes are immutable,
    and any tampering is immediately detected.
    """
    expected_comm = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
    expected_comp = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"
    
    actual_comm = hashlib.sha256(open("envs/comm_model.py", "rb").read()).hexdigest()
    actual_comp = hashlib.sha256(open("envs/comp_model.py", "rb").read()).hexdigest()
    
    assert actual_comm == expected_comm, "comm_model.py hash altered!"
    assert actual_comp == expected_comp, "comp_model.py hash altered!"
