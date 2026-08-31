import pytest
import os
import copy
import yaml
from envs.vec_env import VECEnv
from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from utils.realization import generate_realization, save_realization, load_realization
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent
from utils.seed import set_seed
import torch
import numpy as np

@pytest.fixture
def base_config():
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    return SimulationConfig(**config_data)

def test_01_materialization_determinism(base_config):
    """Test 1: Materialization is deterministic."""
    env1 = VECEnv(config=base_config, port=9980, scenario_geometry="grid_200m", seed=42, max_vehicles=3)
    trace1 = generate_realization(env1)
    env1.close()
    
    env2 = VECEnv(config=base_config, port=9981, scenario_geometry="grid_200m", seed=42, max_vehicles=3)
    trace2 = generate_realization(env2)
    env2.close()
    
    # We strip timestamp and hash, they are unique per generation/save
    del trace1["creation_timestamp"]
    del trace2["creation_timestamp"]
    
    assert trace1 == trace2

def test_02_03_serialization_reloading_determinism(base_config, tmpdir):
    """Test 2 & 3: Serialization is deterministic, reloading produces identical data."""
    env = VECEnv(config=base_config, port=9982, scenario_geometry="grid_200m", seed=42, max_vehicles=3)
    trace = generate_realization(env)
    env.close()
    
    path1 = os.path.join(tmpdir, "real1.json")
    path2 = os.path.join(tmpdir, "real2.json")
    
    save_realization(trace, path1)
    save_realization(trace, path2)
    
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        assert f1.read() == f2.read(), "Serialization must be byte-identical"
        
    loaded = load_realization(path1)
    
    # Check reloading identity
    assert loaded["hash"] == trace["hash"]
    assert loaded["vehicle_trace"] == trace["vehicle_trace"]

def test_04_05_06_algorithmic_immutability_and_identical_exogenous_inputs(base_config, tmpdir):
    """Tests 4, 5, 6: CoTOP and DDQN cannot mutate the trace, and receive identical exogenous inputs."""
    env = VECEnv(config=base_config, port=9983, scenario_geometry="grid_200m", seed=42, max_vehicles=5)
    trace = generate_realization(env)
    env.close()
    
    realization_path = os.path.join(tmpdir, "real_frozen.json")
    save_realization(trace, realization_path)
    
    # Run CoTOP (A3C Agent)
    set_seed(42)
    frozen_env1 = FrozenVECEnv(config=base_config, realization_path=realization_path)
    agent_cotop = ActorCritic(frozen_env1.observation_space.shape[0], frozen_env1.action_space.n)
    
    obs, _ = frozen_env1.reset()
    done = False
    
    trace_cotop_tasks = []
    
    while not done:
        state = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent_cotop(state)
        mask = frozen_env1.get_action_mask()
        logits[0, ~torch.BoolTensor(mask)] = -1e9
        action = torch.argmax(logits, dim=-1).item()
        
        # Capture current task before stepping
        if len(frozen_env1.pending_tasks) > 0:
            trace_cotop_tasks.append(frozen_env1.pending_tasks[0][1].task_id)
            
        obs, _, terminated, _, _ = frozen_env1.step(action)
        done = terminated
        
    # Run DDQN
    set_seed(100) # Different internal seed for DDQN to prove exogenous trace is unaffected
    frozen_env2 = FrozenVECEnv(config=base_config, realization_path=realization_path)
    agent_ddqn = DDQNAgent(
        input_dim=frozen_env2.observation_space.shape[0],
        num_actions=frozen_env2.action_space.n
    )
    
    obs, _ = frozen_env2.reset()
    done = False
    
    trace_ddqn_tasks = []
    
    while not done:
        mask = frozen_env2.get_action_mask()
        action = agent_ddqn.select_action(obs, mask)
        
        if len(frozen_env2.pending_tasks) > 0:
            trace_ddqn_tasks.append(frozen_env2.pending_tasks[0][1].task_id)
            
        obs, _, terminated, _, _ = frozen_env2.step(action)
        done = terminated
        
    # Check that they received identical exogenous tasks
    assert trace_cotop_tasks == trace_ddqn_tasks, "Algorithms must receive exactly identical exogenous task traces"
    
    # Verify no mutation to realization file occurred
    loaded_post = load_realization(realization_path)
    assert loaded_post["hash"] == trace["hash"], "Algorithms must not mutate realization"
