import pytest
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy
from envs.entities import SimulationConfig
import numpy as np
import yaml

def test_local_policy():
    policy = LocalPolicy()
    state = np.zeros(114, dtype=np.float32)
    # Local policy must always choose action 0 (Case 1 Standalone)
    action = policy.select_action(state)
    assert action == 0

def test_greedy_policy():
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = SimulationConfig(**cfg_dict)
    policy = GreedyPolicy(config=config)
    
    # Create a dummy normalized state:
    # Vehicle at x=100m, y=0m -> (100 / 2400) = 0.04166
    # 20 tasks (80 dims)
    # 6 RSUs: RSU 0 at x=200m (nearest), RSU 1 at x=600m
    # Set RSU 0 queue high, RSU 1 queue 0
    state = np.zeros(4 + 80 + 30, dtype=np.float32)
    state[0] = 100.0 / 2400.0 # Vehicle x
    
    rsu_start = 84
    # RSU 0: at x=200m, high queue = 1.0 (100%)
    state[rsu_start + 0] = 200.0 / 2400.0
    state[rsu_start + 2] = 1.0 # max CPU
    state[rsu_start + 3] = 1.0 # 100% queue
    
    # RSU 1: at x=600m, empty queue = 0.0
    state[rsu_start + 5] = 600.0 / 2400.0
    state[rsu_start + 7] = 1.0 # max CPU
    state[rsu_start + 8] = 0.0 # empty queue
    
    action = policy.select_action(state)
    # Greedy should pick RSU 1 for collaboration, which maps to action = 1 + 1 = 2
    assert action == 2
