import pytest
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration

def test_energy_model_strictness():
    # Verify that RF transmission power and computation power are never conflated
    task_size = 1.0e6      # 1 MB -> 8 Mbits
    task_cpu = 20.0e6      # 20 Mcycles
    w_v2r = 8.0e6          # 8 Mbps -> t_trans = 1.0 s
    rsu_cpu = 1.0e9        # 1 GHz -> t_comp = 0.02 s
    power_v = 0.05         # 50 mW
    compute_power = 80.0   # 80 W
    
    # E_trans = 0.05 * 1.0 = 0.05 J
    # E_comp = 0.02 * 80.0 = 1.6 J
    # Total = 1.65 J
    _, energy = calculate_case1_standalone(
        task_size, task_cpu, w_v2r, rsu_cpu, power_v, compute_power, t_wait=0.0
    )
    assert pytest.approx(energy, rel=1e-5) == 1.65
