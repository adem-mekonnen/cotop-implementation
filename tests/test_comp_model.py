import pytest
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration

def test_case1_standalone():
    task_size = 2.0e6      # 2 MB -> 16 Mbits
    task_cpu = 10.0e6      # 10 Mcycles
    w_v2r = 20.0e6         # 20 Mbps -> t_trans = 0.8 s
    rsu_cpu = 1.0e9        # 1 GHz -> t_comp = 0.01 s
    power_v = 0.01
    compute_power = 50.0
    t_wait = 0.05
    
    # total_delay = 0.8 + 0.01 + 0.05 = 0.86 s
    # energy = (0.01 * 0.8) + (0.01 * 50.0) = 0.008 + 0.5 = 0.508 J
    delay, energy = calculate_case1_standalone(
        task_size, task_cpu, w_v2r, rsu_cpu, power_v, compute_power, t_wait
    )
    assert pytest.approx(delay, rel=1e-5) == 0.86
    assert pytest.approx(energy, rel=1e-5) == 0.508

def test_case2_collaboration_parallel():
    # Validates parallel collaborative offloading execution
    task_size = 2.0e6      # 2 MB -> 16 Mbits
    task_cpu = 10.0e6      # 10 Mcycles
    w_v2r = 20.0e6         # 20 Mbps -> t_v2r = 0.8 s
    w_r2r = 100.0e6        # 100 Mbps
    rsu1_cpu = 1.0e9       # 1 GHz
    rsu2_cpu = 2.0e9       # 2 GHz
    dwell_time = 1.0       # 1.0s dwell time
    
    delay, energy = calculate_case2_collaboration(
        task_size_rho=task_size,
        task_cpu_phi=task_cpu,
        w_v2r=w_v2r,
        w_r2r=w_r2r,
        rsu1_cpu_f=rsu1_cpu,
        rsu2_cpu_f=rsu2_cpu,
        t1_dwell_time=dwell_time,
        power_v=0.01,
        tx_power_rsu1=100.0,
        compute_power_rsu1=50.0,
        compute_power_rsu2=50.0,
        t_wait=0.0
    )
    # Analytical: t_v2r = 0.8s, t1 = 0.003333s, t2 = 0.106667s, t3 = 0.003333s
    # max(t1, t2 + t3) = 0.11s -> delay = 0.91s
    assert pytest.approx(delay, rel=1e-4) == 0.91
    assert pytest.approx(energy, rel=1e-3) == 11.008
