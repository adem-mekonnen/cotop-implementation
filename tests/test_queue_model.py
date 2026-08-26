import pytest
from envs.entities import RSU

def test_queue_wait_time_and_depletion():
    rsu = RSU(
        rsu_id=0,
        location=(0.0, 0.0),
        cpu_capacity_f=1.0e9,      # 1 GHz
        queued_cpu_cycles=2.0e9,   # 2 Gcycles queued
        transmission_power_P_R=100.0
    )
    
    # Eq. 5: t_wait = N^{queue} / F^{RSU} = 2.0e9 / 1.0e9 = 2.0 seconds
    t_wait = rsu.queued_cpu_cycles / rsu.cpu_capacity_f
    assert pytest.approx(t_wait, rel=1e-5) == 2.0
    
    # Simulate 1.5 seconds of processing
    dt = 1.5
    rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * dt)
    assert pytest.approx(rsu.queued_cpu_cycles, rel=1e-5) == 0.5e9
    
    # Simulate another 1.0 second (depleting queue to 0)
    rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * 1.0)
    assert rsu.queued_cpu_cycles == 0.0
