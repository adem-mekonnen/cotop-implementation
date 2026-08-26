import pytest
import math
from envs.comm_model import compute_v2r_rate, compute_r2r_rate

def test_v2r_rate_known_values():
    dist = 100.0
    bw = 20.0e6
    power_v = 0.01
    noise = 0.001
    k = 1000.0
    sigma = 2.0
    
    # SINR = (0.01 * 1000) / (0.001 * 10000) = 1.0
    # Rate = 20e6 * log2(2) = 20e6
    rate = compute_v2r_rate(dist, bw, power_v, noise, k, sigma)
    assert pytest.approx(rate, rel=1e-5) == 20.0e6

def test_r2r_rate_known_values():
    dist = 400.0
    bw = 50.0e6
    power_r = 100.0
    noise = 0.001
    k = 1000.0
    sigma = 2.0
    
    # SINR = (100 * 1000) / (0.001 * 160000) = 625.0
    # Rate = 50e6 * log2(626)
    expected_rate = 50.0e6 * math.log2(626.0)
    rate = compute_r2r_rate(dist, bw, power_r, noise, k, sigma)
    assert pytest.approx(rate, rel=1e-5) == expected_rate

def test_comm_zero_distance_safety():
    # Avoid division by zero when vehicle is directly at RSU location
    rate = compute_v2r_rate(0.0, 20.0e6, 0.01, 0.001, 1000.0, 2.0)
    assert rate > 0
    assert not math.isnan(rate)
    assert not math.isinf(rate)
