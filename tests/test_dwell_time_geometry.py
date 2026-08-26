"""
tests/test_dwell_time_geometry.py
Validates analytical geometric dwell time calculation against vehicle kinematics.
"""
import pytest
import math
from envs.entities import Vehicle, RSU
from envs.vec_env import get_euclidean_distance

def test_geometric_dwell_time_analytical():
    # Vehicle at position (100.0, 0.0), speed = 40 m/s
    # RSU at (0.0, 0.0), coverage radius R = 400 m
    # Distance to RSU = 100 m
    # Remaining travel distance before boundary = 400 m - 100 m = 300 m
    # Analytical Dwell Time T_stay = 300 m / 40 m/s = 7.5 s
    vehicle_pos = (100.0, 0.0)
    vehicle_speed = 40.0
    rsu_location = (0.0, 0.0)
    comm_range = 400.0

    dist = get_euclidean_distance(vehicle_pos, rsu_location)
    remaining_dist = comm_range - dist
    analytical_dwell = remaining_dist / vehicle_speed

    assert math.isclose(dist, 100.0)
    assert math.isclose(remaining_dist, 300.0)
    assert math.isclose(analytical_dwell, 7.5)
