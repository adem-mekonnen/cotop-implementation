import pytest
import math

def calculate_step_reward(total_delay, total_energy, max_delay, epsilon=0.5, penalty_z=100.0):
    # Eq. 25:
    # r(t) = -(epsilon * T_total + (1 - epsilon) * E_total) if T_total <= d
    # r(t) = -Z if T_total > d
    if total_delay > max_delay:
        return -penalty_z
    else:
        return -(epsilon * total_delay + (1.0 - epsilon) * total_energy)

def test_reward_function_within_deadline():
    delay = 1.5
    energy = 2.0
    deadline = 5.0
    reward = calculate_step_reward(delay, energy, deadline, epsilon=0.5, penalty_z=100.0)
    # -(0.5 * 1.5 + 0.5 * 2.0) = -(0.75 + 1.0) = -1.75
    assert pytest.approx(reward, rel=1e-5) == -1.75

def test_reward_function_exceeded_deadline():
    delay = 6.0
    energy = 2.0
    deadline = 5.0
    reward = calculate_step_reward(delay, energy, deadline, epsilon=0.5, penalty_z=100.0)
    # Exceeded deadline -> penalty -Z
    assert reward == -100.0
