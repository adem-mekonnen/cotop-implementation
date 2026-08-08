import math
from typing import Tuple

def get_euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    """Helper function to calculate the Euclidean distance between two positions."""
    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

def compute_v2r_rate(
    pos_v: Tuple[float, float], 
    pos_r: Tuple[float, float], 
    bandwidth_B: float, 
    power_P_V: float, 
    noise_power: float,
    fixed_loss_k: float,
    path_loss_factor: float = 2.0
) -> float:
    """
    Calculates w_{n,m}^{V2R} (Eq 1): Data transmission rate from Vehicle n to RSU m.
    Using Shannon's Formula: rate = B * log2(1 + (P * K) / (omega * D^sigma))
    """
    distance = get_euclidean_distance(pos_v, pos_r)
    distance = max(distance, 1.0) # Avoid division by zero
    
    sinr = (power_P_V * fixed_loss_k) / (noise_power * (distance ** path_loss_factor))
    return bandwidth_B * math.log2(1.0 + sinr)

def compute_r2r_rate(
    pos_r1: Tuple[float, float], 
    pos_r2: Tuple[float, float], 
    bandwidth_B: float, 
    power_P_R: float, 
    noise_power: float,
    fixed_loss_k: float,
    path_loss_factor: float = 2.0
) -> float:
    """
    Calculates w_{m,m'}^{R2R} (Eq 2): Data transmission rate from RSU m to RSU m'.
    Using Shannon's Formula: rate = B * log2(1 + (P * K) / (omega * D^sigma))
    Based on the distance between RSU locations.
    """
    distance = get_euclidean_distance(pos_r1, pos_r2)
    distance = max(distance, 1.0) # Avoid division by zero
    
    sinr = (power_P_R * fixed_loss_k) / (noise_power * (distance ** path_loss_factor))
    return bandwidth_B * math.log2(1.0 + sinr)
