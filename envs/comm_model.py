import math
from typing import Tuple

def get_euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    """
    Helper function to calculate the Euclidean distance between two positions.
    """
    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

def compute_v2r_rate(
    pos_v: Tuple[float, float], 
    pos_r: Tuple[float, float], 
    bandwidth_B: float, 
    power_P_V: float, 
    noise_power_sigma2: float,
    path_loss_exponent: float = 2.0
) -> float:
    """
    Calculates w_{n,m}^{V2R} (Eq 1): Data transmission rate from Vehicle n to RSU m.
    Using Shannon's Formula based on vehicle-to-RSU distance.
    """
    distance = get_euclidean_distance(pos_v, pos_r)
    # Ensure distance is not zero to avoid division by zero in channel gain
    distance = max(distance, 1.0)
    
    # H_{n,m}: Channel gain between Vehicle n and RSU m
    channel_gain_H_nm = 1.0 / (distance ** path_loss_exponent)
    
    # w_{n,m}^{V2R} = B * log2(1 + (P_V * H_{n,m}) / sigma^2)
    sinr = (power_P_V * channel_gain_H_nm) / noise_power_sigma2
    return bandwidth_B * math.log2(1.0 + sinr)

def compute_r2r_rate(
    pos_r1: Tuple[float, float], 
    pos_r2: Tuple[float, float], 
    bandwidth_B: float, 
    power_P_R: float, 
    noise_power_sigma2: float,
    path_loss_exponent: float = 2.0
) -> float:
    """
    Calculates w_{m,m'}^{R2R} (Eq 2): Data transmission rate from RSU m to RSU m'.
    Using Shannon's Formula based on RSU-to-RSU distance.
    """
    distance = get_euclidean_distance(pos_r1, pos_r2)
    # Ensure distance is not zero to avoid division by zero in channel gain
    distance = max(distance, 1.0)
    
    # H_{m,m'}: Channel gain between RSU m and RSU m'
    channel_gain_H_mm = 1.0 / (distance ** path_loss_exponent)
    
    # w_{m,m'}^{R2R} = B * log2(1 + (P_R * H_{m,m'}) / sigma^2)
    sinr = (power_P_R * channel_gain_H_mm) / noise_power_sigma2
    return bandwidth_B * math.log2(1.0 + sinr)
