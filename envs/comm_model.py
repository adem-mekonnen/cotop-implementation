import math

def compute_v2r_rate(
    distance: float, 
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
    distance = max(distance, 1.0) # Avoid division by zero
    
    sinr = (power_P_V * fixed_loss_k) / (noise_power * (distance ** path_loss_factor))
    return bandwidth_B * math.log2(1.0 + sinr)

def compute_r2r_rate(
    distance: float, 
    bandwidth_B: float, 
    power_P_R: float, 
    noise_power: float,
    fixed_loss_k: float,
    path_loss_factor: float = 2.0
) -> float:
    """
    Calculates w_{m,m'}^{R2R} (Eq 2): Data transmission rate from RSU m to RSU m'.
    Using Shannon's Formula: rate = B * log2(1 + (P * K) / (omega * D^sigma))
    """
    distance = max(distance, 1.0) # Avoid division by zero
    
    sinr = (power_P_R * fixed_loss_k) / (noise_power * (distance ** path_loss_factor))
    return bandwidth_B * math.log2(1.0 + sinr)
