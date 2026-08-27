from typing import List, Tuple, Optional

def get_rsu_positions(num_rsus: int = 6, traci_conn=None, scenario_mode: str = "auto") -> List[Tuple[float, float]]:
    """
    Computes RSU deployment coordinates for SUMO simulation environments.
    
    Supports:
    1. 'corridor_2400m': 1D arterial corridor with 6 evenly spaced RSUs along the x-axis.
    2. 'grid_200m': 2D Paper-Consistent Reconstructed Scenario (200m x 200m) with 6 RSUs
       deployed at strategic arterial intersections / coverage centroids.
    3. 'auto': Queries traci simulation boundary to select appropriate placement.
    """
    if scenario_mode == "grid_200m":
        xmin, xmax = 0.0, 200.0
        ymin, ymax = 0.0, 200.0
    else:
        xmin, xmax = 0.0, 2400.0
        ymin, ymax = 0.0, 0.0
    
    try:
        if traci_conn is not None:
            boundary = traci_conn.simulation.getNetBoundary()
            xmin, ymin = boundary[0]
            xmax, ymax = boundary[1]
        else:
            import traci
            boundary = traci.simulation.getNetBoundary()
            xmin, ymin = boundary[0]
            xmax, ymax = boundary[1]
    except Exception:
        pass

    width = xmax - xmin
    height = ymax - ymin

    # Determine mode
    if scenario_mode == "grid_200m" or (scenario_mode == "auto" and width <= 300.0 and height > 50.0):
        # 2D 200m x 200m Reconstructed Scenario: 6 RSUs in 2x3 coverage grid
        # Coordinates cover (50, 50), (100, 50), (150, 50), (50, 150), (100, 150), (150, 150)
        positions = [
            (float(xmin + width * 0.25), float(ymin + height * 0.25)), # (50, 50)
            (float(xmin + width * 0.50), float(ymin + height * 0.25)), # (100, 50)
            (float(xmin + width * 0.75), float(ymin + height * 0.25)), # (150, 50)
            (float(xmin + width * 0.25), float(ymin + height * 0.75)), # (50, 150)
            (float(xmin + width * 0.50), float(ymin + height * 0.75)), # (100, 150)
            (float(xmin + width * 0.75), float(ymin + height * 0.75)), # (150, 150)
        ]
        return positions[:num_rsus]

    # Historical 1D corridor placement (e.g. 2400m span)
    y_pos = (ymin + ymax) / 2.0
    padding = width / (num_rsus * 2) if num_rsus > 0 else 0.0
    start_x = xmin + padding
    end_x = xmax - padding
    
    step = (end_x - start_x) / (num_rsus - 1) if num_rsus > 1 else 0.0
    positions = []
    for i in range(num_rsus):
        positions.append((float(start_x + i * step), float(y_pos)))
        
    return positions
