def get_rsu_positions(num_rsus: int, traci_conn=None):
    """
    Query the SUMO map boundary using traci and place RSUs evenly 
    along the main traffic corridor (e.g., every 400m).
    """
    try:
        if traci_conn is not None:
            boundary = traci_conn.simulation.getNetBoundary()
        else:
            import traci
            boundary = traci.simulation.getNetBoundary()
            
        xmin, ymin = boundary[0]
        xmax, ymax = boundary[1]
    except Exception as e:
        print(f"Failed to get network boundary from SUMO: {e}")
        # Fallback to the 2400m span mentioned in the paper
        xmin, xmax = 0, 2400
        ymin, ymax = 0, 0

    # Place evenly along the main corridor (x-axis)
    y_pos = (ymin + ymax) / 2.0
    
    # For a 2400m span, we want 6 RSUs every 400m. Center them within their 400m blocks.
    # Half of 400m is 200m padding.
    padding = (xmax - xmin) / (num_rsus * 2) if num_rsus > 0 else 0
    start_x = xmin + padding
    end_x = xmax - padding
    
    if num_rsus > 1:
        step = (end_x - start_x) / (num_rsus - 1)
    else:
        step = 0
        
    positions = []
    for i in range(num_rsus):
        positions.append((float(start_x + i * step), float(y_pos)))
        
    return positions
