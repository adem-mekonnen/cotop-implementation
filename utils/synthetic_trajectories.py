"""
Synthetic trajectory generation for GAT-GRU mobility model debugging and training.
Generates realistic vehicle kinematics (positions, speeds, headings) on a multi-lane corridor.
"""
import os
import numpy as np
import pandas as pd

def generate_synthetic_trajectory_data(output_dir: str = "data/raw/synthetic", num_vehicles: int = 100, num_frames: int = 50, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    rows = []
    for v_id in range(num_vehicles):
        # Initial state on 2400m road corridor
        x = np.random.uniform(0.0, 1800.0)
        y = np.random.uniform(-10.0, 10.0) # Lane offsets
        speed = np.random.uniform(30.0, 40.0) # Paper speed: 30-40 m/s
        dt = 0.5 # 0.5s per frame
        
        for frame in range(num_frames):
            # ApolloScape format columns:
            # frame_id, object_id, object_type (1: car), pos_x, pos_y, pos_z, length, width, height, heading
            noise_x = np.random.normal(0, 0.2)
            noise_y = np.random.normal(0, 0.05)
            curr_x = x + speed * dt * frame + noise_x
            curr_y = y + noise_y
            
            rows.append([
                frame,
                v_id,
                1, # Small vehicle / car
                curr_x,
                curr_y,
                0.0,
                4.5, 1.8, 1.5,
                0.0
            ])
            
    df = pd.DataFrame(rows, columns=[
        'frame_id', 'object_id', 'object_type', 'pos_x', 'pos_y', 'pos_z',
        'length', 'width', 'height', 'heading'
    ])
    
    out_file = os.path.join(output_dir, "synthetic_trajectories.txt")
    df.to_csv(out_file, sep=" ", index=False, header=False)
    print(f"Generated {len(df)} synthetic trajectory points in {out_file}")
    return out_file

if __name__ == "__main__":
    generate_synthetic_trajectory_data()
