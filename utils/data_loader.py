import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class ApolloScapeTrajectoryDataset(Dataset):
    """
    Parser for the ApolloScape trajectory dataset (or SUMO-generated trajectories).
    Reads all .txt files in a directory and extracts (x, y) rolling windows.
    """
    def __init__(self, data_dir: str, seq_len: int = 5, pred_len: int = 5):
        """
        Args:
            data_dir (str): Directory containing the trajectory .txt files.
            seq_len (int): History window (input).
            pred_len (int): Prediction window (ground truth).
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.trajectories = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        # 1. Iterate through all .txt files in the folder
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        if len(files) == 0:
            print(f"Warning: No .txt files found in {data_dir}")

        for file_path in files:
            self._parse_file(file_path)
            
        # Convert list of windows to a single numpy array
        self.trajectories = np.array(self.trajectories)
        print(f"Loaded {len(self.trajectories)} sequences from {len(files)} files.")

    def _parse_file(self, file_path: str):
        try:
            # 2. Use sep=r'\s+' because ApolloScape is space-separated
            df = pd.read_csv(file_path, sep=r'\s+', header=None)
            
            # Map columns per ApolloScape format
            cols = ['frame_id', 'object_id', 'object_type', 'pos_x', 'pos_y', 'pos_z', 
                    'length', 'width', 'height', 'heading']
            df.columns = cols[:len(df.columns)]
            
            # 3. Filter for vehicles only (Type 1: small vehicles, Type 2: large vehicles)
            # This ensures we don't train on pedestrians or cyclists
            df = df[df['object_type'].isin([1, 2])]
            
            # Sort to ensure chronological order for rolling windows
            df = df.sort_values(by=['object_id', 'frame_id'])
            
            grouped = df.groupby('object_id')
            for _, group in grouped:
                coords = group[['pos_x', 'pos_y']].values
                
                # 4. Check if sequence is long enough
                if len(coords) < self.total_len:
                    continue
                
                # Rolling window windowing
                for i in range(len(coords) - self.total_len + 1):
                    window = coords[i : i + self.total_len]
                    self.trajectories.append(window)
                    
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Returns: (history_tensor, future_tensor)
        Shapes: (seq_len, 2), (pred_len, 2)
        """
        # (total_len, 2)
        seq = self.trajectories[idx]
        
        # Split into input and target
        hist_seq = seq[:self.seq_len]
        future_seq = seq[self.seq_len:]
        
        return (torch.tensor(hist_seq, dtype=torch.float32), 
                torch.tensor(future_seq, dtype=torch.float32))