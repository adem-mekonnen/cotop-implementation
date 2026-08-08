import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ApolloScapeTrajectoryDataset(Dataset):
    """
    Parser for the ApolloScape trajectory dataset.
    Extracts (x, y) coordinate sequences for vehicles.
    """
    def __init__(self, csv_file: str, seq_len: int = 5, pred_len: int = 5):
        """
        Args:
            csv_file (str): Path to the ApolloScape trajectory data file.
            seq_len (int): Number of historical time steps.
            pred_len (int): Number of future time steps to predict.
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        
        # Load and parse the dataset upon initialization
        self.trajectories = self._parse_dataset(csv_file)
        
    def _parse_dataset(self, csv_file: str):
        """
        Reads ApolloScape data and groups (x, y) coordinates by vehicle/object ID.
        Assumes standard format roughly matching:
        [frame_id, object_id, object_type, position_x, position_y, ...]
        """
        try:
            # Read CSV assuming space or comma separated; using simple read_csv here
            df = pd.read_csv(csv_file, header=None)
            
            # Assigning typical column names for ApolloScape
            cols = ['frame_id', 'object_id', 'object_type', 'pos_x', 'pos_y', 'pos_z', 
                    'length', 'width', 'height', 'heading']
            df.columns = cols[:len(df.columns)]
            
            # Sort by object_id and frame_id to ensure sequential chronological order
            df = df.sort_values(by=['object_id', 'frame_id'])
            
            trajectories = []
            grouped = df.groupby('object_id')
            
            for object_id, group in grouped:
                # Extract just the (x, y) coordinates sequence for this object
                coords = group[['pos_x', 'pos_y']].values
                
                # Split into fixed-length rolling windows (historical + future)
                for i in range(len(coords) - self.total_len + 1):
                    sequence = coords[i:i+self.total_len]
                    trajectories.append(sequence)
                    
            return np.array(trajectories)
            
        except Exception as e:
            print(f"Error parsing ApolloScape dataset from {csv_file}: {e}")
            return np.array([])

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Returns a tuple of (historical_sequence, future_sequence)
        Shape: (seq_len, 2), (pred_len, 2)
        """
        seq = self.trajectories[idx]
        
        hist_seq = seq[:self.seq_len]
        future_seq = seq[self.seq_len:]
        
        return torch.tensor(hist_seq, dtype=torch.float32), torch.tensor(future_seq, dtype=torch.float32)
