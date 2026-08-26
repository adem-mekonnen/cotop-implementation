"""
tests/test_mobility_coordinate_consistency.py
Validates that trajectory coordinates, normalization, model inference, and denormalization preserve physical scale.
"""
import pytest
import torch
import numpy as np
from models.mobility_gat import MobilityGAT_GRU
from utils.data_loader import ApolloScapeTrajectoryDataset

def test_mobility_normalization_and_scale():
    dataset = ApolloScapeTrajectoryDataset(data_dir="data/raw/synthetic", seq_len=5, pred_len=5, norm_scale=2400.0)
    assert len(dataset) > 0
    
    hist_norm, future_norm = dataset[0]
    
    # 1. Normalized values must be strictly in [0, 1] range for corridor coordinates
    assert torch.all(hist_norm >= -0.05) and torch.all(hist_norm <= 1.05)
    assert torch.all(future_norm >= -0.05) and torch.all(future_norm <= 1.05)
    
    # 2. Denormalization restores raw physical meters
    raw_hist = hist_norm.numpy() * 2400.0
    assert np.all(raw_hist[:, 0] >= 0.0) and np.all(raw_hist[:, 0] <= 2400.0)

def test_mobility_inference_coordinate_scale():
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    model.load_state_dict(torch.load("results/checkpoints/mobility_model.pth", map_location="cpu"))
    model.eval()

    # Input: Vehicle moving at (500m, 0m) -> normalized = (500/2400, 0)
    traj_norm = np.array([[500.0 + i*17.5, 0.0] for i in range(5)], dtype=np.float32) / 2400.0
    x_seq = torch.FloatTensor(traj_norm).unsqueeze(0)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    with torch.no_grad():
        pred_norm = model(x_seq, edge_index)

    assert pred_norm.shape == (1, 5, 2)
    pred_meters = pred_norm[0].numpy() * 2400.0

    # The predicted position after 5 steps should be forward along the road corridor
    assert pred_meters[-1, 0] > 500.0
    assert pred_meters[-1, 0] < 2400.0
