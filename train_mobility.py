import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Internal Imports
from utils.data_loader import ApolloScapeTrajectoryDataset
from models.mobility_gat import MobilityGAT_GRU
from utils.seed import set_seed

def get_proximity_edge_index(positions, radius=0.083333, device='cpu'):
    """
    Constructs a spatial graph based on Euclidean distance between vehicles (nodes).
    Avoids degenerate self-loop-only graphs.
    Default radius: 200.0m / 2400.0m = 0.083333.
    """
    num_nodes = positions.shape[0]
    if num_nodes <= 1:
        # Single vehicle: self-loop so GAT can process
        return torch.tensor([[0], [0]], dtype=torch.long, device=device)
    
    # Compute pairwise Euclidean distance at the last historical frame
    last_pos = positions[:, -1, :].cpu().numpy() # (N, 2)
    diff = last_pos[:, np.newaxis, :] - last_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    adj = (dist <= radius).astype(int)
    # Include self-loops
    np.fill_diagonal(adj, 1)
    
    edge_index = np.argwhere(adj == 1).T
    return torch.tensor(edge_index, dtype=torch.long, device=device)

def train_mobility_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training Mobility GAT-GRU on: {device} (Mode: {args.mode})")
    
    data_dir = args.data_path
    if args.mode == "synthetic" or not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        data_dir = "data/raw/synthetic"
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            from utils.synthetic_trajectories import generate_synthetic_trajectory_data
            generate_synthetic_trajectory_data(data_dir, num_vehicles=100, num_frames=50, seed=args.seed)
            
    print(f"Loading trajectory data from: {data_dir}")
    dataset = ApolloScapeTrajectoryDataset(data_dir=data_dir, seq_len=5, pred_len=5)
    
    if len(dataset) == 0:
        print(f"[ERROR] No sequences found in {data_dir}.")
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Architecture from Section IV-B and Table II
    model = MobilityGAT_GRU(
        input_dim=2, 
        embed_dim=64,   # 64 dims
        num_heads=4,    # 4 attention heads
        gru_hidden=64, 
        output_dim=2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # LR=0.0002 per Section V-C
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for hist_seq, future_seq in train_loader:
            hist_seq, future_seq = hist_seq.to(device), future_seq.to(device)
            edge_index = get_proximity_edge_index(hist_seq, radius=200.0, device=device)
            
            optimizer.zero_grad()
            predictions = model(hist_seq, edge_index)
            loss = criterion(predictions, future_seq)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * hist_seq.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist_seq, future_seq in val_loader:
                hist_seq, future_seq = hist_seq.to(device), future_seq.to(device)
                edge_index = get_proximity_edge_index(hist_seq, radius=200.0, device=device)
                predictions = model(hist_seq, edge_index)
                loss = criterion(predictions, future_seq)
                val_loss += loss.item() * hist_seq.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch [{epoch+1:03d}/{args.epochs}] - Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "mobility_model.pth"))
            
    print(f"[SUCCESS] Mobility training complete. Model saved to: {os.path.join(args.save_dir, 'mobility_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["synthetic", "apolloscape"], default="synthetic")
    parser.add_argument("--data_path", type=str, default="data/raw/apolloscape")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()
    
    train_mobility_model(args)