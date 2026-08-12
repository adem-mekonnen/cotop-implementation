import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np

# Internal Imports
from utils.data_loader import ApolloScapeTrajectoryDataset
from models.mobility_gat import MobilityGAT_GRU

def get_fc_edge_index(num_nodes, device):
    """
    Vectorized creation of a fully connected edge index.
    Avoids slow Python loops.
    """
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Create all-to-all connectivity
    adj = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    edge_index = np.argwhere(adj == 1).T
    return torch.tensor(edge_index, dtype=torch.long, device=device)

def train_mobility_model(args):
    # Colab Readiness: Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training starting on: {device}")
    
    # 1. Dataset & DataLoader (90/10 Train-Validation Split)
    # Changed 'csv_file' to 'data_dir' to match our robust DataLoader
    dataset = ApolloScapeTrajectoryDataset(data_dir=args.data_path, seq_len=5, pred_len=5)
    
    if len(dataset) == 0:
        print(f"❌ Error: No sequences found in {args.data_path}. check your dataset paths.")
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Important: We set batch_size=1 here because each sample in our 
    # DataLoader is already a window of data. GAT will process 'N' nodes 
    # if we structure the batch correctly. 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Initialize Model (Architecture from Sec IV-B)
    model = MobilityGAT_GRU(
        input_dim=2, 
        embed_dim=64,   # Paper uses 64 (Eq 15-18)
        num_heads=4,    # Paper uses 4 heads
        gru_hidden=64, 
        output_dim=2
    ).to(device)
    
    # 3. Training Setup (Eq. 22 & Fig 4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # LR=0.0002 per Fig 4
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for hist_seq, future_seq in train_loader:
            # Move to device. Shape: (Batch, Seq_len, 2)
            hist_seq, future_seq = hist_seq.to(device), future_seq.to(device)
            
            # For trajectory datasets, we treat the batch as nodes in a single graph
            num_nodes = hist_seq.shape[0]
            edge_index = get_fc_edge_index(num_nodes, device)
            
            optimizer.zero_grad()
            
            # Forward pass (Eq 15-21)
            predictions = model(hist_seq, edge_index)
            
            # Loss calculation (Eq 22)
            loss = criterion(predictions, future_seq)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * hist_seq.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist_seq, future_seq in val_loader:
                hist_seq, future_seq = hist_seq.to(device), future_seq.to(device)
                num_nodes = hist_seq.shape[0]
                edge_index = get_fc_edge_index(num_nodes, device)
                
                predictions = model(hist_seq, edge_index)
                loss = criterion(predictions, future_seq)
                val_loss += loss.item() * hist_seq.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Log results
        print(f"Epoch [{epoch+1:03d}/{args.epochs}] - Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
            
        # 5. Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "mobility_model.pth"))
            
    print(f"✅ Training complete. Best model saved to: results/checkpoints/mobility_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default to point to the folder structure we created
    parser.add_argument("--data_path", type=str, default="/content/dataset/raw/train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()
    
    train_mobility_model(args)