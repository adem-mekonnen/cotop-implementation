import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.data_loader import ApolloScapeTrajectoryDataset
from models.mobility_gat import MobilityGAT_GRU
import argparse

def create_fully_connected_edge_index(num_nodes, device):
    """Creates a fully connected graph for GAT based on the batch."""
    source_nodes = []
    target_nodes = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                source_nodes.append(i)
                target_nodes.append(j)
    if not source_nodes:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=device)

def train_mobility_model(args):
    # Colab Readiness: Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & DataLoader (90/10 Train-Validation Split)
    dataset = ApolloScapeTrajectoryDataset(csv_file=args.data_path, seq_len=5, pred_len=5)
    
    if len(dataset) == 0:
        print(f"Warning: Dataset is empty from {args.data_path}. Cannot start training.")
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Initialize the MobilityGAT_GRU model
    model = MobilityGAT_GRU(
        input_dim=2, 
        embed_dim=32, 
        num_heads=4, 
        gru_hidden=32, 
        output_dim=2
    ).to(device)
    
    # 3. Training Setup (Eq. 22 & Fig 4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # LR=0.0002 for stability
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for hist_seq, future_seq in train_loader:
            hist_seq, future_seq = hist_seq.to(device), future_seq.to(device)
            num_nodes = hist_seq.shape[0]
            edge_index = create_fully_connected_edge_index(num_nodes, device)
            
            optimizer.zero_grad()
            predictions = model(hist_seq, edge_index)
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
                edge_index = create_fully_connected_edge_index(num_nodes, device)
                
                predictions = model(hist_seq, edge_index)
                loss = criterion(predictions, future_seq)
                val_loss += loss.item() * hist_seq.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # REQUIREMENT: Print both Train and Validation MSE loss every epoch
        print(f"Epoch [{epoch+1:03d}/{args.epochs}] - Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
            
        # 5. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, "mobility_model_best.pth")
            torch.save(model.state_dict(), best_path)
            
    print(f"Training complete. Best model saved to: {os.path.join(args.save_dir, 'mobility_model_best.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/apolloscape.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()
    
    train_mobility_model(args)
