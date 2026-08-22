import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class MobilityGAT_GRU(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2):
        super(MobilityGAT_GRU, self).__init__()
        
        # Eq. 15: Expansion MLP
        self.coordinate_expansion_mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Eq. 16-18: Spatial GAT — 2 stacked layers (Fig. 3)
        # Layer 1: embed_dim → embed_dim (concat, heads=4)
        self.gat_layer1 = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim // num_heads,
            heads=num_heads,
            concat=True
        )
        # Layer 2: embed_dim → embed_dim (concat, heads=4)
        self.gat_layer2 = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim // num_heads,
            heads=num_heads,
            concat=True
        )
        
        # Eq. 19-21: Temporal GRU
        self.encoder_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        self.decoder_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        
        self.output_layer = nn.Linear(gru_hidden, output_dim)

    def forward(self, x_seq, edge_index):
        """
        x_seq: (num_nodes, seq_len, 2)
        edge_index: (2, num_edges) proximity-based graph
        """
        num_nodes, seq_len, _ = x_seq.shape
        device = x_seq.device
        
        # --- SPATIAL FEATURE EXTRACTION ---
        spatial_features = []
        for t in range(seq_len):
            h_t  = self.coordinate_expansion_mlp(x_seq[:, t, :])  # Eq 15
            z_t1 = F.relu(self.gat_layer1(h_t,  edge_index))       # GAT layer 1 (Fig. 3)
            z_t2 = F.relu(self.gat_layer2(z_t1, edge_index))       # GAT layer 2 (Fig. 3)
            spatial_features.append(z_t2)
            
        spatial_features = torch.stack(spatial_features, dim=1) # (N, T, embed)
        
        # --- ENCODING ---
        _, encoder_hidden = self.encoder_gru(spatial_features)
        
        # --- DECODING (Prediction) ---
        # Start with the last known spatial feature as first decoder input
        decoder_input = spatial_features[:, -1, :].unsqueeze(1) 
        decoder_hidden = encoder_hidden
        
        predictions = []
        for t in range(seq_len): # Predict next 'seq_len' steps
            out, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            pred_pos = self.output_layer(out.squeeze(1))
            predictions.append(pred_pos)
            
            # Autoregressive feedback: Expand pred and feed back in
            decoder_input = self.coordinate_expansion_mlp(pred_pos).unsqueeze(1)
            
        return torch.stack(predictions, dim=1) # (N, T_pred, 2)

# --- QUICK SANITY CHECK ---
if __name__ == "__main__":
    model = MobilityGAT_GRU()
    # Simulate 5 vehicles, 5 frames of history
    dummy_x = torch.randn(5, 5, 2) 
    # Fully connected graph for 5 nodes
    dummy_edge_index = torch.tensor([[0,1,1,2,3,4], [1,0,2,1,4,3]], dtype=torch.long)
    
    out = model(dummy_x, dummy_edge_index)
    print(f"Output Shape: {out.shape}") # Expected: (5, 5, 2)
    assert out.shape == (5, 5, 2)
    print("✅ Model logic verified.")