import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class MobilityGAT_GRU(nn.Module):
    """
    Spatial-Temporal Graph Attention Mobility Predictor (Eq. 15–21).
    
    Spatial Feature Extraction:
    - Layer 1 (Eq. 17): Multi-head GAT with concatenation (heads=4, out_dim=16 -> 64)
    - Layer 2 (Eq. 18): Multi-head GAT with head averaging (heads=4, concat=False, out_dim=64)
    
    Temporal Processing:
    - Eq. 19-21: GRU Encoder-Decoder for autoregressive trajectory forecasting.
    """
    def __init__(self, input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2):
        super(MobilityGAT_GRU, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Eq. 15: Expansion MLP
        self.coordinate_expansion_mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Eq. 16-17: Spatial GAT Layer 1 (Multi-head Concatenation)
        self.gat_layer1 = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim // num_heads,
            heads=num_heads,
            concat=True
        )
        
        # Eq. 18: Spatial GAT Layer 2 (Multi-head Averaging across heads)
        # Operational implementation mapping of Eq. 18 mean head aggregation
        self.gat_layer2 = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            heads=num_heads,
            concat=False
        )
        
        # Eq. 19-21: Temporal GRU
        self.encoder_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        self.decoder_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        
        self.output_layer = nn.Linear(gru_hidden, output_dim)

    def forward(self, x_seq, edge_index):
        """
        x_seq: (num_nodes, seq_len, 2)
        edge_index: (2, num_edges) spatial proximity interaction graph
        """
        num_nodes, seq_len, _ = x_seq.shape
        
        # --- SPATIAL FEATURE EXTRACTION ---
        spatial_features = []
        for t in range(seq_len):
            h_t  = self.coordinate_expansion_mlp(x_seq[:, t, :])  # Eq 15
            z_t1 = F.relu(self.gat_layer1(h_t,  edge_index))       # Eq 17 (Concat)
            z_t2 = F.relu(self.gat_layer2(z_t1, edge_index))       # Eq 18 (Averaging)
            spatial_features.append(z_t2)
            
        spatial_features = torch.stack(spatial_features, dim=1) # (N, T, embed_dim)
        
        # --- ENCODING ---
        _, encoder_hidden = self.encoder_gru(spatial_features)
        
        # --- DECODING (Prediction) ---
        decoder_input = spatial_features[:, -1, :].unsqueeze(1) 
        decoder_hidden = encoder_hidden
        
        predictions = []
        for t in range(seq_len): # Predict next 'seq_len' steps
            out, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            pred_pos = self.output_layer(out.squeeze(1))
            predictions.append(pred_pos)
            
            # Autoregressive feedback
            decoder_input = self.coordinate_expansion_mlp(pred_pos).unsqueeze(1)
            
        return torch.stack(predictions, dim=1) # (N, T_pred, 2)