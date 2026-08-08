import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class MobilityGAT_GRU(nn.Module):
    """
    Mobility-Aware Prediction Model (Sec IV-B)
    Captures vehicle interactions and predicts future trajectories.
    """
    def __init__(self, input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2):
        super(MobilityGAT_GRU, self).__init__()
        
        # 1. 2-layer MLP to expand coordinates (Eq 15)
        self.coordinate_expansion_mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. 4-head Graph Attention Network (GAT) to capture vehicle interactions (Eq 16-18)
        # out_channels is embed_dim // num_heads so that concatenation restores embed_dim
        self.gat_layer = GATConv(
            in_channels=embed_dim, 
            out_channels=embed_dim // num_heads, 
            heads=num_heads, 
            concat=True
        )
        
        # 3. GRU Encoder-Decoder to predict future positions (Eq 19-21)
        # Encoder GRU
        self.encoder_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        
        # Decoder GRU
        self.decoder_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        
        # Final output layer to map GRU hidden states back to (x, y) coordinates
        self.output_layer = nn.Linear(gru_hidden, output_dim)

    def forward(self, x_seq, edge_index):
        """
        Args:
            x_seq: Tensor of shape (num_nodes, seq_len, input_dim) containing historical (x,y)
            edge_index: Graph connectivity of shape (2, num_edges)
        Returns:
            predicted_positions: Tensor of shape (num_nodes, pred_seq_len, output_dim)
        """
        num_nodes, seq_len, _ = x_seq.shape
        
        # Store spatial features for each time step
        spatial_features = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :] # Shape: (num_nodes, 2)
            
            # Eq 15: Expand coordinates
            h_t = self.coordinate_expansion_mlp(x_t) 
            h_t = F.relu(h_t)
            
            # Eq 16-18: GAT spatial interaction
            z_t = self.gat_layer(h_t, edge_index)
            z_t = F.relu(z_t)
            
            spatial_features.append(z_t)
            
        # Stack features temporally: Shape (num_nodes, seq_len, embed_dim)
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Eq 19-21: Encoder-Decoder GRU
        # Encode the historical sequence
        _, encoder_hidden = self.encoder_gru(spatial_features)
        
        pred_seq_len = seq_len # Predicting same length into the future
        
        # Initial decoder input (zeros or final spatial feature)
        decoder_input = torch.zeros((num_nodes, 1, spatial_features.size(-1)), device=x_seq.device)
        decoder_hidden = encoder_hidden
        
        predictions = []
        for t in range(pred_seq_len):
            out, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            
            # Map hidden state to (x, y) prediction
            pred_pos = self.output_layer(out.squeeze(1))
            predictions.append(pred_pos)
            
            # Feedback loop: expand prediction to form the next decoder input
            decoder_input = self.coordinate_expansion_mlp(pred_pos).unsqueeze(1)
            
        return torch.stack(predictions, dim=1)
