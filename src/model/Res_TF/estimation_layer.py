import torch
import torch.nn as nn

class EstimationDecoderFC(nn.Module):
    def __init__(self, input_dim=6, layer_nodes=[64, 64, 32], dropout_pct=0.0, output_dim=1):
        super(EstimationDecoderFC, self).__init__()
        
        self.input_dim = input_dim
        self.layer_nodes = layer_nodes
        self.dropout_pct = dropout_pct
        self.output_dim = output_dim
        
        # Create a list of layers dynamically
        layers = []
        prev_dim = input_dim
        for node_count in layer_nodes:
            layers.append(nn.Linear(prev_dim, node_count))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_pct))
            prev_dim = node_count
        
        # Add the output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create a sequential model with all layers
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)