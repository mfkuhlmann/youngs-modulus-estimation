import torch
import torch.nn as nn

class DecoderFC(nn.Module):
    def __init__(self,
                 input_dim=3 * 512,
                 layer_nodes=[256, 256, 64],
                 dropout_pct=0.5,
                 output_dim=1):
        super(DecoderFC, self).__init__()
        
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
        if self.output_dim == 1:
            return torch.sigmoid(x)
        else:
            return x