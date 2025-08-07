import torch.nn as nn


class ForceFC(nn.Module):
    def __init__(self, input_dim=1, hidden_size=16, output_dim=16, dropout_pct=0.5):
        super(ForceFC, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_pct = dropout_pct

        self.fc1 = nn.Linear(input_dim, self.hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        return x