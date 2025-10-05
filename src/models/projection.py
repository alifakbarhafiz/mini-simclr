# src/models/projection.py
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    2-layer projection head (as in SimCLR): input_dim -> hidden -> out_dim
    """
    def __init__(self, input_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
