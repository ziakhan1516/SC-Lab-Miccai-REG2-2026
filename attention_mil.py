import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) encoder.

    Input:
        feats -> [num_patches, in_dim]

    Output:
        bag_embedding -> [hidden_dim]
        attention_weights -> [num_patches, 1]
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        attention_dim: int = 128,
        dropout: float = 0.25
    ):
        super().__init__()

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def forward(self, feats):
        if feats.dim() != 2:
            raise ValueError(
                "AttentionMIL expects features shaped "
                f"[num_patches, feature_dim], got {tuple(feats.shape)}"
            )

        if feats.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected feature_dim={self.in_dim}, got {feats.shape[-1]}"
            )

        # Project features
        H = self.feature_proj(feats)      # [N, 512]

        # Compute attention scores
        A = self.attention(H)             # [N, 1]

        # Normalize attention across patches
        A = torch.softmax(A, dim=0)

        # Weighted aggregation
        bag_embedding = torch.sum(A * H, dim=0)   # [512]

        return bag_embedding, A
