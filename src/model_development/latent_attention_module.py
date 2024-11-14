from typing import Optional
import torch
import torch.nn as nn


class LatentAttentionLayer(nn.Module):
    """Latent Attention Layer from NV-Embed Paper: https://arxiv.org/pdf/2405.17428"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = 512,
        num_latents: Optional[int] = 512,
        num_heads: Optional[int] = 8,
        mlp_dim: Optional[int] = 1024,
    ):
        super(LatentAttentionLayer, self).__init__()
        # Dimensions
        # Length of the input sequence, l
        self.hidden_dim = hidden_dim  # Hidden dimension, d
        self.num_latents = num_latents  # Number of latents, r

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Define the latent array K = V, a learnable parameter with shape [r, d]
        self.latent_array = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # Multi-head attention for cross-attention between Q and latent_array
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
        )

        # MLP layer (typically two linear layers with activation in between)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim),
        )

    def forward(self, Q):
        """
        Forward pass of the Latent Attention Layer with MLP and mean pooling.

        Args:
            Q (torch.Tensor): Query tensor from the decoder, shape [batch_size, seq_len, d]

        Returns:
            torch.Tensor: Pooled output tensor, shape [batch_size, d]
        """
        # Project the input query
        Q = self.input_proj(Q)

        # Repeat the latent array for each item in the batch
        seq_len = Q.size(1)
        K = self.latent_array.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # Shape [r, batch_size, d]

        # Cross-attention between Q and K
        O, _ = self.attention(query=Q, key=K, value=K)  # O has shape [l, batch_size, d]

        # Apply the MLP layer to each element in the sequence
        O = self.mlp(O)  # Shape [l, batch_size, d]

        # Mean pooling across the sequence length dimension (dim=0)
        O_pooled = O.mean(dim=1)  # Shape [batch_size, d]

        return O_pooled
