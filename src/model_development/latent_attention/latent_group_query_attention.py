from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LatentGroupQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: Optional[int] = 768,
        num_latents: Optional[int] = 512,
        num_query_heads: Optional[int] = 8,
        num_kv_heads: Optional[int] = 2,  # Fewer KV heads than query heads
        dropout: Optional[float] = 0.1,
        mlp_ratio: Optional[float] = 4.0
    ):
        """
        Latent Attention Layer implementation using Group Query Attention.
        
        Args:
            hidden_dim: Dimension of the hidden representations
            num_latents: Number of latent vectors in the trainable dictionary
            num_query_heads: Number of query attention heads
            num_kv_heads: Number of key/value attention heads (fewer than query heads)
            dropout: Dropout probability
            mlp_ratio: Expansion ratio for the MLP hidden dimension
        """
        super().__init__()
        
        assert hidden_dim % num_query_heads == 0, "hidden_dim must be divisible by num_query_heads"
        assert hidden_dim % num_kv_heads == 0, "hidden_dim must be divisible by num_kv_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.query_head_dim = hidden_dim // num_query_heads
        self.kv_head_dim = hidden_dim // num_kv_heads
        
        # Trainable latent array (dictionary) K = V ∈ R^(r×d)
        self.latent_array = nn.Parameter(torch.randn(num_latents, hidden_dim))
        
        # Group Query Attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # Project to num_query_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # MLP block
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using Group Query Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_dim)
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries
        queries = self.q_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape queries for GQA
        queries = queries.view(
            batch_size, seq_len, self.num_query_heads, self.query_head_dim
        )
        queries = queries.permute(0, 2, 1, 3)  # (batch_size, num_query_heads, seq_len, query_head_dim)
        
        # Reshape latent array for GQA
        # First reshape to KV heads
        latents = self.latent_array.view(self.num_latents, self.num_kv_heads, self.kv_head_dim)
        latents = latents.permute(1, 0, 2)  # (num_kv_heads, num_latents, kv_head_dim)
        
        # Repeat KV heads for each query head in the group
        latents = latents.repeat_interleave(self.num_queries_per_kv, dim=0)
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Compute attention scores
        scale = self.query_head_dim ** -0.5
        scores = torch.matmul(queries, latents.transpose(-2, -1)) * scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to latents
        out = torch.matmul(attention, latents)  # (batch_size, num_query_heads, seq_len, query_head_dim)
        
        # Reshape and combine heads
        out = out.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_query_heads, query_head_dim)
        out = out.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Add residual connection and normalize
        out = self.norm(out + x)
        
        # Apply MLP block
        out = out + self.mlp(out)
        out = self.final_norm(out)
        
        # Mean pooling over sequence length
        out = out.mean(dim=1)  # (batch_size, hidden_dim)
        
        return out

def test_group_query_attention():
    # Test the implementation
    batch_size = 32
    seq_len = 128
    hidden_dim = 768
    
    model = LatentGroupQueryAttention(
        hidden_dim=hidden_dim,
        num_latents=512,
        num_query_heads=8,
        num_kv_heads=2  # Using 2 KV heads for 8 query heads (4:1 ratio)
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = model(x)
    
    assert out.shape == (batch_size, hidden_dim), f"Expected shape {(batch_size, hidden_dim)}, got {out.shape}"
    print("Test passed! Output shape:", out.shape)

if __name__ == "__main__":
    test_group_query_attention()