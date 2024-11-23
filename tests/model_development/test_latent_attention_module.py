import pytest
import torch

from src.model_development.latent_attention.latent_multi_head_attention import (
    LatentMultiHeadAttention,
)


@pytest.fixture(scope="module")
def latent_attention_layer():
    return LatentMultiHeadAttention(128)


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_dim",
    [(1, 10, 128), (2, 10, 128), (4, 10, 128)],
)
def test_forward(
    latent_attention_layer: LatentMultiHeadAttention,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
):
    Q = torch.randn(batch_size, seq_len, hidden_dim)
    output = latent_attention_layer(Q)
    assert output.shape == (batch_size, latent_attention_layer.hidden_dim)
