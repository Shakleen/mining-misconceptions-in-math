import pytest
import torch
from src.model_development.latent_attention_module import LatentAttentionLayer
from src.configurations.recall_model_config import RecallModelConfig


@pytest.fixture(scope="module")
def config():
    return RecallModelConfig(
        ".cache/Mistral-7B-v0.1",
        0,
        hidden_dim=512,
        num_latents=512,
        num_heads=8,
        mlp_dim=1024,
    )


@pytest.fixture(scope="module")
def latent_attention_layer(config: RecallModelConfig):
    return LatentAttentionLayer(
        input_dim=128,
        hidden_dim=config.hidden_dim,
        num_latents=config.num_latents,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
    )


def test_init(latent_attention_layer: LatentAttentionLayer):
    assert hasattr(latent_attention_layer, "input_proj")
    assert hasattr(latent_attention_layer, "latent_array")
    assert hasattr(latent_attention_layer, "attention")
    assert hasattr(latent_attention_layer, "mlp")


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_dim",
    [(1, 10, 128), (2, 10, 128), (4, 10, 128)],
)
def test_forward(
    latent_attention_layer: LatentAttentionLayer,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
):
    Q = torch.randn(batch_size, seq_len, hidden_dim)
    output = latent_attention_layer(Q)
    assert output.shape == (batch_size, hidden_dim)
