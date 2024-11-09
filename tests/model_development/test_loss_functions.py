import pytest
import torch

from src.model_development.loss_functions import info_nce_loss


@pytest.mark.parametrize(
    ("batch_size", "num_negatives", "temperature"),
    [(4, 1, 0.07), (8, 5, 0.07), (16, 10, 0.07)],
)
def test_info_nce_loss(batch_size: int, num_negatives: int, temperature: float):
    similarities = torch.rand((batch_size, num_negatives + 1))
    labels = torch.randint(0, num_negatives + 1, (batch_size,))
    loss = info_nce_loss(similarities, labels, temperature)
    assert loss.item() >= 0
