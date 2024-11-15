from typing import Optional

import torch
import torch.nn.functional as F


def info_nce_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    temperature: Optional[float] = 0.07,
):
    """
    Calculate InfoNCE loss

    Args:
        similarities (torch.Tensor): Tensor of shape (batch_size, num_negatives + 1) containing similarity scores
        labels (torch.Tensor): Tensor of shape (batch_size) containing indices of positive examples
        temperature (Optional[float]): Temperature parameter to scale the similarities (default: 0.07)

    Returns:
        loss: Mean InfoNCE loss across the batch
    """
    similarities = similarities / temperature
    loss = F.cross_entropy(similarities, labels, reduction="mean")
    return loss
