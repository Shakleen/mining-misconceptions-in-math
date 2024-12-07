from typing import Optional

import torch
import torch.nn.functional as F


def info_nce_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    temperature: Optional[float] = 0.07,
    alpha: float = 0.1,
):
    """
    Calculate InfoNCE loss

    Args:
        similarities (torch.Tensor): Tensor of shape (batch_size, num_negatives + 1) containing similarity scores
        labels (torch.Tensor): Tensor of shape (batch_size) containing indices of positive examples
        temperature (Optional[float]): Temperature parameter to scale the similarities (default: 0.07)
        alpha (float): Label smoothing parameter between 0 and 1 (default: 0.1)

    Returns:
        loss: Mean InfoNCE loss across the batch
    """
    similarities = similarities / temperature
    loss = F.cross_entropy(
        similarities,
        labels,
        reduction="mean",
        label_smoothing=alpha,
    )
    return loss


def contrastive_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    temperature: Optional[float] = 0.07,
    alpha: float = 0.1,
):
    """Calculate contrastive loss with label smoothing

    Args:
        similarities (torch.Tensor): Tensor of shape (batch_size, num_negatives + 1) containing similarity scores
        labels (torch.Tensor): Tensor of shape (batch_size) containing indices of positive examples
        temperature (Optional[float]): Temperature parameter to scale the similarities (default: 0.07)
        alpha (float): Label smoothing parameter between 0 and 1 (default: 0.1)

    Returns:
        loss: Mean contrastive loss across the batch with label smoothing
    """
    batch_size, num_classes = similarities.shape

    # Create smoothed target distribution
    smooth_mask = torch.full_like(similarities, alpha / (num_classes - 1))
    smooth_mask[torch.arange(batch_size), labels] = 1.0 - alpha

    exp = torch.exp(similarities / temperature)
    log_probs = torch.log(exp / exp.sum(dim=1, keepdim=True))

    loss = -(smooth_mask * log_probs).sum(dim=1).mean()
    return loss
