from typing import List, Optional
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.data_preparation.negative_sampler.abstract_negative_sampler import (
    AbstractNegativeSampler,
)


class HardNegativeSamplerV2(AbstractNegativeSampler):
    """Hard negative sampler v2.

    Hard negatives are sampled from a larger set of similar misconceptions. The larger superset
    size is controlled by the `super_set_size` parameter. For validation, we sample only the top
    `sample_size` hard negatives. There is no randomness in validation sampling other than shuffling.
    """

    def __init__(
        self,
        sample_size: int,
        misconception_embeddings: np.ndarray,
        is_validation: bool,
        super_set_size: Optional[int] = 10,
    ):
        """Random hard negative sampler.

        Args:
            sample_size (int): Number of negative samples to sample.
            total_misconceptions (int): Total number of misconceptions.
            misconception_embeddings (np.ndarray): Embeddings of all misconceptions.
            is_validation (bool): Whether the sampler is for validation or training.
            super_set_size (Optional[int], optional): Size of the super set. Ideally,
            we want to sample from a larger set of similar misconceptions. This defines
            how many similar misconceptions we want to sample from. Defaults to 10.
        """
        super().__init__(sample_size)
        self.total_misconceptions = misconception_embeddings.shape[0]
        self.misconception_embeddings = misconception_embeddings
        self.is_validation = is_validation
        self.super_set_size_multiplier = super_set_size

        if not self.is_validation:
            self.count = np.zeros(self.total_misconceptions, dtype=np.int32)

    def sample(self, actual_misconception_id: int) -> List[int]:
        output = self._get_hard_negative_samples(actual_misconception_id)

        if not self.is_validation:
            output = self._downsample(output)

        output.append(actual_misconception_id)
        random.shuffle(output)

        return output

    def _get_hard_negative_samples(self, actual_misconception_id):
        m_embedding = self.misconception_embeddings[actual_misconception_id].reshape(
            1, -1
        )
        hard_negative_count = (
            self.sample_size * self.super_set_size_multiplier
            if not self.is_validation
            else self.sample_size
        )

        similarities = cosine_similarity(
            m_embedding, self.misconception_embeddings
        ).flatten()
        hard_negative_ids = np.argsort(similarities)[::-1][
            1:hard_negative_count  # Skip the actual misconception
        ].tolist()
        return list(set(hard_negative_ids))

    def _downsample(self, hard_negative_ids: List[int]) -> List[int]:
        # Vectorized indexing to get counts
        candidate_counts = self.count[hard_negative_ids]

        # Add 1 to avoid division by zero and to give non-zero probability to all items
        weights = 1 / (candidate_counts + 1)

        # Normalize weights to probabilities
        probabilities = weights / weights.sum()

        # Sample without replacement using the computed probabilities
        selected_indices = np.random.choice(
            len(hard_negative_ids),
            size=self.sample_size - 1,
            replace=False,
            p=probabilities,
        )

        output = [hard_negative_ids[i] for i in selected_indices]
        self.count[output] += 1
        return output


if __name__ == "__main__":
    from src.utils.seed_everything import seed_everything

    seed_everything(42)

    misconception_embeddings = np.load("assets/misconception_embeddings.npy")
    sampler = HardNegativeSamplerV2(
        sample_size=10,
        misconception_embeddings=misconception_embeddings,
        is_validation=False,
        super_set_size=10,
    )

    for i in range(10):
        print(sampler.sample(0))
