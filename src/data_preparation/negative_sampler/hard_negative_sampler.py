from typing import List, Optional
import random
import numpy as np
from src.data_preparation.negative_sampler.abstract_negative_sampler import (
    AbstractNegativeSampler,
)
from src.utils.searcher.similarity_searcher import SimilaritySearcher
from src.constants.dll_paths import DLLPaths


class HardNegativeSampler(AbstractNegativeSampler):
    """Hard negative sampler."""

    def __init__(
        self,
        sample_size: int,
        total_misconceptions: int,
        misconception_embeddings: np.ndarray,
        hard_to_random_ratio: Optional[float] = 0.5,
    ):
        super().__init__(sample_size)
        self.total_misconceptions = total_misconceptions
        self.hard_to_random_ratio = hard_to_random_ratio
        self.similarity_searcher = SimilaritySearcher(DLLPaths.SIMILARITY_SEARCH)
        self.misconception_embeddings = misconception_embeddings

    def sample(self, actual_misconception_id: int) -> List[int]:
        hard_negative_ids = self._get_hard_negative_samples(actual_misconception_id)
        hard_negative_ids.append(actual_misconception_id)
        output = set(hard_negative_ids)

        while len(output) < self.sample_size:
            output.add(random.randint(0, self.total_misconceptions - 1))

        output = list(output)
        random.shuffle(output)

        return output

    def _get_hard_negative_samples(self, actual_misconception_id):
        m_embedding = self.misconception_embeddings[actual_misconception_id]
        hard_negative_count = int(self.sample_size * self.hard_to_random_ratio)

        hard_negative_ids = self.similarity_searcher.search(
            m_embedding,
            self.misconception_embeddings,
            k=hard_negative_count,
        )

        return hard_negative_ids.tolist()
