from typing import List
import random

from src.data_preparation.negative_sampler.abstract_negative_sampler import (
    AbstractNegativeSampler,
)


class RandomNegativeSampler(AbstractNegativeSampler):
    """Random negative sampler."""

    def __init__(self, sample_size: int, total_misconceptions: int):
        super().__init__(sample_size)
        self.total_misconceptions = total_misconceptions

    def sample(self, actual_misconception_id: int) -> List[int]:
        output = random.sample(
            range(self.total_misconceptions),
            self.sample_size - 1,
        )
        output.append(actual_misconception_id)
        random.shuffle(output)
        return output
