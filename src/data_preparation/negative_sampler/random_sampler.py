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
        output = set([actual_misconception_id])

        while len(output) < self.sample_size:
            output.add(random.randint(0, self.total_misconceptions - 1))

        output = list(output)
        random.shuffle(output)

        return output
