from abc import ABC, abstractmethod
from typing import List


class AbstractNegativeSampler(ABC):
    """Abstract class for negative sampling."""

    def __init__(self, sample_size: int):
        self.sample_size = sample_size

    @abstractmethod
    def sample(self, actual_misconception_id: int) -> List[int]:
        """Sample negative misconceptions for a given actual misconception."""
        raise NotImplementedError("Subclasses must implement this method.")
