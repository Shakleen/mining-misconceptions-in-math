from typing import List
import pytest

from src.data_preparation.negative_sampler.abstract_negative_sampler import (
    AbstractNegativeSampler,
)


class TestAbstractNegativeSampler(AbstractNegativeSampler):
    def sample(self, actual_misconception_id: int) -> List[int]:
        return [actual_misconception_id]


def test_abstract_negative_sampler_not_implemented():
    with pytest.raises(TypeError):
        sampler = AbstractNegativeSampler(sample_size=1)
        sampler.sample(1)


def test_has_abstract_method():
    assert hasattr(AbstractNegativeSampler, "sample")


def test_abstract_negative_sampler():
    sampler = TestAbstractNegativeSampler(sample_size=1)
    assert sampler.sample(1) == [1]
