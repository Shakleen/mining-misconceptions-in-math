import pytest

from src.data_preparation.negative_sampler.random_sampler import RandomNegativeSampler


def test_init():
    sampler = RandomNegativeSampler(sample_size=1, total_misconceptions=10)
    assert sampler.sample_size == 1
    assert sampler.total_misconceptions == 10


@pytest.mark.parametrize("sample_size", [1, 2, 3])
def test_random_negative_sampler(sample_size: int):
    sampler = RandomNegativeSampler(sample_size=sample_size, total_misconceptions=10)
    assert len(sampler.sample(1)) == sample_size
