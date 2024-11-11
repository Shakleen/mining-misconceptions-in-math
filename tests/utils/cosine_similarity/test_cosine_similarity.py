import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.cosine_similarity.cosine_similarity import CosineSimilarity
from src.constants.dll_paths import DLLPaths


@pytest.fixture(scope="module")
def cosine_similarity_dll():
    return CosineSimilarity(DLLPaths.COSINE_SIMILARITY)


def test_cosine_similarity_dll_init(cosine_similarity_dll):
    assert hasattr(cosine_similarity_dll, "lib")


@pytest.mark.parametrize("size", [256, 512, 1024, 2048, 4096])
def test_cosine_similarity_dll(cosine_similarity_dll, size):
    arr1 = np.random.rand(size)
    arr2 = np.random.rand(size)

    expected = cosine_similarity(arr1.reshape(1, -1), arr2.reshape(1, -1))[0][0]
    actual = cosine_similarity_dll.calculate(arr1, arr2)

    assert round(actual, 5) == round(expected, 5)


def test_cosine_similarity_dll_raises_error_on_different_shapes(cosine_similarity_dll):
    arr1 = np.random.rand(1024)
    arr2 = np.random.rand(1025)

    with pytest.raises(AssertionError):
        cosine_similarity_dll.calculate(arr1, arr2)


def test_cosine_similarity_dll_raises_error_on_non_1d_vectors(cosine_similarity_dll):
    arr1 = np.random.rand(1024, 1)
    arr2 = np.random.rand(1024)

    with pytest.raises(AssertionError):
        cosine_similarity_dll.calculate(arr1, arr2)
