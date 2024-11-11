import pytest
import numpy as np
from src.utils.searcher.similarity_searcher import SimilaritySearcher
from src.constants.dll_paths import DLLPaths


@pytest.fixture(scope="module")
def similarity_searcher():
    return SimilaritySearcher(DLLPaths.SIMILARITY_SEARCH)


@pytest.mark.parametrize("dim, k", [(1024, 100), (2048, 50), (4096, 25)])
def test_search(similarity_searcher, dim, k):
    database = np.random.rand(2587, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)

    top_indices = similarity_searcher.search(query, database, k=k)
    assert top_indices.shape == (k,)


@pytest.mark.parametrize("dim, k, batch_size", [(1024, 100, 100), (2048, 50, 100), (4096, 25, 10)])
def test_batch_search(similarity_searcher, dim, k, batch_size):
    database = np.random.rand(2587, dim).astype(np.float32)
    queries = np.random.rand(batch_size, dim).astype(np.float32)

    top_indices = similarity_searcher.batch_search(queries, database, k=k)
    assert top_indices.shape == (batch_size, k)
