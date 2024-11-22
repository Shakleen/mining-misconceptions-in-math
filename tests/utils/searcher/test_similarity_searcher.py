import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.searcher.similarity_searcher import SimilaritySearcher
from src.constants.dll_paths import DLLPaths
from src.utils.seed_everything import seed_everything

seed_everything()


@pytest.fixture(scope="module")
def similarity_searcher():
    return SimilaritySearcher(DLLPaths.SIMILARITY_SEARCH)


@pytest.mark.parametrize("dim, k", [(1024, 100), (2048, 50), (4096, 25)])
def test_search(similarity_searcher, dim, k):
    database = np.random.rand(2587, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)

    top_indices = similarity_searcher.search(query, database, k=k)
    assert top_indices.shape == (k,)

    similarities = cosine_similarity(query.reshape(1, -1), database).flatten()
    expected_top_indices = np.argsort(similarities)[::-1][:k]
    assert (top_indices == expected_top_indices).all()


def test_search_raises_error_on_different_shapes(similarity_searcher):
    database = np.random.rand(2587, 1024).astype(np.float32)
    query = np.random.rand(1025).astype(np.float32)

    with pytest.raises(AssertionError):
        similarity_searcher.search(query, database)


def test_search_raises_error_on_non_1d_query(similarity_searcher):
    database = np.random.rand(2587, 1024).astype(np.float32)
    query = np.random.rand(1024, 1).astype(np.float32)

    with pytest.raises(AssertionError):
        similarity_searcher.search(query, database)


def test_search_raises_error_on_non_2d_database(similarity_searcher):
    database = np.random.rand(2587, 1024, 1).astype(np.float32)
    query = np.random.rand(1024).astype(np.float32)

    with pytest.raises(AssertionError):
        similarity_searcher.search(query, database)


@pytest.mark.parametrize(
    "dim, k, batch_size", [(1024, 10, 10), (2048, 5, 10), (4096, 2, 10)]
)
def test_batch_search(similarity_searcher, dim, k, batch_size):
    database = np.random.rand(2587, dim).astype(np.float32)
    queries = np.random.rand(batch_size, dim).astype(np.float32)

    top_indices = similarity_searcher.batch_search(queries, database, k=k)
    assert top_indices.shape == (batch_size, k)

    similarities = cosine_similarity(queries, database)
    expected_top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :k]
    assert (top_indices == expected_top_indices).all()


def test_batch_search_raises_error_on_different_shapes(similarity_searcher):
    database = np.random.rand(2587, 1024).astype(np.float32)
    queries = np.random.rand(100, 1025).astype(np.float32)

    with pytest.raises(AssertionError):
        similarity_searcher.batch_search(queries, database)


def test_batch_search_raises_error_on_non_2d_queries(similarity_searcher):
    database = np.random.rand(2587, 1024).astype(np.float32)
    queries = np.random.rand(100, 1024, 1).astype(np.float32)

    with pytest.raises(AssertionError):
        similarity_searcher.batch_search(queries, database)
