import ctypes
import numpy as np
from typing import Optional


class SimilaritySearcher:
    """Wrapper for the similarity search C library."""

    def __init__(self, lib_path):
        # Load the compiled C library
        self.lib = ctypes.CDLL(str(lib_path))

        # Define argument types for the C functions
        self.lib.find_top_similar.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),  # query
            np.ctypeslib.ndpointer(dtype=np.float32),  # database
            ctypes.c_int,  # vector_dim
            ctypes.c_int,  # num_vectors
            np.ctypeslib.ndpointer(dtype=np.int32),  # top_indices
            ctypes.c_int,  # num_threads
            ctypes.c_int,  # k
        ]

        self.lib.find_top_similar_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),  # queries
            np.ctypeslib.ndpointer(dtype=np.float32),  # database
            ctypes.c_int,  # vector_dim
            ctypes.c_int,  # num_vectors
            ctypes.c_int,  # num_queries
            np.ctypeslib.ndpointer(dtype=np.int32),  # top_indices
            ctypes.c_int,  # num_threads
            ctypes.c_int,  # k
        ]

        # Constants from C code
        self.TOP_K = 25

    def search(
        self,
        query: np.ndarray,
        database: np.ndarray,
        num_threads: Optional[int] = 8,
        k: Optional[int] = 25,
    ) -> np.ndarray:
        """
        Search for most similar vectors to a single query vector.

        Args:
            query (np.ndarray): Query vector of shape (vector_dim,)
            database (np.ndarray): Database of vectors of shape (num_vectors, vector_dim)
            num_threads (Optional[int]): Number of threads to use (8 for default)
            k (Optional[int]): Number of top matches to return (25 for default)

        Returns:
            np.ndarray: Indices of top-K most similar vectors
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        database = np.ascontiguousarray(database, dtype=np.float32)

        vector_dim = query.shape[0]
        num_vectors = len(database)

        top_indices = np.zeros(k, dtype=np.int32)

        self.lib.find_top_similar(
            query, database, vector_dim, num_vectors, top_indices, num_threads, k
        )

        return top_indices

    def batch_search(
        self,
        queries: np.ndarray,
        database: np.ndarray,
        num_threads: Optional[int] = 8,
        k: Optional[int] = 25,
    ) -> np.ndarray:
        """
        Search for most similar vectors for multiple queries in batch.

        Args:
            queries (np.ndarray): Query vectors of shape (num_queries, vector_dim)
            database (np.ndarray): Database of vectors of shape (num_vectors, vector_dim)
            num_threads (Optional[int]): Number of threads to use (8 for default)
            k (Optional[int]): Number of top matches to return (25 for default)

        Returns:
            np.ndarray: Indices of top-K most similar vectors for each query
                       Shape: (num_queries, TOP_K)
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        database = np.ascontiguousarray(database, dtype=np.float32)

        num_queries, vector_dim = queries.shape
        num_vectors = len(database)

        top_indices = np.zeros(num_queries * k, dtype=np.int32)

        self.lib.find_top_similar_batch(
            queries,
            database,
            vector_dim,
            num_vectors,
            num_queries,
            top_indices,
            num_threads,
            k,
        )

        return top_indices.reshape(num_queries, k)


if __name__ == "__main__":
    import time
    from src.constants.dll_paths import DLLPaths

    similarity_search = SimilaritySearcher(DLLPaths.SIMILARITY_SEARCH)

    database = np.random.rand(2587, 1024).astype(np.float32)
    query = np.random.rand(1024).astype(np.float32)

    start_time = time.time()
    for i in range(10000):
        similarity_search.search(query, database)
    end_time = time.time()
    print(f"Individual search time taken: {end_time - start_time} seconds")

    queries = np.random.rand(10000, 1024).astype(np.float32)
    start_time = time.time()
    similarity_search.batch_search(queries, database)
    end_time = time.time()
    print(f"Batch search time taken: {end_time - start_time} seconds")
