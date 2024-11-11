import ctypes
import numpy as np


class CosineSimilarity:
    """
    Calculate the cosine similarity between two vectors using a C library.
    """

    def __init__(self, path: str):
        """
        Initialize the CosineSimilarity class.

        Args:
            path (str): Path to the C library.
        """
        self.lib = ctypes.CDLL(path)

        self.lib.cosine_similarity.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
        ]
        self.lib.cosine_similarity.restype = ctypes.c_float

    def calculate(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity between the two vectors.
        """
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)

        return self.lib.cosine_similarity(vec1, vec2, len(vec1))


if __name__ == "__main__":
    import time

    from src.constants.dll_paths import DLLPaths

    cosine_similarity = CosineSimilarity(DLLPaths.COSINE_SIMILARITY)
    arr1 = np.random.rand(1024)
    arr2 = np.random.rand(1024)

    start_time = time.time()

    for _ in range(10000):
        cosine_similarity.calculate(arr1, arr2)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")