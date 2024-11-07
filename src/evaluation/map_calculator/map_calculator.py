# map_calculator.py
import ctypes
import numpy as np
import os
from typing import Union, List

from src.constants.dll_paths import DLLPaths


class MAPCalculator:
    """Python wrapper for compiled MAP Calculator library."""

    def __init__(self, lib_path: str):
        """
        Initialize the MAP Calculator with the path to the compiled library.

        Args:
            lib_path: Path to the compiled library. If None, will look in the current directory.
        """
        # Load the library
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise OSError(f"Failed to load the library at {lib_path}: {e}")

        # Set up the function signatures
        self.lib.calculate_map.argtypes = [
            ctypes.c_int,  # actual_index
            ctypes.POINTER(ctypes.c_int),  # rankings
            ctypes.c_int,  # rankings_size
        ]
        self.lib.calculate_map.restype = ctypes.c_double

        self.lib.calculate_batch_map.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # actual_indices
            ctypes.POINTER(ctypes.c_int),  # rankings_flat
            ctypes.c_int,  # num_queries
            ctypes.c_int,  # rankings_per_query
        ]
        self.lib.calculate_batch_map.restype = ctypes.c_double

    def calculate_map(
        self, actual_index: int, rankings: Union[List[int], np.ndarray]
    ) -> float:
        """
        Calculate Mean Average Precision for a single query.

        Args:
            actual_index: The true position/index
            rankings: Array of predicted rankings

        Returns:
            float: MAP value for the query
        """
        # Convert input to numpy array if it isn't already
        rankings_array = np.asarray(rankings, dtype=np.int32)

        return self.lib.calculate_map(
            ctypes.c_int(actual_index),
            rankings_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(len(rankings_array)),
        )

    def calculate_batch_map(
        self,
        actual_indices: Union[List[int], np.ndarray],
        rankings_batch: Union[List[List[int]], np.ndarray],
    ) -> float:
        """
        Calculate Mean Average Precision for a batch of queries.

        Args:
            actual_indices: Array of true positions/indices
            rankings_batch: 2D array where each row contains rankings for one query

        Returns:
            float: Average MAP value across all queries
        """
        # Convert inputs to numpy arrays if they aren't already
        actual_indices_array = np.asarray(actual_indices, dtype=np.int32)
        rankings_batch_array = np.asarray(rankings_batch, dtype=np.int32)

        num_queries = len(actual_indices_array)
        rankings_per_query = rankings_batch_array.shape[1]

        # Ensure the rankings array is contiguous in memory
        rankings_batch_array = np.ascontiguousarray(rankings_batch_array)

        return self.lib.calculate_batch_map(
            actual_indices_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            rankings_batch_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(num_queries),
            ctypes.c_int(rankings_per_query),
        )


# Example usage
if __name__ == "__main__":
    # Initialize the calculator with default library path
    calculator = MAPCalculator(DLLPaths.MAP_CALCULATOR)

    # Single query example
    rankings = [3, 1, 4, 2, 0]
    actual_index = 2
    map_value = calculator.calculate_map(actual_index, rankings)
    print(f"Single query MAP: {map_value}")

    # Batch example
    actual_indices = [2, 1, 3]
    rankings_batch = [[3, 1, 4, 2, 0], [1, 2, 3, 4, 0], [2, 3, 1, 4, 0]]
    batch_map = calculator.calculate_batch_map(actual_indices, rankings_batch)
    print(f"Batch MAP: {batch_map}")

    # Using numpy arrays
    np_rankings = np.array([3, 1, 4, 2, 0])
    np_map = calculator.calculate_map(actual_index, np_rankings)
    print(f"Single query MAP (numpy): {np_map}")
