import pytest

from src.evaluation.map_calculator.map_calculator import MAPCalculator
from src.constants.dll_paths import DLLPaths


@pytest.fixture(scope="module")
def map_calculator():
    return MAPCalculator(DLLPaths.MAP_CALCULATOR)


@pytest.mark.parametrize(
    "actual_index, rankings, expected_map",
    [
        (1, [1, 2, 3, 4], 1.0),
        (1, [2, 1, 3, 4], 0.5),
        (1, [3, 2, 4, 1], 0.25),
        (0, [3, 2, 4, 1], 0.0),
    ],
)
def test_map_calculator(map_calculator, actual_index, rankings, expected_map):
    assert map_calculator.calculate_map(actual_index, rankings) == expected_map


def test_batch_map_calculator(map_calculator):
    actual_indices = [1, 1, 1, 0]
    rankings = [
        [1, 2, 3, 4],
        [2, 1, 3, 4],
        [3, 2, 4, 1],
        [4, 3, 2, 1],
    ]
    expected_map = 0.4375
    assert map_calculator.calculate_batch_map(actual_indices, rankings) == expected_map
