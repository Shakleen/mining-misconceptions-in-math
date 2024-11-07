import pytest

from src.evaluation.map_calculator.map_calculator import MAPCalculator
from src.constants.dll_paths import DLLPaths


@pytest.fixture(scope="module")
def map_calculator():
    return MAPCalculator(DLLPaths.MAP_CALCULATOR)


@pytest.mark.parametrize(
    "actual_index, rankings, expected_map",
    [
        (1, [1, 2, 3], 1.0),
        (1, [2, 1, 3], 0.5),
        (0, [3, 2, 1], 0.0),
    ],
)
def test_map_calculator(map_calculator, actual_index, rankings, expected_map):
    assert map_calculator.calculate_map(actual_index, rankings) == expected_map
