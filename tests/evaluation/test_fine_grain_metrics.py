import pytest
import pandas as pd

from src.evaluation.fine_grain_metrics import calculate_fine_grain_metrics
from src.constants.column_names import EvaluationCSVColumns


@pytest.fixture(scope="module")
def eval_df():
    return pd.DataFrame(
        {
            EvaluationCSVColumns.QUESTION_ID: [1, 2, 3, 4],
            EvaluationCSVColumns.SUBJECT_ID: [1, 2, 1, 2],
            EvaluationCSVColumns.CONSTRUCT_ID: [1, 3, 1, 3],
            EvaluationCSVColumns.ACTUAL_INDEX: [0, 1, 2, 0],
            EvaluationCSVColumns.RANKINGS: [[1, 2, 3], [2, 1, 3], [3, 2, 1], [1, 2, 3]],
        }
    )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"A": [1, 2, 3]}),
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
    ],
)
def test_calculate_fine_grain_metrics_throws_error_if_expected_columns_not_present(
    df: pd.DataFrame,
):
    with pytest.raises(ValueError):
        calculate_fine_grain_metrics(df)


def test_calculate_fine_grain_metrics_returns_expected_metrics(eval_df: pd.DataFrame):
    qdf, sdf, cdf = calculate_fine_grain_metrics(eval_df)
    assert qdf.columns.tolist() == [
        EvaluationCSVColumns.QUESTION_ID,
        EvaluationCSVColumns.MAP,
    ]
    assert sdf.columns.tolist() == [
        EvaluationCSVColumns.SUBJECT_ID,
        EvaluationCSVColumns.MAP,
    ]
    assert cdf.columns.tolist() == [
        EvaluationCSVColumns.CONSTRUCT_ID,
        EvaluationCSVColumns.MAP,
    ]
