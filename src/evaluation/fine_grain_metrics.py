from typing import Tuple
import pandas as pd

from src.evaluation.map_calculator.map_calculator import MAPCalculator
from src.constants.dll_paths import DLLPaths
from src.constants.column_names import EvaluationCSVColumns


def assert_expected_columns(eval_df: pd.DataFrame):
    expected_columns = [
        EvaluationCSVColumns.QUESTION_ID,
        EvaluationCSVColumns.SUBJECT_ID,
        EvaluationCSVColumns.CONSTRUCT_ID,
        EvaluationCSVColumns.ACTUAL_INDEX,
        EvaluationCSVColumns.RANKINGS,
    ]
    if len(eval_df.columns) < len(expected_columns) or not all(
        col in expected_columns for col in eval_df.columns
    ):
        raise ValueError(
            f"Expected columns: {expected_columns}, but got: {eval_df.columns}"
        )


def calculate(eval_df, column, map_calculator):
    return (
        eval_df.groupby(column)[
            [
                EvaluationCSVColumns.ACTUAL_INDEX,
                EvaluationCSVColumns.RANKINGS,
            ]
        ]
        .apply(
            lambda row: map_calculator.calculate_map(
                row[EvaluationCSVColumns.ACTUAL_INDEX].to_numpy()[0],
                row[EvaluationCSVColumns.RANKINGS].to_numpy()[0],
            )
        )
        .reset_index(name=EvaluationCSVColumns.MAP)
    )


def calculate_fine_grain_metrics(
    eval_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate fine grain metrics for question, subject and construct wise metrics.

    Args:
        eval_df (pd.DataFrame): Evaluation dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Question wise, subject wise and construct wise metrics.
    """
    assert_expected_columns(eval_df)

    map_calculator = MAPCalculator(DLLPaths.MAP_CALCULATOR)

    question_wise_metrics = calculate(
        eval_df, EvaluationCSVColumns.QUESTION_ID, map_calculator
    )
    subject_wise_metrics = calculate(
        eval_df, EvaluationCSVColumns.SUBJECT_ID, map_calculator
    )
    construct_wise_metrics = calculate(
        eval_df, EvaluationCSVColumns.CONSTRUCT_ID, map_calculator
    )

    return question_wise_metrics, subject_wise_metrics, construct_wise_metrics
