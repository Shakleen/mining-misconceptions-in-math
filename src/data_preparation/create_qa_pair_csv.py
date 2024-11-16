import argparse
import wandb
import pandas as pd
from tqdm import tqdm

from src.constants.wandb_project import WandbProject
from src.constants.column_names import (
    TrainCSVColumns,
    QAPairCSVColumns,
    MisconceptionsCSVColumns,
)
from src.utils.wandb_artifact import load_dataframe_artifact, log_dataframe_artifact


def parse_args():
    parser = argparse.ArgumentParser(
        prog="create-qa-pair-dataset",
        description="""
        Create a CSV file for QA pair dataset. Each row contains the following columns:
        - `QuestionId`: Id of the question.
        - `QuestionText`: Text of the question.
        - `SubjectId`: Id of the subject.
        - `SubjectName`: Name of the subject.
        - `ConstructId`: Id of the construct.
        - `ConstructName`: Name of the construct.
        - `AnswerText`: Text of the answer.
        - `MisconceptionId`: Id of the misconception.
        - `MisconceptionName`: Name of the misconception.
        """,
        usage="""
        python src/data_preparation/create_qa_pair_csv.py
        --train-dataset-version <version>
        --misconceptions-dataset-version <version>
        """,
    )
    parser.add_argument(
        "--train-dataset-version",
        type=str,
        required=False,
        default="latest",
        help="Version of the train dataset to use. Default is latest.",
    )
    parser.add_argument(
        "--misconceptions-dataset-version",
        type=str,
        required=False,
        default="latest",
        help="Version of the misconceptions dataset to use. Default is latest.",
    )
    return parser.parse_args()


def create_new_row(
    row: pd.Series,
    option: str,
    misconception_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a new row for the QA pair dataset.

    Args:
        row (pd.Series): Row of the train dataset.
        option (str): Option of the answer.
        misconception_df (pd.DataFrame): Dataframe of the misconceptions dataset.

    Returns:
        pd.DataFrame: New row for the QA pair dataset.
    """
    new_row = {
        QAPairCSVColumns.QUESTION_ID: [row[TrainCSVColumns.QUESTION_ID]],
        QAPairCSVColumns.QUESTION_TEXT: [row[TrainCSVColumns.QUESTION_TEXT]],
        QAPairCSVColumns.SUBJECT_ID: [row[TrainCSVColumns.SUBJECT_ID]],
        QAPairCSVColumns.SUBJECT_NAME: [row[TrainCSVColumns.SUBJECT_NAME]],
        QAPairCSVColumns.CONSTRUCT_ID: [row[TrainCSVColumns.CONSTRUCT_ID]],
        QAPairCSVColumns.CONSTRUCT_NAME: [row[TrainCSVColumns.CONSTRUCT_NAME]],
        QAPairCSVColumns.ANSWER_TEXT: [
            row[TrainCSVColumns.ANSWER_FORMAT.format(option=option)]
        ],
        QAPairCSVColumns.MISCONCEPTION_ID: [
            row[TrainCSVColumns.MISCONCEPTION_FORMAT.format(option=option)]
        ],
        QAPairCSVColumns.MISCONCEPTION_NAME: [
            misconception_df.loc[
                row[TrainCSVColumns.MISCONCEPTION_FORMAT.format(option=option)],
                MisconceptionsCSVColumns.MISCONCEPTION_NAME,
            ]
        ],
    }

    return pd.DataFrame(new_row)


def main(args: argparse.Namespace):
    wandb.init(
        project=WandbProject.PROJECT_NAME,
        job_type="create-contrastive-dataset",
        name=f"contrastive-dataset-random",
        config=args,
    )

    train_df = load_dataframe_artifact(
        WandbProject.TRAIN_DATASET_NAME,
        args.train_dataset_version,
    )
    misconception_df = load_dataframe_artifact(
        WandbProject.MISCONCEPTIONS_DATASET_NAME,
        args.misconceptions_dataset_version,
    )

    qa_df = pd.DataFrame()

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Creating QA pair dataset",
    ):
        for option in ["A", "B", "C", "D"]:
            if option == row[TrainCSVColumns.CORRECT_ANSWER] or pd.isna(
                row[TrainCSVColumns.MISCONCEPTION_FORMAT.format(option=option)]
            ):
                continue

            qa_df = pd.concat([qa_df, create_new_row(row, option, misconception_df)])

    log_dataframe_artifact(
        qa_df,
        artifact_name=WandbProject.QA_PAIR_DATASET_NAME,
        artifact_type="dataset",
        description="""
        QA pair dataset.

        Each row of the dataset contains the following columns:
        - `QuestionId`: Id of the question.
        - `QuestionText`: Text of the question.
        - `SubjectId`: Id of the subject.
        - `SubjectName`: Name of the subject.
        - `ConstructId`: Id of the construct.
        - `ConstructName`: Name of the construct.
        - `AnswerText`: Text of the answer.
        - `MisconceptionId`: Id of the misconception.
        - `MisconceptionName`: Name of the misconception.
        """,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
