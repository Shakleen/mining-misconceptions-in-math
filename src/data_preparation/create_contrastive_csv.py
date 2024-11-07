import argparse
from tqdm import tqdm
import wandb
import pandas as pd
import random

from src.utils.wandb_artifact import load_dataframe_artifact, log_dataframe_artifact
from src.constants.wandb_project import WandbProject
from src.constants.column_names import (
    TrainCSVColumns,
    MisconceptionsCSVColumns,
    ContrastiveCSVColumns,
)
from src.utils.seed_everything import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        prog="create-contrastive-dataset",
        description="""
        Create contrastive learning dataset where negative samples are randomly selected from the corpus.

        Each row of the dataset contains the following columns:
        - `QuestionId`: Used for cv fold splitting.
        - `SubjectId`: Id of the subject. Used for subjectwise performance evaluation.
        - `ConstructId`: Id of the construct. Used for constructwise performance evaluation.
        - `QuestionDetails`: Formatted question details. Fed into question encoder.
        - `MisconceptionList`: List of misconception texts. Fed into misconception encoder.
        - `Label`: Index of the actual related misconception in the `MisconceptionList`.
        - `MisconceptionId`: Id of the actual related misconception. Used for calculating 
        MAP@K score.
        """,
        usage="""
        python src/data_preparation/create_contrastive_csv.py
        --train-dataset-version <version>
        --misconceptions-dataset-version <version>
        --number-of-negative-samples <number>
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
    parser.add_argument(
        "--number-of-negative-samples",
        type=int,
        required=False,
        default=10,
        help="Number of negative samples to include in the dataset. Default is 10.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    seed_everything()

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

    query = (
        "Given subject, construct, question and incorrect answer, "
        + "retrieve the list of misconceptions that are related to the incorrect answer."
        + "\nSubject: {subject}"
        + "\nConstruct: {construct}"
        + "\nQuestion: {question}"
        + "\nIncorrect Answer: {incorrect_answer}"
    )

    contrastive_df = pd.DataFrame()

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Creating contrastive dataset with random negative samples",
    ):
        for option in ["A", "B", "C", "D"]:
            if option == row[TrainCSVColumns.CORRECT_ANSWER] or pd.isna(
                row[TrainCSVColumns.MISCONCEPTION_FORMAT.format(option=option)]
            ):
                continue

            actual_m_id = int(
                row[TrainCSVColumns.MISCONCEPTION_FORMAT.format(option=option)]
            )
            misconception_id_list, actual_m_index = populate_misconception_list(
                args,
                misconception_df.shape[0],
                actual_m_id,
            )
            misconceptions_text = ContrastiveCSVColumns.DELIMITER.join(
                misconception_df.loc[
                    misconception_df[MisconceptionsCSVColumns.MISCONCEPTION_ID] == m_id,
                    MisconceptionsCSVColumns.MISCONCEPTION_NAME,
                ].values[0]
                for m_id in misconception_id_list
            )

            row_query = query.format(
                subject=row[TrainCSVColumns.SUBJECT_NAME],
                construct=row[TrainCSVColumns.CONSTRUCT_NAME],
                question=row[TrainCSVColumns.QUESTION_TEXT],
                incorrect_answer=row[
                    TrainCSVColumns.ANSWER_FORMAT.format(option=option)
                ],
            )

            contrastive_df = pd.concat(
                [
                    contrastive_df,
                    create_new_row(
                        row,
                        actual_m_id,
                        actual_m_index,
                        misconceptions_text,
                        row_query,
                    ),
                ]
            )

    log_dataframe_artifact(
        contrastive_df,
        artifact_name=WandbProject.CONTRASTIVE_DATASET_NAME,
        artifact_type="dataset",
        description="""
        Contrastive dataset with random negative samples.

        Each row of the dataset contains the following columns:
        - `QuestionId`: Used for cv fold splitting.
        - `SubjectId`: Id of the subject. Used for subjectwise performance evaluation.
        - `ConstructId`: Id of the construct. Used for constructwise performance evaluation.
        - `QuestionDetails`: Formatted question details.
        - `MisconceptionList`: List of misconception texts.
        - `Label`: Index of the actual related misconception in the `MisconceptionList`.
        - `MisconceptionId`: Id of the actual related misconception. Used for calculating MAP@K score.
        """,
    )


def create_new_row(
    row: pd.Series,
    actual_m_id: int,
    actual_m_index: int,
    misconceptions_text: str,
    row_query: str,
) -> pd.DataFrame:
    """Create a new row for the contrastive dataset.

    Args:
        row (pd.Series): Row of the train dataset.
        actual_m_id (int): Id of the actual misconception.
        actual_m_index (int): Index of the actual misconception in the misconception list.
        misconceptions_text (str): Text of the misconception list.
        row_query (str): Formatted query for the question.

    Returns:
        pd.DataFrame: New row for the contrastive dataset.
    """
    return pd.DataFrame(
        {
            ContrastiveCSVColumns.QUESTION_ID: [row[TrainCSVColumns.QUESTION_ID]],
            ContrastiveCSVColumns.SUBJECT_ID: [row[TrainCSVColumns.SUBJECT_ID]],
            ContrastiveCSVColumns.CONSTRUCT_ID: [row[TrainCSVColumns.CONSTRUCT_ID]],
            ContrastiveCSVColumns.QUESTION_DETAILS: [row_query],
            ContrastiveCSVColumns.MISCONCEPTION_LIST: [misconceptions_text],
            ContrastiveCSVColumns.LABEL: [actual_m_index],
            ContrastiveCSVColumns.MISCONCEPTION_ID: [actual_m_id],
        }
    )


def populate_misconception_list(
    args: argparse.Namespace,
    total_misconception_count: int,
    actual_m_id: int,
) -> tuple[list[int], int]:
    """Populate the misconception list with random negative samples.

    Args:
        args (argparse.Namespace): Arguments.
        total_misconception_count (int): Total number of misconceptions in the dataset.
        actual_m_id (int): Id of the actual misconception.

    Throws:
        AssertionError: If the number of negative samples is not equal to the number of negative samples specified in the arguments.

    Returns:
        tuple[list[int], int]: List of negative misconception ids and the index of the actual misconception in the list.
    """
    negative_sample_set = set([actual_m_id])

    while len(negative_sample_set) < args.number_of_negative_samples:
        random_m_id = random.randint(0, total_misconception_count - 1)
        negative_sample_set.add(random_m_id)

    assert len(negative_sample_set) == args.number_of_negative_samples

    negative_sample_list = list(negative_sample_set)
    random.shuffle(negative_sample_list)

    return negative_sample_list, negative_sample_list.index(actual_m_id)


if __name__ == "__main__":
    args = parse_args()
    main(args)
