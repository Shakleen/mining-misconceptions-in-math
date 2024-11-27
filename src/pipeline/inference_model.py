import pandas as pd

from src.constants.column_names import (
    TrainCSVColumns,
    MisconceptionsCSVColumns,
    ContrastiveCSVColumns,
    SubmissionCSVColumns,
)


def convert_to_qa_pair(df: pd.DataFrame) -> pd.DataFrame:
    query = (
        "Given subject, construct, question and incorrect answer, "
        + "retrieve the list of misconceptions that are related to the incorrect answer."
        + "\nSubject: {subject}"
        + "\nConstruct: {construct}"
        + "\nQuestion: {question}"
        + "\nIncorrect Answer: {incorrect_answer}"
    )

    contrastive_df = pd.DataFrame()

    for _, row in df.iterrows():
        for option in ["A", "B", "C", "D"]:
            if option == row[TrainCSVColumns.CORRECT_ANSWER]:
                continue

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
                    pd.DataFrame(
                        {
                            SubmissionCSVColumns.QUESTION_ID_ANSWER: [
                                f"{row[TrainCSVColumns.QUESTION_ID]}_{option}"
                            ],
                            ContrastiveCSVColumns.QUESTION_DETAILS: [row_query],
                        }
                    ),
                ]
            )

    return contrastive_df


if __name__ == "__main__":
    df = pd.read_csv("data/test-datasetdgy7k5rw.csv")
    contrastive_df = convert_to_qa_pair(df)
    
