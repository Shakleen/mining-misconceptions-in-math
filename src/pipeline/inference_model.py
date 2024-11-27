import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from src.constants.column_names import (
    TrainCSVColumns,
    QAPairCSVColumns,
    SubmissionCSVColumns,
)
from src.data_preparation.datasets.question_details_dataset import (
    QuestionDetailsDataset,
)
from src.model_development.two_tower_model import TwoTowerModel
from src.configurations.recall_model_config import RecallModelConfig
from src.pipeline.embbed_misconceptions import (
    get_misconception_embeddings,
    create_misconception_dataloader,
)


def convert_to_qa_pair(df: pd.DataFrame) -> pd.DataFrame:
    contrastive_df = pd.DataFrame()

    for _, row in df.iterrows():
        for option in ["A", "B", "C", "D"]:
            if option == row[TrainCSVColumns.CORRECT_ANSWER]:
                continue

            contrastive_df = pd.concat(
                [
                    contrastive_df,
                    pd.DataFrame(
                        {
                            QAPairCSVColumns.QUESTION_TEXT: [
                                row[TrainCSVColumns.QUESTION_TEXT]
                            ],
                            QAPairCSVColumns.SUBJECT_NAME: [
                                row[TrainCSVColumns.SUBJECT_NAME]
                            ],
                            QAPairCSVColumns.CONSTRUCT_NAME: [
                                row[TrainCSVColumns.CONSTRUCT_NAME]
                            ],
                            QAPairCSVColumns.ANSWER_TEXT: [
                                row[TrainCSVColumns.ANSWER_FORMAT.format(option=option)]
                            ],
                            QAPairCSVColumns.QUESTION_ID: [
                                f"{row[TrainCSVColumns.QUESTION_ID]}_{option}"
                            ],
                        }
                    ),
                ]
            )

    return contrastive_df


def generate_submission_df(
    dataloader: DataLoader,
    misconception_embeddings: torch.Tensor,
    model: TwoTowerModel,
) -> pd.DataFrame:
    submission_df = pd.DataFrame()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Generating submission dataframe"
        ):
            question_ids = batch["input_ids"].to(model.device)
            question_mask = batch["attention_mask"].to(model.device)

            q_embeddings = (
                model.get_query_features(question_ids, question_mask).detach().cpu()
            )

            similarities = q_embeddings @ misconception_embeddings.T
            rankings = similarities.argsort(dim=1, descending=True)[:, :25].tolist()

            ids = qa_df.iloc[batch["index"].tolist()][
                QAPairCSVColumns.QUESTION_ID
            ].tolist()

            mids = [" ".join(str(x) for x in ranking) for ranking in rankings]

            print(len(ids), len(mids))

            submission_df = pd.concat(
                [
                    submission_df,
                    pd.DataFrame(
                        {
                            SubmissionCSVColumns.QUESTION_ID_ANSWER: ids,
                            SubmissionCSVColumns.MISCONCEPTION_ID: mids,
                        }
                    ),
                ]
            )

    return submission_df


def inference(
    df: pd.DataFrame,
    misconceptions_df: pd.DataFrame,
    model: TwoTowerModel,
    tokenizer: AutoTokenizer,
) -> pd.DataFrame:
    qa_df = convert_to_qa_pair(df)
    dataset = QuestionDetailsDataset(qa_df, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    misconception_dataloader = create_misconception_dataloader(
        misconceptions_df, tokenizer, batch_size=16, num_workers=4
    )

    misconception_embeddings = get_misconception_embeddings(
        misconception_dataloader, model
    )

    return generate_submission_df(dataloader, misconception_embeddings, model)


if __name__ == "__main__":
    df = pd.read_csv("data/test-datasetdgy7k5rw.csv")
    misconceptions_df = pd.read_csv("data/misconceptions-datasetas216_mx.csv")

    qa_df = convert_to_qa_pair(df)

    tokenizer = AutoTokenizer.from_pretrained(".cache/deberta-v3-base/")
    model_config = RecallModelConfig.from_json("config/recall_model_config.json")
    model = TwoTowerModel.load_from_checkpoint(
        "output_dir/Kaggle_EEDI/351xrmvk/checkpoints/best-checkpoint-0.ckpt",
        config=model_config,
    ).eval()

    submission_df = inference(df, misconceptions_df, model, tokenizer)

    submission_df.to_csv("submission.csv", index=False)
