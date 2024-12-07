# This script performs inference using multiple candidate generation models.
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from src.pipeline.inference_model import convert_to_qa_pair
from src.model_development.two_tower_model import TwoTowerModel
from src.configurations.recall_model_config import RecallModelConfig
from src.constants.column_names import (
    TrainCSVColumns,
    QAPairCSVColumns,
    SubmissionCSVColumns,
)
from src.data_preparation.datasets.question_details_dataset import (
    QuestionDetailsDataset,
)
from src.pipeline.embbed_misconceptions import (
    get_misconception_embeddings,
    create_misconception_dataloader,
)


def candidate_generation(
    qa_df: pd.DataFrame,
    misconceptions_df: pd.DataFrame,
    checkpoint_path: str,
    model_config: RecallModelConfig,
    candidate_count: int = 1000,
    batch_size: int = 16,
    num_workers: int = 4,
    qa_max_length: int = 256,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    dataset = QuestionDetailsDataset(qa_df, tokenizer, max_length=qa_max_length)
    qa_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    misconception_dataloader = create_misconception_dataloader(
        misconceptions_df,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = TwoTowerModel.load_from_checkpoint(
        checkpoint_path, config=model_config
    ).eval()

    misconception_embeddings = get_misconception_embeddings(
        misconception_dataloader, model
    )

    all_rankings = []

    with torch.no_grad():
        for batch in tqdm(
            qa_dataloader,
            total=len(qa_dataloader),
            desc=f"Candidate generation for {candidate_count} misconceptions",
        ):
            question_ids = batch["input_ids"].to(model.device)
            question_mask = batch["attention_mask"].to(model.device)

            q_embeddings = (
                model.get_query_features(question_ids, question_mask).detach().cpu()
            )

            similarities = q_embeddings @ misconception_embeddings.T
            rankings = similarities.argsort(dim=1, descending=True)[
                :, :candidate_count
            ].tolist()

            for ranking in rankings:
                all_rankings.append(ranking)

    return all_rankings


def generate_submission_df(
    qa_df: pd.DataFrame,
    misconceptions_df: pd.DataFrame,
    checkpoint_path: str,
    model_config: RecallModelConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    qa_max_length: int = 256,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    dataset = QuestionDetailsDataset(qa_df, tokenizer, max_length=qa_max_length)
    qa_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    misconception_dataloader = create_misconception_dataloader(
        misconceptions_df,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = TwoTowerModel.load_from_checkpoint(
        checkpoint_path, config=model_config
    ).eval()

    misconception_embeddings = get_misconception_embeddings(
        misconception_dataloader, model
    )

    idx = 0
    submission_df = pd.DataFrame()

    with torch.no_grad():
        for batch in tqdm(
            qa_dataloader,
            total=len(qa_dataloader),
            desc=f"Ranking misconceptions",
        ):
            question_ids = batch["input_ids"].to(model.device)
            question_mask = batch["attention_mask"].to(model.device)

            q_embeddings = (
                model.get_query_features(question_ids, question_mask).detach().cpu()
            )

            m_embeddings = []

            for i in range(idx, min(idx + batch_size, len(qa_df))):
                ranking = qa_df.iloc[i]["rankings"]
                m_embeddings.append(misconception_embeddings[ranking])

            m_embeddings = torch.stack(m_embeddings).view(-1, misconception_embeddings.shape[1])

            similarities = q_embeddings @ m_embeddings.T
            rankings = similarities.argsort(dim=1, descending=True)[:, :25].tolist()

            ids = qa_df.iloc[batch["index"].tolist()][
                QAPairCSVColumns.QUESTION_ID
            ].tolist()

            mids = [" ".join(str(x) for x in ranking) for ranking in rankings]

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

            idx += batch_size

    return submission_df


if __name__ == "__main__":
    df = pd.read_csv("data/test-datasetdgy7k5rw.csv")
    misconceptions_df = pd.read_csv("data/misconception_dataset.csv")

    qa_df = convert_to_qa_pair(df)

    model_config = RecallModelConfig.from_json("config/recall_model_config.json")
    model_config.model_path = "/media/ishrak/volume_1/Projects/mining-misconceptions-in-math/.cache/deberta-v3-base"

    rankings = candidate_generation(
        qa_df=qa_df,
        misconceptions_df=misconceptions_df,
        checkpoint_path="output_dir/best-checkpoint.ckpt",
        model_config=model_config,
        candidate_count=1000,
    )
    qa_df["rankings"] = rankings

    model_config.model_path = "/media/ishrak/volume_1/Projects/mining-misconceptions-in-math/.cache/deberta-v3-large"
    submission_df = generate_submission_df(
        qa_df=qa_df,
        misconceptions_df=misconceptions_df,
        checkpoint_path="output_dir/TT-tsqlufea/Kaggle_EEDI/x0kjdld5/checkpoints/best-checkpoint.ckpt",
        model_config=model_config,
    )
    submission_df.to_csv("submission_df.csv", index=False)
