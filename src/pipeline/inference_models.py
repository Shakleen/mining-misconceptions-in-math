# This script performs inference using multiple candidate generation models.
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from src.pipeline.inference_model import generate_submission_df, convert_to_qa_pair
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
    checkpoint_path: str,
    model_config: RecallModelConfig,
    candidate_count: int = 1000,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    dataset = QuestionDetailsDataset(qa_df, tokenizer, max_length=256)
    qa_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    misconception_dataloader = create_misconception_dataloader(
        misconceptions_df, tokenizer, batch_size=16, num_workers=4
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


def ranking():
    # Model 2: Ranking [25]
    #   1. Embed misconceptions
    #   2. Process question. Select best 25 from narrowed down list of misconceptions
    pass


if __name__ == "__main__":
    df = pd.read_csv("data/test-datasetdgy7k5rw.csv")
    misconceptions_df = pd.read_csv("data/misconception_dataset.csv")

    qa_df = convert_to_qa_pair(df)

    model_config = RecallModelConfig.from_json("config/recall_model_config.json")
    model_config.model_path = "/media/ishrak/volume_1/Projects/mining-misconceptions-in-math/.cache/deberta-v3-base"

    rankings = candidate_generation(
        checkpoint_path="output_dir/best-checkpoint.ckpt",
        model_config=model_config,
        candidate_count=1000,
    )
    qa_df["rankings"] = rankings

    print(qa_df.head())
