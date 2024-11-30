import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import numpy as np

from src.configurations.recall_model_config import RecallModelConfig
from src.model_development.two_tower_model import TwoTowerModel
from src.data_preparation.datasets.misconception_dataset import MisconceptionDataset


def get_misconception_embeddings(
    misconception_dataloader: DataLoader,
    model: TwoTowerModel,
) -> torch.Tensor:
    return torch.cat(
        [
            model.get_docs_features(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            )
            .detach()
            .cpu()
            for batch in tqdm(
                misconception_dataloader,
                total=len(misconception_dataloader),
                desc="Generating misconception embeddings",
            )
        ],
        dim=0,
    )


def create_misconception_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = MisconceptionDataset(df, tokenizer, 32)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Embed misconceptions")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_path)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    dataloader = create_misconception_dataloader(
        df, tokenizer, args.batch_size, args.num_workers
    )

    model_config = RecallModelConfig.from_json("config/recall_model_config.json")
    model = TwoTowerModel.load_from_checkpoint(args.model_path, config=model_config)
    embeddings = get_misconception_embeddings(dataloader, model)

    print("Embeddings shape:", embeddings.shape)
    np.save(args.output_path, embeddings.numpy())
