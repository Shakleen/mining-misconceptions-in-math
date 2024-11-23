from typing import List, Tuple
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

from src.model_development.recall_model import RecallModel
from src.data_preparation.datasets.misconception_dataset import MisconceptionDataset
from src.constants.column_names import ContrastiveTorchDatasetColumns
from src.utils.searcher.similarity_searcher import SimilaritySearcher
from src.constants.dll_paths import DLLPaths


def inference(
    best_model: RecallModel,
    misconception_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
    val_loader: DataLoader,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Inference for the recall model.

    Args:
        best_model (RecallModel): Best model.
        misconception_df (pd.DataFrame): Misconception dataframe.
        tokenizer (AutoTokenizer): Tokenizer.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        val_loader (DataLoader): Validation data loader.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Predictions and labels.
    """
    misconception_dataloader = create_misconception_dataloader(
        misconception_df,
        tokenizer,
        batch_size,
        num_workers,
    )
    searcher = SimilaritySearcher(DLLPaths.SIMILARITY_SEARCH)

    with torch.no_grad():
        all_misconceptions_embeddings = get_misconception_embeddings(
            misconception_dataloader, best_model
        )

        all_labels = None
        all_preds = None

        for batch in tqdm(
            val_loader,
            total=len(val_loader),
            desc="MAP scoring validation set",
        ):
            batch = {k: v.to(best_model.device) for k, v in batch.items()}
            question_embeddings = (
                best_model.get_features(
                    input_ids=batch[ContrastiveTorchDatasetColumns.QUESTION_IDS],
                    attention_mask=batch[
                        ContrastiveTorchDatasetColumns.QUESTION_MASK
                    ],
                )
                .detach()
                .cpu()
                .numpy()
            )

            top_k_misconceptions = searcher.batch_search(
                question_embeddings,
                all_misconceptions_embeddings,
                k=25,
            )

            if all_preds is None:
                all_preds = top_k_misconceptions
            else:
                all_preds = np.concatenate([all_preds, top_k_misconceptions])

            if all_labels is None:
                all_labels = batch[ContrastiveTorchDatasetColumns.LABEL].detach().cpu().numpy()
            else:
                all_labels = np.concatenate([all_labels, batch[ContrastiveTorchDatasetColumns.LABEL].detach().cpu().numpy()])

    return all_preds, all_labels


def get_misconception_embeddings(
    misconception_dataloader: DataLoader,
    best_model: RecallModel,
) -> torch.Tensor:
    return torch.cat(
        [
            best_model.get_features(
                input_ids=batch["input_ids"].to(best_model.device),
                attention_mask=batch["attention_mask"].to(best_model.device),
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
    dataset = MisconceptionDataset(df, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )