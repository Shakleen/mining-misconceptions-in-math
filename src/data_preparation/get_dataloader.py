import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_preparation.datasets.abstract_dataset import AbstractDataset
from src.data_preparation.datasets.base_dataset import BaseDataset


def get_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    DatasetClass: Optional[AbstractDataset] = BaseDataset,
    batch_size: Optional[int] = 8,
    num_workers: Optional[int] = 8,
    shuffle: Optional[bool] = False,
    include_meta_data: Optional[bool] = False,
) -> DataLoader:
    """Get a dataloader for a given dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
        DatasetClass (Optional[AbstractDataset]): Dataset class to use.
        batch_size (Optional[int]): Batch size. Defaults to 8.
        num_workers (Optional[int]): Number of workers. Defaults to 8.
        shuffle (Optional[bool]): Whether to shuffle the dataset. Defaults to False.
        include_meta_data (Optional[bool]): Whether to include the meta data.
        Defaults to False.

    Returns:
        DataLoader: Dataloader for the dataset.
    """
    dataset = DatasetClass(df, tokenizer, include_meta_data=include_meta_data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: dataset.collate_fn(x, include_meta_data),
    )
