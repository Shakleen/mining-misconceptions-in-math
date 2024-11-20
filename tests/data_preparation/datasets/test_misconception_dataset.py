import pytest
import pandas as pd
from unittest.mock import Mock
import torch
from transformers import AutoTokenizer

from src.data_preparation.datasets.misconception_dataset import MisconceptionDataset
from src.constants.column_names import (
    QAPairCSVColumns,
    MisconceptionsCSVColumns,
    ContrastiveTorchDatasetColumns,
)

_EMBEDDING_SIZE = 512


@pytest.fixture(scope="module")
def misconceptions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            MisconceptionsCSVColumns.MISCONCEPTION_ID: [0, 1, 2, 3, 4],
            MisconceptionsCSVColumns.MISCONCEPTION_NAME: [
                "misconception 0",
                "misconception 1",
                "misconception 2",
                "misconception 3",
                "misconception 4",
            ],
        }
    )


@pytest.fixture(scope="module")
def tokenizer() -> Mock:
    mock = Mock(spec=AutoTokenizer)
    mock.return_value = {
        "input_ids": torch.randint(0, 100, (1, _EMBEDDING_SIZE)),
        "attention_mask": torch.randint(0, 2, (1, _EMBEDDING_SIZE)),
    }
    return mock


@pytest.fixture(scope="module")
def dataset(misconceptions_df: pd.DataFrame, tokenizer: Mock) -> MisconceptionDataset:
    return MisconceptionDataset(misconceptions_df, tokenizer)


def test_misconception_dataset_init(dataset: MisconceptionDataset):
    assert hasattr(dataset, "df")
    assert hasattr(dataset, "tokenizer")
    assert hasattr(dataset, "misconception_max_length")


def test_misconception_dataset_len(dataset: MisconceptionDataset):
    assert len(dataset) == len(dataset.df)


def test_misconception_dataset_getitem(dataset: MisconceptionDataset):
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert isinstance(item, dict)

        assert "input_ids" in item
        assert "attention_mask" in item

        assert item["input_ids"].shape == (_EMBEDDING_SIZE,)
        assert item["attention_mask"].shape == (_EMBEDDING_SIZE,)
