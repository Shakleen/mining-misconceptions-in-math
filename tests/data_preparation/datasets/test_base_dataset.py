import pytest
import pandas as pd
import torch
from unittest.mock import Mock
from transformers import AutoTokenizer

from src.data_preparation.datasets.base_dataset import BaseDataset
from src.constants.column_names import (
    ContrastiveCSVColumns,
    ContrastiveTorchDatasetColumns,
)


@pytest.fixture(scope="module")
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ContrastiveCSVColumns.QUESTION_ID: [1, 2],
            ContrastiveCSVColumns.SUBJECT_ID: [1, 2],
            ContrastiveCSVColumns.CONSTRUCT_ID: [1, 2],
            ContrastiveCSVColumns.QUESTION_DETAILS: ["question 1", "question 2"],
            ContrastiveCSVColumns.MISCONCEPTION_LIST: ["a###b###c", "d###e###f"],
            ContrastiveCSVColumns.LABEL: [0, 1],
            ContrastiveCSVColumns.MISCONCEPTION_ID: [100, 200],
        }
    )


@pytest.fixture(scope="module")
def tokenizer() -> Mock:
    mock = Mock(spec=AutoTokenizer)
    mock.return_value = {
        "input_ids": torch.randint(0, 100, (1, 512)),
        "attention_mask": torch.randint(0, 2, (1, 512)),
    }
    return mock


@pytest.fixture(scope="module")
def dataset(dataframe: pd.DataFrame, tokenizer: Mock) -> BaseDataset:
    return BaseDataset(dataframe, tokenizer)


def test_base_dataset_init(dataframe: pd.DataFrame, dataset: BaseDataset):
    assert len(dataset) == len(dataframe)


def test_base_dataset_getitem(dataset: BaseDataset, dataframe: pd.DataFrame):
    for i in range(len(dataset)):
        item = dataset[i]

        assert isinstance(item, dict)
        assert item[ContrastiveTorchDatasetColumns.QUESTION_IDS].shape == (512,)
        assert item[ContrastiveTorchDatasetColumns.QUESTION_MASK].shape == (512,)
        assert item[ContrastiveTorchDatasetColumns.MISCONCEPTION_IDS].shape == (3, 512)
        assert item[ContrastiveTorchDatasetColumns.MISCONCEPTION_MASK].shape == (3, 512)
        assert (
            item[ContrastiveTorchDatasetColumns.LABEL].item()
            == dataframe.iloc[i][ContrastiveCSVColumns.LABEL]
        )


def test_base_dataset_include_meta_data(dataframe: pd.DataFrame, tokenizer: Mock):
    dataset = BaseDataset(dataframe, tokenizer, include_meta_data=True)

    for i in range(len(dataset)):
        item = dataset[i]

        assert (
            item[ContrastiveTorchDatasetColumns.META_DATA_QUESTION_ID].item()
            == dataframe.iloc[i][ContrastiveCSVColumns.QUESTION_ID]
        )
        assert (
            item[ContrastiveTorchDatasetColumns.META_DATA_SUBJECT_ID].item()
            == dataframe.iloc[i][ContrastiveCSVColumns.SUBJECT_ID]
        )
        assert (
            item[ContrastiveTorchDatasetColumns.META_DATA_CONSTRUCT_ID].item()
            == dataframe.iloc[i][ContrastiveCSVColumns.CONSTRUCT_ID]
        )
        assert (
            item[ContrastiveTorchDatasetColumns.META_DATA_MISCONCEPTION_ID].item()
            == dataframe.iloc[i][ContrastiveCSVColumns.MISCONCEPTION_ID]
        )


