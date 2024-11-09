import pytest
import pandas as pd
import torch
from unittest.mock import Mock
from transformers import AutoTokenizer

from src.data_preparation.get_dataloader import get_dataloader
from src.constants.column_names import (
    ContrastiveCSVColumns,
    ContrastiveTorchDatasetColumns,
)


@pytest.fixture(scope="module")
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ContrastiveCSVColumns.QUESTION_ID: [1] * 32,
            ContrastiveCSVColumns.SUBJECT_ID: [1] * 32,
            ContrastiveCSVColumns.CONSTRUCT_ID: [1] * 32,
            ContrastiveCSVColumns.QUESTION_DETAILS: ["question 1"] * 32,
            ContrastiveCSVColumns.MISCONCEPTION_LIST: ["a###b###c"] * 32,
            ContrastiveCSVColumns.LABEL: [0] * 32,
            ContrastiveCSVColumns.MISCONCEPTION_ID: [100] * 32,
        }
    )


@pytest.fixture(scope="module")
def tokenizer() -> Mock:
    mock = Mock(spec=AutoTokenizer)
    mock.return_value = {
        "input_ids": torch.randint(0, 100, (1, 32)),
        "attention_mask": torch.randint(0, 2, (1, 32)),
    }
    return mock


@pytest.mark.parametrize("batch_size", [8, 16])
def test_dataloader_without_meta_data(
    dataframe: pd.DataFrame,
    tokenizer: Mock,
    batch_size: int,
):
    dataloader = get_dataloader(dataframe, tokenizer, batch_size=batch_size)

    assert len(dataloader) == len(dataframe) // batch_size

    for batch in dataloader:
        assert ContrastiveTorchDatasetColumns.QUESTION_IDS in batch.keys()
        question_ids = batch[ContrastiveTorchDatasetColumns.QUESTION_IDS]
        assert question_ids.shape == (batch_size, 32)

        assert ContrastiveTorchDatasetColumns.QUESTION_MASK in batch.keys()
        question_mask = batch[ContrastiveTorchDatasetColumns.QUESTION_MASK]
        assert question_mask.shape == (batch_size, 32)

        assert ContrastiveTorchDatasetColumns.MISCONCEPTION_IDS in batch.keys()
        misconception_ids = batch[ContrastiveTorchDatasetColumns.MISCONCEPTION_IDS]
        assert misconception_ids.shape == (batch_size, 3, 32)

        assert ContrastiveTorchDatasetColumns.MISCONCEPTION_MASK in batch.keys()
        misconception_mask = batch[ContrastiveTorchDatasetColumns.MISCONCEPTION_MASK]
        assert misconception_mask.shape == (batch_size, 3, 32)

        assert ContrastiveTorchDatasetColumns.LABEL in batch.keys()
        label = batch[ContrastiveTorchDatasetColumns.LABEL]
        assert label.shape == (batch_size,)

        assert ContrastiveTorchDatasetColumns.META_DATA_QUESTION_ID not in batch.keys()
        assert ContrastiveTorchDatasetColumns.META_DATA_SUBJECT_ID not in batch.keys()
        assert ContrastiveTorchDatasetColumns.META_DATA_CONSTRUCT_ID not in batch.keys()
        assert (
            ContrastiveTorchDatasetColumns.META_DATA_MISCONCEPTION_ID
            not in batch.keys()
        )


@pytest.mark.parametrize("batch_size", [8, 16])
def test_dataloader_with_meta_data(
    dataframe: pd.DataFrame,
    tokenizer: Mock,
    batch_size: int,
):
    dataloader = get_dataloader(
        dataframe,
        tokenizer,
        batch_size=batch_size,
        include_meta_data=True,
    )

    assert len(dataloader) == len(dataframe) // batch_size

    for batch in dataloader:
        assert ContrastiveTorchDatasetColumns.QUESTION_IDS in batch.keys()
        question_ids = batch[ContrastiveTorchDatasetColumns.QUESTION_IDS]
        assert question_ids.shape == (batch_size, 32)

        assert ContrastiveTorchDatasetColumns.QUESTION_MASK in batch.keys()
        question_mask = batch[ContrastiveTorchDatasetColumns.QUESTION_MASK]
        assert question_mask.shape == (batch_size, 32)

        assert ContrastiveTorchDatasetColumns.MISCONCEPTION_IDS in batch.keys()
        misconception_ids = batch[ContrastiveTorchDatasetColumns.MISCONCEPTION_IDS]
        assert misconception_ids.shape == (batch_size, 3, 32)

        assert ContrastiveTorchDatasetColumns.MISCONCEPTION_MASK in batch.keys()
        misconception_mask = batch[ContrastiveTorchDatasetColumns.MISCONCEPTION_MASK]
        assert misconception_mask.shape == (batch_size, 3, 32)

        assert ContrastiveTorchDatasetColumns.LABEL in batch.keys()
        label = batch[ContrastiveTorchDatasetColumns.LABEL]
        assert label.shape == (batch_size,)

        assert ContrastiveTorchDatasetColumns.META_DATA_QUESTION_ID in batch.keys()
        question_ids = batch[ContrastiveTorchDatasetColumns.META_DATA_QUESTION_ID]
        assert question_ids.shape == (batch_size,)

        assert ContrastiveTorchDatasetColumns.META_DATA_SUBJECT_ID in batch.keys()
        subject_ids = batch[ContrastiveTorchDatasetColumns.META_DATA_SUBJECT_ID]
        assert subject_ids.shape == (batch_size,)

        assert ContrastiveTorchDatasetColumns.META_DATA_CONSTRUCT_ID in batch.keys()
        construct_ids = batch[ContrastiveTorchDatasetColumns.META_DATA_CONSTRUCT_ID]
        assert construct_ids.shape == (batch_size,)

        assert ContrastiveTorchDatasetColumns.META_DATA_MISCONCEPTION_ID in batch.keys()
        misconception_ids = batch[
            ContrastiveTorchDatasetColumns.META_DATA_MISCONCEPTION_ID
        ]
        assert misconception_ids.shape == (batch_size,)
