import pytest
import pandas as pd
import torch
from unittest.mock import Mock
from typing import List
from transformers import AutoTokenizer

from src.data_preparation.datasets.base_dataset_v2 import BaseDatasetV2
from src.constants.column_names import (
    QAPairCSVColumns,
    MisconceptionsCSVColumns,
    ContrastiveTorchDatasetColumns,
)
from src.data_preparation.negative_sampler.random_sampler import RandomNegativeSampler
from src.utils.seed_everything import seed_everything


_EMBEDDING_SIZE = 512
_NEGATIVE_SAMPLE_SIZE = 2

seed_everything(20)


@pytest.fixture(scope="module")
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            QAPairCSVColumns.SUBJECT_ID: [1, 2],
            QAPairCSVColumns.SUBJECT_NAME: ["subject 1", "subject 2"],
            QAPairCSVColumns.CONSTRUCT_ID: [1, 2],
            QAPairCSVColumns.CONSTRUCT_NAME: ["construct 1", "construct 2"],
            QAPairCSVColumns.QUESTION_ID: [1, 2],
            QAPairCSVColumns.QUESTION_TEXT: ["question 1", "question 2"],
            QAPairCSVColumns.ANSWER_TEXT: ["answer 1", "answer 2"],
            QAPairCSVColumns.MISCONCEPTION_ID: [1, 2],
            QAPairCSVColumns.MISCONCEPTION_NAME: ["misconception 1", "misconception 2"],
        }
    )


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
def dataset(
    dataframe: pd.DataFrame, tokenizer: Mock, misconceptions_df: pd.DataFrame
) -> BaseDatasetV2:
    random_negative_sampler = RandomNegativeSampler(
        sample_size=_NEGATIVE_SAMPLE_SIZE,
        total_misconceptions=5,
    )
    return BaseDatasetV2(
        dataframe,
        misconceptions_df,
        tokenizer,
        random_negative_sampler,
    )


def test_base_dataset_v2_init(dataframe: pd.DataFrame, dataset: BaseDatasetV2):
    assert hasattr(dataset, "misconceptions_df")
    assert hasattr(dataset, "negative_sampler")
    assert hasattr(dataset, "query")

    assert len(dataset) == len(dataframe)


def test_property_CSVColName(dataset: BaseDatasetV2):
    assert dataset.CSVColName == QAPairCSVColumns


def test_property_TorchColName(dataset: BaseDatasetV2):
    assert dataset.TorchColName == ContrastiveTorchDatasetColumns


@pytest.mark.parametrize("negative_sample_id_list", [[1, 2], [1, 2, 3, 4]])
def test_get_misconception_encodings(
    dataset: BaseDatasetV2, negative_sample_id_list: List[int]
):
    misconception_encodings = dataset._get_misconception_encodings(
        negative_sample_id_list
    )
    assert len(misconception_encodings) == len(negative_sample_id_list)

    for encoding in misconception_encodings:
        assert isinstance(encoding, dict)
        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert encoding["input_ids"].shape == (1, _EMBEDDING_SIZE)
        assert encoding["attention_mask"].shape == (1, _EMBEDDING_SIZE)


def test_get_query_encoding(dataset: BaseDatasetV2, dataframe: pd.DataFrame):
    query_encoding = dataset._get_query_encoding(dataframe.iloc[0])

    assert isinstance(query_encoding, dict)
    assert "input_ids" in query_encoding
    assert "attention_mask" in query_encoding
    assert query_encoding["input_ids"].shape == (1, _EMBEDDING_SIZE)
    assert query_encoding["attention_mask"].shape == (1, _EMBEDDING_SIZE)


def test_add_meta_data(dataset: BaseDatasetV2, dataframe: pd.DataFrame):
    output = {}
    dataset._add_meta_adata(dataframe.iloc[0], output)

    assert isinstance(output, dict)
    assert dataset.TorchColName.META_DATA_QUESTION_ID in output
    assert dataset.TorchColName.META_DATA_SUBJECT_ID in output
    assert dataset.TorchColName.META_DATA_CONSTRUCT_ID in output
    assert dataset.TorchColName.META_DATA_MISCONCEPTION_ID in output
    assert (
        output[dataset.TorchColName.META_DATA_QUESTION_ID].item()
        == dataframe.iloc[0][dataset.CSVColName.QUESTION_ID]
    )
    assert (
        output[dataset.TorchColName.META_DATA_SUBJECT_ID].item()
        == dataframe.iloc[0][dataset.CSVColName.SUBJECT_ID]
    )
    assert (
        output[dataset.TorchColName.META_DATA_CONSTRUCT_ID].item()
        == dataframe.iloc[0][dataset.CSVColName.CONSTRUCT_ID]
    )
    assert (
        output[dataset.TorchColName.META_DATA_MISCONCEPTION_ID].item()
        == dataframe.iloc[0][dataset.CSVColName.MISCONCEPTION_ID]
    )


@pytest.mark.parametrize("negative_sample_id_list", [[1, 2], [1, 2, 3, 4]])
def test_get_output_dict(
    dataset: BaseDatasetV2,
    dataframe: pd.DataFrame,
    negative_sample_id_list: List[int],
):
    misconception_encodings = dataset._get_misconception_encodings(
        negative_sample_id_list
    )
    query_encoding = dataset._get_query_encoding(dataframe.iloc[0])

    output = dataset._get_output_dict(
        dataframe.iloc[0], query_encoding, 0, misconception_encodings
    )

    assert isinstance(output, dict)
    assert dataset.TorchColName.QUESTION_IDS in output
    assert dataset.TorchColName.QUESTION_MASK in output
    assert dataset.TorchColName.MISCONCEPTION_IDS in output
    assert dataset.TorchColName.MISCONCEPTION_MASK in output
    assert dataset.TorchColName.LABEL in output

    assert output[dataset.TorchColName.QUESTION_IDS].shape == (_EMBEDDING_SIZE,)
    assert output[dataset.TorchColName.QUESTION_MASK].shape == (_EMBEDDING_SIZE,)
    assert output[dataset.TorchColName.MISCONCEPTION_IDS].shape == (
        len(negative_sample_id_list),
        _EMBEDDING_SIZE,
    )
    assert output[dataset.TorchColName.MISCONCEPTION_MASK].shape == (
        len(negative_sample_id_list),
        _EMBEDDING_SIZE,
    )
    assert output[dataset.TorchColName.LABEL].shape == ()


def test_get_item(dataset: BaseDatasetV2):
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert item[dataset.TorchColName.QUESTION_IDS].shape == (_EMBEDDING_SIZE,)
        assert item[dataset.TorchColName.QUESTION_MASK].shape == (_EMBEDDING_SIZE,)
        assert item[dataset.TorchColName.MISCONCEPTION_IDS].shape == (
            _NEGATIVE_SAMPLE_SIZE,
            _EMBEDDING_SIZE,
        )
        assert item[dataset.TorchColName.MISCONCEPTION_MASK].shape == (
            _NEGATIVE_SAMPLE_SIZE,
            _EMBEDDING_SIZE,
        )
        assert item[dataset.TorchColName.LABEL].shape == ()


def test_collate_fn(dataset: BaseDatasetV2):
    batch = [dataset[i] for i in range(len(dataset))]
    collated_batch = dataset.collate_fn(batch)
    assert isinstance(collated_batch, dict)
    assert collated_batch[dataset.TorchColName.QUESTION_IDS].shape == (
        len(batch),
        _EMBEDDING_SIZE,
    )
    assert collated_batch[dataset.TorchColName.QUESTION_MASK].shape == (
        len(batch),
        _EMBEDDING_SIZE,
    )
    assert collated_batch[dataset.TorchColName.MISCONCEPTION_IDS].shape == (
        len(batch),
        _NEGATIVE_SAMPLE_SIZE,
        _EMBEDDING_SIZE,
    )
    assert collated_batch[dataset.TorchColName.MISCONCEPTION_MASK].shape == (
        len(batch),
        _NEGATIVE_SAMPLE_SIZE,
        _EMBEDDING_SIZE,
    )
    assert collated_batch[dataset.TorchColName.LABEL].shape == (len(batch),)
