import pytest
import pandas as pd
from unittest.mock import Mock
import torch
from transformers import AutoTokenizer

from src.data_preparation.datasets.question_details_dataset import (
    QuestionDetailsDataset,
)
from src.constants.column_names import TrainCSVColumns
from src.constants.column_names import (
    QAPairCSVColumns,
)

_EMBEDDING_SIZE = 512


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
def tokenizer() -> Mock:
    mock = Mock(spec=AutoTokenizer)
    mock.return_value = {
        "input_ids": torch.randint(0, 100, (1, _EMBEDDING_SIZE)),
        "attention_mask": torch.randint(0, 2, (1, _EMBEDDING_SIZE)),
    }
    return mock


@pytest.fixture(scope="module")
def dataset(dataframe: pd.DataFrame, tokenizer: Mock) -> QuestionDetailsDataset:
    return QuestionDetailsDataset(dataframe, tokenizer)


def test_question_details_dataset_init(dataset: QuestionDetailsDataset):
    assert hasattr(dataset, "df")
    assert hasattr(dataset, "tokenizer")
    assert hasattr(dataset, "max_length")
    assert hasattr(dataset, "query")


def test_question_details_dataset_len(dataset: QuestionDetailsDataset):
    assert len(dataset) == len(dataset.df)


def test_question_details_dataset_getitem(dataset: QuestionDetailsDataset):
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item

        assert item["input_ids"].shape == (_EMBEDDING_SIZE,)
        assert item["attention_mask"].shape == (_EMBEDDING_SIZE,)
