import pytest
import pandas as pd

from src.data_preparation.datasets.abstract_dataset import AbstractDataset
from src.constants.column_names import ContrastiveCSVColumns


class Dataset(AbstractDataset):
    def __getitem__(self, idx):
        return self.df.iloc[idx]


@pytest.fixture
def dataset():
    df = pd.DataFrame({"a": [1, 2, 3]})
    return Dataset(df, None)


def test_abstract_dataset():
    with pytest.raises(TypeError):
        AbstractDataset()


def test_has_abstract_methods():
    assert AbstractDataset.__abstractmethods__ == {"__getitem__"}


def test_len(dataset: Dataset):
    assert len(dataset) == 3


def test_getitem(dataset: Dataset):
    for i in range(len(dataset)):
        assert dataset[i]["a"] == i + 1


def test_split_misconception_list(dataset: Dataset):
    misconception_text = ContrastiveCSVColumns.DELIMITER.join(["a", "b", "c"])
    assert dataset.split_misconception_list(misconception_text) == [
        "a",
        "b",
        "c",
    ]


def test_parse_list_from_string(dataset: Dataset):
    string = "[1, 2, 3]"
    assert dataset.parse_list_from_string(string) == [1, 2, 3]
