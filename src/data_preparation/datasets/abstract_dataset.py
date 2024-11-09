from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

from src.constants.column_names import ContrastiveCSVColumns


class AbstractDataset(Dataset, ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        question_max_length: Optional[int] = 512,
        misconception_max_length: Optional[int] = 64,
        include_meta_data: Optional[bool] = False,
    ):
        """Initialize the dataset.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the dataset.
            tokenizer (AutoTokenizer): Tokenizer to use for the dataset.
            question_max_length (Optional[int], optional): Maximum length of the question. Defaults to 512.
            misconception_max_length (Optional[int], optional): Maximum length of the misconception. Defaults to 64.
            include_meta_data (Optional[bool], optional): Whether to include meta data. Defaults to False.
        """
        self.df = dataframe
        self.tokenizer = tokenizer
        self.question_max_length = question_max_length
        self.misconception_max_length = misconception_max_length
        self.include_meta_data = include_meta_data

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def split_misconception_list(self, misconception_text: str) -> list[str]:
        """Split a misconception text into a list of misconception texts.

        For example, if the misconception text is "a###b###c", the output will be ["a", "b", "c"].

        Args:
            misconception_text (str): Text to split.

        Returns:
            list[str]: List of misconception texts.
        """
        misconception_list = misconception_text.split(ContrastiveCSVColumns.DELIMITER)
        misconception_list = [
            misconception.strip() for misconception in misconception_list
        ]
        return misconception_list

    def parse_list_from_string(self, string: str) -> list[int]:
        """Parse a list from a string.

        Input: "[1, 2, 3]"
        Output: [1, 2, 3]

        Args:
            string (str): String to parse.

        Returns:
            list[int]: List of integers.
        """
        return [int(m) for m in string.replace("[", "").replace("]", "").split(",")]

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]):
        raise NotImplementedError("Subclasses must implement this method")
