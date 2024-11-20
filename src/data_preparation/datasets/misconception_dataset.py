from typing import Optional, List, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

from src.constants.column_names import MisconceptionsCSVColumns


class MisconceptionDataset(Dataset):
    """Dataset for misconceptions.

    This dataset is used to get all misconception texts. This is necessary when
    recall model needs to encode all misconception texts.
    """

    @property
    def ColumnNames(self):
        return MisconceptionsCSVColumns

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        misconception_max_length: Optional[int] = 64,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.misconception_max_length = misconception_max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        misconception_text = row[self.ColumnNames.MISCONCEPTION_NAME]
        encoded_misconception = self.tokenizer(
            misconception_text,
            max_length=self.misconception_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_misconception["input_ids"].squeeze(0),
            "attention_mask": encoded_misconception["attention_mask"].squeeze(0),
        }

    def collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }
