from typing import Optional, List, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

from src.constants.column_names import QAPairCSVColumns


class QuestionDetailsDataset(Dataset):
    """Dataset for question details.

    This dataset is used to get all question details. This is necessary when
    recall model needs to encode all question details.
    """

    @property
    def ColumnNames(self):
        return QAPairCSVColumns

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: Optional[int] = 64,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query = (
            "Subject: {subject}"
            + "\nConstruct: {construct}"
            + "\nQuestion: {question}"
            + "\nIncorrect Answer: {incorrect_answer}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        query_encoding = self._get_query_encoding(row)

        return {
            "input_ids": query_encoding["input_ids"].squeeze(0),
            "attention_mask": query_encoding["attention_mask"].squeeze(0),
        }

    def _get_query_encoding(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        question_text = row[self.ColumnNames.QUESTION_TEXT]
        subject_name = row[self.ColumnNames.SUBJECT_NAME]
        construct_name = row[self.ColumnNames.CONSTRUCT_NAME]
        answer_text = row[self.ColumnNames.ANSWER_TEXT]

        query_text = self.query.format(
            question=question_text,
            subject=subject_name,
            construct=construct_name,
            incorrect_answer=answer_text,
        )

        query_encoding = self.tokenizer(
            query_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return query_encoding

    def collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }
