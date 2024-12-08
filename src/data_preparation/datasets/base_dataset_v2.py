from typing import Optional, List, Dict

import torch
import pandas as pd
from transformers import AutoTokenizer

from src.data_preparation.datasets.abstract_dataset import AbstractDataset
from src.constants.column_names import (
    QAPairCSVColumns,
    ContrastiveTorchDatasetColumns,
    MisconceptionsCSVColumns,
)
from src.data_preparation.negative_sampler.abstract_negative_sampler import (
    AbstractNegativeSampler,
)


class BaseDatasetV2(AbstractDataset):
    """Base dataset for the contrastive learning task.

    The dataset returns the following rows:
    - question_ids: List[int]
    - question_mask: List[int]
    - misconception_ids: List[List[int]]
    - misconception_mask: List[List[int]]
    - label: int

    If include_meta_data is True, the dataset also returns the following rows:
    - meta_data_question_id: int
    - meta_data_subject_id: int
    - meta_data_construct_id: int
    - meta_data_misconception_id: int

    Metadata should only be included for validation or test sets. These are used
    for assessing performance across different subjects, constructs and question ids.

    This version of dataset uses QA-pairs dataframe and misconceptions dataframe. Unlike
    previous version, it uses a `negative_sampler` to sample negative misconceptions.

    Example usage:
    ```python
    from transformers import AutoTokenizer
    from src.data_preparation.negative_sampler.random_negative_sampler import RandomNegativeSampler

    tokenizer = AutoTokenizer.from_pretrained("...")
    negative_sampler = RandomNegativeSampler(sample_size=10, total_misconceptions=100)
    dataset = BaseDatasetV2(dataframe, misconceptions_df, tokenizer, negative_sampler)
    ```
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        misconceptions_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        negative_sampler: AbstractNegativeSampler,
        question_max_length: Optional[int] = 512,
        misconception_max_length: Optional[int] = 64,
        include_meta_data: Optional[bool] = False,
    ):
        super().__init__(
            dataframe,
            tokenizer,
            question_max_length,
            misconception_max_length,
            include_meta_data,
        )
        self.misconceptions_df = misconceptions_df
        self.negative_sampler = negative_sampler
        self.query = (
            "Instruct: Given subject name, construct name, question, and incorrect answer, retrieve relevant misconceptions."
            + "\nSubject: {subject}"
            + "\nConstruct: {construct}"
            + "\nQuestion: {question}"
            + "\nIncorrect Answer: {incorrect_answer}"
        )

    @property
    def CSVColName(self):
        return QAPairCSVColumns

    @property
    def TorchColName(self):
        return ContrastiveTorchDatasetColumns

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        query_encoding = self._get_query_encoding(row)

        misconception_id = int(row[self.CSVColName.MISCONCEPTION_ID])

        negative_sample_id_list = self.negative_sampler.sample(misconception_id)
        label = negative_sample_id_list.index(misconception_id)

        misconception_encodings = self._get_misconception_encodings(
            negative_sample_id_list
        )

        output = self._get_output_dict(
            row,
            query_encoding,
            label,
            misconception_encodings,
        )

        return output

    def _get_output_dict(
        self,
        row: pd.Series,
        query_encoding: Dict[str, torch.Tensor],
        label: int,
        misconception_encodings: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        output = {
            self.TorchColName.QUESTION_IDS: query_encoding["input_ids"].squeeze(0),
            self.TorchColName.QUESTION_MASK: query_encoding["attention_mask"].squeeze(
                0
            ),
            self.TorchColName.MISCONCEPTION_IDS: torch.stack(
                [m["input_ids"].squeeze(0) for m in misconception_encodings]
            ),
            self.TorchColName.MISCONCEPTION_MASK: torch.stack(
                [m["attention_mask"].squeeze(0) for m in misconception_encodings]
            ),
            self.TorchColName.LABEL: torch.tensor(label, dtype=torch.long),
        }

        if self.include_meta_data:
            self._add_meta_adata(row, output)

        return output

    def _add_meta_adata(self, row: pd.Series, output: Dict[str, torch.Tensor]):
        # output[self.TorchColName.META_DATA_QUESTION_ID] = torch.tensor(
        #     row[self.CSVColName.QUESTION_ID],
        #     dtype=torch.int32,
        # )
        # output[self.TorchColName.META_DATA_SUBJECT_ID] = torch.tensor(
        #     row[self.CSVColName.SUBJECT_ID],
        #     dtype=torch.int32,
        # )
        # output[self.TorchColName.META_DATA_CONSTRUCT_ID] = torch.tensor(
        #     row[self.CSVColName.CONSTRUCT_ID],
        #     dtype=torch.int32,
        # )
        output[self.TorchColName.META_DATA_MISCONCEPTION_ID] = torch.tensor(
            row[self.CSVColName.MISCONCEPTION_ID],
            dtype=torch.long,
        )

    def _get_misconception_encodings(
        self,
        negative_sample_id_list: List[int],
    ) -> List[Dict[str, torch.Tensor]]:
        negative_sample_text_list = [
            self.misconceptions_df.loc[
                self.misconceptions_df[MisconceptionsCSVColumns.MISCONCEPTION_ID]
                == m_id,
                MisconceptionsCSVColumns.MISCONCEPTION_NAME,
            ].values[0]
            for m_id in negative_sample_id_list
        ]
        misconception_encodings = [
            self.tokenizer(
                misc,
                max_length=self.misconception_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for misc in negative_sample_text_list
        ]

        return misconception_encodings

    def _get_query_encoding(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        question_text = row[self.CSVColName.QUESTION_TEXT]
        subject_name = row[self.CSVColName.SUBJECT_NAME]
        construct_name = row[self.CSVColName.CONSTRUCT_NAME]
        answer_text = row[self.CSVColName.ANSWER_TEXT]

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
            max_length=self.question_max_length,
        )

        return query_encoding

    def collate_fn(
        self,
        batch: List[Dict[str, torch.Tensor]],
        include_meta_data: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """Collate function for the contrastive dataset with meta data.

        Args:
            batch (List[Dict[str, torch.Tensor]]): Batch of items.
            include_meta_data (Optional[bool]): Whether to include the meta data.
            Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the collated items.
        """

        output = {
            self.TorchColName.QUESTION_IDS: torch.stack(
                [item[self.TorchColName.QUESTION_IDS] for item in batch]
            ),
            self.TorchColName.QUESTION_MASK: torch.stack(
                [item[self.TorchColName.QUESTION_MASK] for item in batch]
            ),
            self.TorchColName.MISCONCEPTION_IDS: torch.stack(
                [item[self.TorchColName.MISCONCEPTION_IDS] for item in batch]
            ),
            self.TorchColName.MISCONCEPTION_MASK: torch.stack(
                [item[self.TorchColName.MISCONCEPTION_MASK] for item in batch]
            ),
            self.TorchColName.LABEL: torch.stack(
                [item[self.TorchColName.LABEL] for item in batch]
            ),
        }

        if include_meta_data:
            output.update(
                {
                    # self.TorchColName.META_DATA_QUESTION_ID: torch.stack(
                    #     [
                    #         item[self.TorchColName.META_DATA_QUESTION_ID]
                    #         for item in batch
                    #     ]
                    # ),
                    # self.TorchColName.META_DATA_SUBJECT_ID: torch.stack(
                    #     [item[self.TorchColName.META_DATA_SUBJECT_ID] for item in batch]
                    # ),
                    # self.TorchColName.META_DATA_CONSTRUCT_ID: torch.stack(
                    #     [
                    #         item[self.TorchColName.META_DATA_CONSTRUCT_ID]
                    #         for item in batch
                    #     ]
                    # ),
                    self.TorchColName.META_DATA_MISCONCEPTION_ID: torch.stack(
                        [
                            item[self.TorchColName.META_DATA_MISCONCEPTION_ID]
                            for item in batch
                        ]
                    ),
                }
            )

        return output
