import torch

from src.constants.column_names import (
    ContrastiveCSVColumns,
    ContrastiveTorchDatasetColumns,
)
from src.data_preparation.datasets.abstract_dataset import AbstractDataset


class BaseDataset(AbstractDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get label
        label = torch.tensor(row[ContrastiveCSVColumns.LABEL], dtype=torch.long)

        # Get question and answer
        question = row[ContrastiveCSVColumns.QUESTION_DETAILS]
        misconception_list = self.split_misconception_list(
            row[ContrastiveCSVColumns.MISCONCEPTION_LIST]
        )

        # Embed question and misconception list
        question_embedding = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.question_max_length,
        )
        misconception_encodings = [
            self.tokenizer(
                misc,
                max_length=self.misconception_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for misc in misconception_list
        ]

        output = {
            ContrastiveTorchDatasetColumns.QUESTION_IDS: question_embedding[
                "input_ids"
            ].squeeze(0),
            ContrastiveTorchDatasetColumns.QUESTION_MASK: question_embedding[
                "attention_mask"
            ].squeeze(0),
            ContrastiveTorchDatasetColumns.MISCONCEPTION_IDS: torch.stack(
                [m["input_ids"].squeeze(0) for m in misconception_encodings]
            ),
            ContrastiveTorchDatasetColumns.MISCONCEPTION_MASK: torch.stack(
                [m["attention_mask"].squeeze(0) for m in misconception_encodings]
            ),
            ContrastiveTorchDatasetColumns.LABEL: label,
        }

        if self.include_meta_data:
            output[ContrastiveTorchDatasetColumns.META_DATA_QUESTION_ID] = torch.tensor(
                row[ContrastiveCSVColumns.QUESTION_ID],
                dtype=torch.long,
            )
            output[ContrastiveTorchDatasetColumns.META_DATA_SUBJECT_ID] = torch.tensor(
                row[ContrastiveCSVColumns.SUBJECT_ID],
                dtype=torch.long,
            )
            output[ContrastiveTorchDatasetColumns.META_DATA_CONSTRUCT_ID] = (
                torch.tensor(
                    row[ContrastiveCSVColumns.CONSTRUCT_ID],
                    dtype=torch.long,
                )
            )
            output[ContrastiveTorchDatasetColumns.META_DATA_MISCONCEPTION_ID] = (
                torch.tensor(
                    row[ContrastiveCSVColumns.MISCONCEPTION_ID],
                    dtype=torch.long,
                )
            )

        return output
