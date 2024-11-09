from typing import Optional, List, Dict
import torch

from src.constants.column_names import (
    ContrastiveCSVColumns,
    ContrastiveTorchDatasetColumns,
)
from src.data_preparation.datasets.abstract_dataset import AbstractDataset


class BaseDataset(AbstractDataset):
    @property
    def CSVColName(self):
        """Column names for the CSV file."""
        return ContrastiveCSVColumns

    @property
    def TorchColName(self):
        """Column names for the torch dataset."""
        return ContrastiveTorchDatasetColumns

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get label
        label = torch.tensor(row[self.CSVColName.LABEL], dtype=torch.long)

        # Get question and answer
        question = row[self.CSVColName.QUESTION_DETAILS]
        misconception_list = self.split_misconception_list(
            row[self.CSVColName.MISCONCEPTION_LIST]
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
            self.TorchColName.QUESTION_IDS: question_embedding["input_ids"].squeeze(0),
            self.TorchColName.QUESTION_MASK: question_embedding[
                "attention_mask"
            ].squeeze(0),
            self.TorchColName.MISCONCEPTION_IDS: torch.stack(
                [m["input_ids"].squeeze(0) for m in misconception_encodings]
            ),
            self.TorchColName.MISCONCEPTION_MASK: torch.stack(
                [m["attention_mask"].squeeze(0) for m in misconception_encodings]
            ),
            self.TorchColName.LABEL: label,
        }

        if self.include_meta_data:
            output[self.TorchColName.META_DATA_QUESTION_ID] = torch.tensor(
                row[self.CSVColName.QUESTION_ID],
                dtype=torch.long,
            )
            output[self.TorchColName.META_DATA_SUBJECT_ID] = torch.tensor(
                row[self.CSVColName.SUBJECT_ID],
                dtype=torch.long,
            )
            output[self.TorchColName.META_DATA_CONSTRUCT_ID] = torch.tensor(
                row[self.CSVColName.CONSTRUCT_ID],
                dtype=torch.long,
            )
            output[self.TorchColName.META_DATA_MISCONCEPTION_ID] = torch.tensor(
                row[self.CSVColName.MISCONCEPTION_ID],
                dtype=torch.long,
            )

        return output

    def collate_fn(
        self,
        batch: List[Dict[str, torch.Tensor]],
        include_meta_data: Optional[bool] = False,
    ):
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
                    self.TorchColName.META_DATA_QUESTION_ID: torch.stack(
                        [
                            item[self.TorchColName.META_DATA_QUESTION_ID]
                            for item in batch
                        ]
                    ),
                    self.TorchColName.META_DATA_SUBJECT_ID: torch.stack(
                        [item[self.TorchColName.META_DATA_SUBJECT_ID] for item in batch]
                    ),
                    self.TorchColName.META_DATA_CONSTRUCT_ID: torch.stack(
                        [
                            item[self.TorchColName.META_DATA_CONSTRUCT_ID]
                            for item in batch
                        ]
                    ),
                    self.TorchColName.META_DATA_MISCONCEPTION_ID: torch.stack(
                        [
                            item[self.TorchColName.META_DATA_MISCONCEPTION_ID]
                            for item in batch
                        ]
                    ),
                }
            )

        return output
