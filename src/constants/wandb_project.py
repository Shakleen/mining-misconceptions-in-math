from enum import Enum


class WandbProject(Enum):
    """Enum for the names of the datasets and the project."""

    PROJECT_NAME = "Kaggle_EEDI"
    TRAIN_DATASET = "train-dataset"
    TEST_DATASET = "test-dataset"
    MISCONCEPTIONS_DATASET = "misconceptions-dataset"
