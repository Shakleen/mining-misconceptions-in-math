import os


class Paths:
    DATA = "data"
    TRAIN_COMPETITION_DATA = os.path.join(DATA, "competition", "train.csv")
    TEST_COMPETITION_DATA = os.path.join(DATA, "competition", "test.csv")
    MISCONCEPTIONS_MAPPING_DATA = os.path.join(
        DATA, "competition", "misconception_mapping.csv"
    )

    ARTIFACTS = "artifacts"
    WANDB = "wandb"
    OUTPUT_DIR = "output_dir"
    DOCS = "docs"
