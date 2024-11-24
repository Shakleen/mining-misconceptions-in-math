from dataclasses import asdict, dataclass
import json


@dataclass
class DataConfig:
    """Configuration for the data used in the training pipeline."""
    qa_pair_data_version: str
    misconception_data_version: str
    num_folds: int
    batch_size: int
    num_workers: int
    negative_sample_size: int
    question_max_length: int
    misconception_max_length: int
    hard_to_random_ratio: float
    misconception_embeddings_path: str

    @classmethod
    def from_json(cls, json_path: str) -> "DataConfig":
        with open(json_path, "r") as f:
            raw_config = json.load(f)

        config_dict = {key: data["value"] for key, data in raw_config.items()}

        return cls(**config_dict)

    def to_dict(self) -> dict:
        return asdict(self)
