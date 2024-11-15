from dataclasses import dataclass
import json


@dataclass
class DataConfig:
    """Configuration for the data used in the training pipeline."""
    contrastive_data_name: str
    contrastive_data_version: str
    num_folds: int
    batch_size: int
    num_workers: int

    @classmethod
    def from_json(cls, json_path: str) -> "DataConfig":
        with open(json_path, "r") as f:
            raw_config = json.load(f)

        config_dict = {key: data["value"] for key, data in raw_config.items()}

        return cls(**config_dict)
