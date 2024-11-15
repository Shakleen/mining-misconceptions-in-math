from dataclasses import dataclass
import json

@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    output_dir: str
    checkpoint_dir: str
    num_epochs: int
    patience: int
    logging_steps: int

    @classmethod
    def from_json(cls, json_path: str) -> "TrainerConfig":
        with open(json_path, "r") as f:
            raw_config = json.load(f)

        config_dict = {key: data["value"] for key, data in raw_config.items()}

        return cls(**config_dict)
