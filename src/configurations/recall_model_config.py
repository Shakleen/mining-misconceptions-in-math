from dataclasses import asdict, dataclass
import json


@dataclass
class RecallModelConfig:
    model_path: str
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    gradient_checkpointing: bool = True
    hidden_dim: int = 512
    num_latents: int = 512
    num_heads: int = 8
    mlp_ratio: int = 4
    output_dim: int = 1024
    sentence_pooling_method: str = "cls"
    dropout_p: float = 0.1

    @classmethod
    def from_json(cls, json_path: str) -> "RecallModelConfig":
        """
        Create a RecallModelConfig instance from a JSON file.

        Args:
            json_path: Path to the JSON configuration file
            Each parameter in the JSON should have 'value' and 'description' keys.
            Only 'value' will be used for configuration.

        Returns:
            RecallModelConfig instance
        """
        with open(json_path, "r") as f:
            raw_config = json.load(f)

        # Extract only the 'value' from each parameter
        config_dict = {key: data["value"] for key, data in raw_config.items()}

        return cls(**config_dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        allowed_sentence_pooling_methods = {"mean", "cls", "last", "attention"}

        if self.sentence_pooling_method not in allowed_sentence_pooling_methods:
            raise ValueError(
                f"Invalid sentence pooling method '{self.sentence_pooling_method}'. "
                + f"Must be one of {allowed_sentence_pooling_methods}."
            )
