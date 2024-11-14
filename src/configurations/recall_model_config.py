from typing import List
from dataclasses import dataclass


@dataclass
class RecallModelConfig:
    model_path: str
    fold: int
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    gradient_checkpointing: bool = True
    hidden_dim: int = 512
    num_latents: int = 512
    num_heads: int = 8
    mlp_dim: int = 1024
    output_dim: int = 1024
    sentence_pooling_method: str = "cls"

    def __post_init__(self):
        allowed_sentence_pooling_methods = {"mean", "cls", "last", "attention"}

        if self.sentence_pooling_method not in allowed_sentence_pooling_methods:
            raise ValueError(
                f"Invalid sentence pooling method '{self.sentence_pooling_method}'. "
                + f"Must be one of {allowed_sentence_pooling_methods}."
            )
