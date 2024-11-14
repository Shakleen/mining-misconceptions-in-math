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
    output_dim: int = 1024
