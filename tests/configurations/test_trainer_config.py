import pytest

from src.configurations.trainer_config import TrainerConfig


def test_init():
    config = TrainerConfig("output_dir", "checkpoints", 5, 2, 20)
    assert isinstance(config, TrainerConfig)


def test_init_from_json():
    config = TrainerConfig.from_json("config/trainer_config.json")
    assert isinstance(config, TrainerConfig)
