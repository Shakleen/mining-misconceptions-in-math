import pytest

from src.configurations.data_config import DataConfig


def test_init():
    config = DataConfig("xyz", "latest", 5, 16, 4)
    assert isinstance(config, DataConfig)


def test_init_from_json():
    config = DataConfig.from_json("config/data_config.json")
    assert isinstance(config, DataConfig)
