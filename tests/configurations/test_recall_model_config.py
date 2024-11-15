import pytest

from src.configurations.recall_model_config import RecallModelConfig


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_init(fold: int):
    config = RecallModelConfig("xyz", fold)
    assert isinstance(config, RecallModelConfig)


def test_init_from_json():
    config = RecallModelConfig.from_json("config/recall_model_config.json")
    assert isinstance(config, RecallModelConfig)


def test_exception_on_invalid_sentence_pooling_method():
    with pytest.raises(ValueError):
        config = RecallModelConfig(model_path="xyz", sentence_pooling_method="invalid")


def test_io_exception_on_invalid_json():
    with pytest.raises(FileNotFoundError):
        config = RecallModelConfig.from_json("invalid/path/to/config.json")
