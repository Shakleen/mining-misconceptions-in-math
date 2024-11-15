import pytest
import torch

from src.model_development.recall_model import RecallModel
from src.configurations.recall_model_config import RecallModelConfig


@pytest.fixture(scope="module")
def config():
    return RecallModelConfig(".cache/Mistral-7B-v0.1", 0)


@pytest.fixture(scope="module")
def recall_model(config: RecallModelConfig):
    return RecallModel(config).eval()


def test_init(recall_model: RecallModel):
    assert isinstance(recall_model, RecallModel)


def test_module_attributes(recall_model: RecallModel):
    assert hasattr(recall_model, "model")


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "hidden_size"),
    [(4, 10, 128), (8, 20, 256), (16, 30, 512)],
)
def test_last_token_pool(
    recall_model: RecallModel,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
):
    last_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones((batch_size, seq_len))
    last_token_pool = recall_model.last_token_pool(last_hidden_states, attention_mask)
    assert last_token_pool.shape == (batch_size, hidden_size)


@pytest.mark.parametrize(
    "pooling_method",
    ["last", "cls", "mean"],
)
def test_pool_sentence_embedding(recall_model: RecallModel, pooling_method: str):
    batch_size = 1
    seq_len = 2
    hidden_size = recall_model.model.config.hidden_size

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones((batch_size, seq_len))

    pooled_embedding = recall_model.pool_sentence_embedding(
        pooling_method=pooling_method,
        hidden_state=hidden_state,
        mask=mask,
    )
    assert pooled_embedding.shape == (batch_size, hidden_size)


@pytest.mark.parametrize(
    ("batch_size", "seq_len"),
    [(4, 10), (8, 20), (16, 30)],
)
def test_get_features(recall_model: RecallModel, batch_size: int, seq_len: int):
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    features = recall_model.get_features(input_ids, attention_mask)
    assert features.shape == (batch_size, recall_model.model.config.vocab_size)


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_of_misconceptions"),
    [(4, 10, 1), (8, 10, 5)],
)
def test_forward(
    recall_model: RecallModel,
    batch_size: int,
    seq_len: int,
    num_of_misconceptions: int,
):
    question_ids = torch.randint(0, 10000, (batch_size, seq_len))
    question_mask = torch.ones_like(question_ids)
    misconceptions_ids = torch.randint(
        0, 10000, (batch_size, num_of_misconceptions, seq_len)
    )
    misconceptions_mask = torch.ones_like(misconceptions_ids)
    similarities = recall_model(
        question_ids,
        question_mask,
        misconceptions_ids,
        misconceptions_mask,
    )
    assert similarities.shape == (batch_size, num_of_misconceptions)
