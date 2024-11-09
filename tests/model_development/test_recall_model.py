import pytest
import torch

from src.model_development.recall_model import RecallModel


@pytest.fixture(scope="module")
def recall_model():
    model_path = "output_dir/stella_v0/model"
    fold = 0
    optimizer = None
    scheduler = None
    map_calculator = None
    return RecallModel(model_path, fold, optimizer, scheduler, map_calculator).eval()


def test_init(recall_model: RecallModel):
    assert isinstance(recall_model, RecallModel)


def test_module_attributes(recall_model: RecallModel):
    assert hasattr(recall_model, "model")
    assert hasattr(recall_model, "vector_linear")


@pytest.mark.parametrize(
    ("batch_size", "seq_len"),
    [(4, 10), (8, 20), (16, 30)],
)
def test_get_features(recall_model: RecallModel, batch_size: int, seq_len: int):
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    features = recall_model.get_features(input_ids, attention_mask)
    assert features.shape == (batch_size, 1024)


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
