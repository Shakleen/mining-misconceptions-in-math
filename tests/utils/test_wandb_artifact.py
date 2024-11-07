import pytest
from mock import patch, Mock
import pandas as pd
from src.utils.wandb_artifact import load_dataframe_artifact


@pytest.mark.parametrize("version", ["invalid_version", "v-1", "latst"])
def test_load_dataframe_artifact_throws_value_error_for_invalid_version(version: str):
    with pytest.raises(ValueError):
        load_dataframe_artifact("test", version)


@pytest.mark.parametrize("artifact_name", ["test", "test_with_csv", "test_without_csv"])
def test_load_dataframe_artifact_raises_value_error_if_no_csv_file_is_found(
    artifact_name: str,
):
    with (
        patch(
            "src.utils.wandb_artifact.os.listdir",
            return_value=[],
        ),
        patch(
            "src.utils.wandb_artifact.wandb.use_artifact",
            return_value=Mock(download=lambda: "test_dir"),
        ),
        pytest.raises(
            ValueError,
            match=f"No CSV file found for artifact: {artifact_name}",
        ),
    ):
        load_dataframe_artifact(artifact_name, "latest")


def test_load_dataframe_artifact_with_csv_file():
    mock_dataframe = Mock(pd.DataFrame)
    with (
        patch(
            "src.utils.wandb_artifact.os.listdir",
            return_value=["test.csv"],
        ),
        patch(
            "src.utils.wandb_artifact.wandb.use_artifact",
            return_value=Mock(download=lambda: "test_dir"),
        ),
        patch(
            "src.utils.wandb_artifact.pd.read_csv",
            return_value=mock_dataframe,
        ),
    ):
        df = load_dataframe_artifact("test_name", "latest")
        assert df is mock_dataframe
