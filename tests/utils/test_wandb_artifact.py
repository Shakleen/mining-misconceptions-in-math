import pytest
from mock import patch, Mock
import pandas as pd
import wandb

from src.utils.wandb_artifact import (
    load_dataframe_artifact,
    log_dataframe_artifact,
    _create_temp_csv,
    _generate_metadata,
)


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


def test_create_temp_csv_file():
    with patch(
        "src.utils.wandb_artifact.tempfile.mktemp", return_value="temp_file"
    ), patch("src.utils.wandb_artifact.pd.read_csv", return_value=Mock(pd.DataFrame)):
        df = _create_temp_csv(Mock(pd.DataFrame))
        assert df == "temp_file"


def test_generate_metadata():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, None, 6], "c": ["a1", "b2", "c3"]})
    metadata = _generate_metadata(df)
    expected_metadata = {
        "total_rows": 3,
        "total_columns": 3,
        "total_null_count": 1,
        "null_percentage": 11.11111111111111,
        "columns": {
            "a": {
                "dtype": "int64",
                "non_null_count": 3,
                "unique_count": 3,
                "null_count": 0,
                "null_percentage": 0,
            },
        },
    }

    assert metadata["total_rows"] == expected_metadata["total_rows"]
    assert metadata["total_columns"] == expected_metadata["total_columns"]
    assert metadata["total_null_count"] == expected_metadata["total_null_count"]
    assert metadata["null_percentage"] == expected_metadata["null_percentage"]


def test_log_dataframe_artifact():
    mock_artifact = Mock(wandb.Artifact)
    df = pd.DataFrame()

    with (
        patch("src.utils.wandb_artifact.wandb.Artifact", return_value=mock_artifact),
        patch("src.utils.wandb_artifact.wandb.log_artifact", return_value=None),
        patch("src.utils.wandb_artifact.wandb.log", return_value=None),
        patch("src.utils.wandb_artifact._create_temp_csv", return_value="temp_file"),
        patch("src.utils.wandb_artifact._generate_metadata", return_value={}),
    ):
        artifact = log_dataframe_artifact(df, "test", "test", "test")
        assert artifact is mock_artifact
