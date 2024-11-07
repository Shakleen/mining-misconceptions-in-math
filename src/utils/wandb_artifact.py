import os
from typing import Optional
import re
import tempfile
import wandb
import pandas as pd


def load_dataframe_artifact(
    artifact_name: str,
    version: Optional[str] = "latest",
) -> pd.DataFrame:
    """Load a CSV artifact from W&B

    Args:
        artifact_name (str): Name of the artifact
        version (Optional[str]): Version of the artifact. Must be either "latest" or
        "v" followed by a number (e.g., "v0", "v1"). Defaults to "latest".

    Raises:
        ValueError: If no CSV file is found in the artifact
        ValueError: If version string format is invalid

    Returns:
        pd.DataFrame: DataFrame loaded from the artifact
    """
    # Validate version string format
    if not (version == "latest" or re.match(r"^v\d+$", version)):
        raise ValueError(
            'Version must be either "latest" or "v" followed by a non-negative integer (e.g., "v0", "v1")'
        )

    artifact = wandb.use_artifact(f"{artifact_name}:{version}")

    # Download artifact contents
    artifact_dir = artifact.download()

    # Find the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(artifact_dir) if f.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No CSV file found for artifact: {artifact_name}")

    # Load the CSV file into a DataFrame
    df_path = os.path.join(artifact_dir, csv_files[0])
    df = pd.read_csv(df_path)

    return df


def log_dataframe_artifact(
    df: pd.DataFrame,
    artifact_name: str,
    artifact_type: str,
    description: str,
) -> wandb.Artifact:
    """
    Create and log a Weights & Biases artifact for a single DataFrame `df`.

    Args:
        df (pd.DataFrame): DataFrame to be logged
        artifact_name (str): Name of the artifact
        artifact_type (str): Type of the artifact
        description (str): Description of the artifact.

    Returns:
        wandb.Artifact: Logged artifact
    """
    # Create an artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
    )

    # Create temporary CSV file
    temp_csv = _create_temp_csv(df, artifact_name)

    # Add the CSV file to the artifact
    artifact.add_file(temp_csv)

    metadata = _generate_metadata(df)
    artifact.metadata.update(metadata)

    # Log the artifact
    wandb.log_artifact(artifact)

    # Log dataset overview statistics
    wandb.log(
        {
            f"{artifact_name}_size": len(df),
            f"{artifact_name}_columns": len(df.columns),
            f"{artifact_name}_null_count": df.isnull().sum().sum(),
        }
    )

    return artifact


def _generate_metadata(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive metadata for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze

    Returns:
        dict: Comprehensive metadata about the DataFrame
    """
    # Column details
    column_metadata = {}
    for col in df.columns:
        column_metadata[col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": df[col].count(),
            "unique_count": df[col].nunique(),
            "null_count": df[col].isnull().sum(),
            "null_percentage": df[col].isnull().mean() * 100,
        }

    # Overall DataFrame metadata
    metadata = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_null_count": df.isnull().sum().sum(),
        "null_percentage": df.isnull().mean().mean() * 100,
        "columns": column_metadata,
    }

    return metadata


def _create_temp_csv(df: pd.DataFrame, prefix: str = "dataset") -> str:
    """
    Create a temporary CSV file for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to be saved as CSV
        prefix (str, optional): Prefix for the temporary filename. Defaults to "dataset".

    Returns:
        str: Path to the temporary CSV file
    """
    # Create a temporary file
    temp_file = tempfile.mktemp(suffix=".csv", prefix=prefix)

    # Save DataFrame to the temporary CSV
    df.to_csv(temp_file, index=False)

    return temp_file


if __name__ == "__main__":
    from src.constants.wandb_project import WandbProject
    from src.constants.paths import Paths

    train_df = pd.read_csv(Paths.TRAIN_COMPETITION_DATA)
    test_df = pd.read_csv(Paths.TEST_COMPETITION_DATA)
    misconception_mapping_df = pd.read_csv(Paths.MISCONCEPTIONS_MAPPING_DATA)

    wandb.init(project=WandbProject.PROJECT_NAME, name="original-datasets")

    try:
        # Log train dataset
        log_dataframe_artifact(
            train_df,
            WandbProject.TRAIN_DATASET_NAME,
            "dataset",
            "Train dataset for the Kaggle competition",
        )

        # Log test dataset
        log_dataframe_artifact(
            test_df,
            WandbProject.TEST_DATASET_NAME,
            "dataset",
            "Test dataset for the Kaggle competition",
        )

        # Log misconceptions dataset
        log_dataframe_artifact(
            misconception_mapping_df,
            WandbProject.MISCONCEPTIONS_DATASET_NAME,
            "dataset",
            "Misconceptions dataset for the Kaggle competition",
        )

    except Exception as e:
        print(f"Error logging artifacts: {e}")

    finally:
        # Close the wandb run
        wandb.finish()
