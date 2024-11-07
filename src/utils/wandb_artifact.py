import os
from typing import Optional
import pandas as pd
import wandb
import re


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
