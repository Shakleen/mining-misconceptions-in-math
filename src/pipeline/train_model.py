import os
import json
import argparse
import gc
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import wandb

from src.constants.wandb_project import WandbProject
from src.configurations.recall_model_config import RecallModelConfig
from src.configurations.data_config import DataConfig
from src.configurations.trainer_config import TrainerConfig
from src.constants.column_names import QAPairCSVColumns
from src.utils.seed_everything import seed_everything
from src.model_development.stella_model import StellaModel
from src.utils.wandb_artifact import load_dataframe_artifact
from src.data_preparation.datasets.base_dataset_v2 import BaseDatasetV2
from src.data_preparation.negative_sampler.hard_negative_sampler_v2 import (
    HardNegativeSamplerV2,
)
from src.pipeline.embbed_misconceptions import create_misconception_dataloader


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_HOME"] = ".cache"
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the recall model.",
        usage="python train_recall_model.py",
        prog="train_model",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Whether to run in debug mode. Defaults to False.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Defaults to 42.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    seed_everything(args.seed)

    model_config = RecallModelConfig.from_json("config/recall_model_config.json")
    data_config = DataConfig.from_json("config/data_config.json")
    trainer_config = TrainerConfig.from_json("config/trainer_config.json")

    run_name = f"ST-{wandb.util.generate_id()}"

    wandb.init(
        project=WandbProject.PROJECT_NAME,
        job_type="train-two-tower-model",
        config={
            "model_config": model_config.to_dict(),
            "data_config": data_config.to_dict(),
            "trainer_config": trainer_config.to_dict(),
            "debug": args.debug,
            "seed": args.seed,
        },
        name=run_name,
    )

    df = load_dataframe_artifact(
        WandbProject.QA_PAIR_DATASET_NAME,
        data_config.qa_pair_data_version,
    )
    misconception_df = load_dataframe_artifact(
        WandbProject.MISCONCEPTIONS_DATASET_NAME,
        data_config.misconception_data_version,
    )

    if args.debug:
        df = df.sample(frac=0.1).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    misconception_dataloader = create_misconception_dataloader(
        misconception_df,
        tokenizer,
        data_config.batch_size,
        data_config.num_workers,
    )

    train_loader, val_loader = get_data_loaders(
        data_config,
        df,
        misconception_df,
        tokenizer,
    )

    train_model(
        model_config,
        trainer_config,
        train_loader,
        val_loader,
        misconception_dataloader,
        run_name,
    )

    wandb.finish()


def get_data_loaders(
    data_config: DataConfig,
    df: pd.DataFrame,
    misconception_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """Get the training and validation data loaders.

    Args:
        data_config (DataConfig): Data configuration.
        df (pd.DataFrame): DataFrame containing the dataset.
        misconception_df (pd.DataFrame): DataFrame containing the misconceptions dataset.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    train_df = df.loc[df[QAPairCSVColumns.SPLIT] == "train"].reset_index(drop=True)
    val_df = df.loc[df[QAPairCSVColumns.SPLIT] == "test"].reset_index(drop=True)

    misconception_embeddings = np.load(data_config.misconception_embeddings_path)

    train_loader = get_loader(
        train_df,
        misconception_df,
        tokenizer,
        data_config,
        misconception_embeddings,
        False,
        data_config.super_set_size_multiplier,
    )

    val_loader = get_loader(
        val_df,
        misconception_df,
        tokenizer,
        data_config,
        misconception_embeddings,
        True,
        1,
    )

    return train_loader, val_loader


def get_loader(
    df: pd.DataFrame,
    misconception_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    data_config: DataConfig,
    misconception_embeddings: np.ndarray,
    is_validation: bool,
    super_set_size_multiplier: int,
):
    """Get the data loader for the given dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        misconception_df (pd.DataFrame): DataFrame containing the misconceptions dataset.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
        data_config (DataConfig): Data configuration.
        misconception_embeddings (np.ndarray): Embeddings of all misconceptions.
        is_validation (bool): Whether the loader is for validation or training.
        super_set_size_multiplier (int): Multiplier for the super set size.

    Returns:
        DataLoader: Data loader for the given dataframe.
    """
    sampler = HardNegativeSamplerV2(
        sample_size=data_config.negative_sample_size,
        misconception_embeddings=misconception_embeddings,
        is_validation=is_validation,
        super_set_size_multiplier=super_set_size_multiplier,
    )
    dataset = BaseDatasetV2(
        dataframe=df,
        misconceptions_df=misconception_df,
        tokenizer=tokenizer,
        negative_sampler=sampler,
        include_meta_data=is_validation,
        question_max_length=data_config.question_max_length,
        misconception_max_length=data_config.misconception_max_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=not is_validation,
        collate_fn=lambda x: dataset.collate_fn(x, is_validation),
    )
    return data_loader


def train_model(
    model_config: RecallModelConfig,
    trainer_config: TrainerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    misconception_dataloader: DataLoader,
    run_name: str,
) -> StellaModel:
    """Train the recall model.

    Args:
        model_config (RecallModelConfig): Model configuration.
        trainer_config (TrainerConfig): Trainer configuration.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        misconception_dataloader (DataLoader): Misconception data loader.
        run_name (str): Name of the run.
    """
    save_dir = f"output_dir/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    json.dump(
        model_config.to_dict(),
        open(os.path.join(save_dir, "model_config.json"), "w"),
    )

    model = StellaModel(model_config)
    model.set_misconception_dataloader(misconception_dataloader)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"best-checkpoint",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=trainer_config.patience,
    )
    wandb_logger = WandbLogger(
        project=WandbProject.PROJECT_NAME,
        job_type="train-two-tower-model",
        dir="output_dir",
        save_dir=save_dir,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        precision="bf16-mixed",
        max_epochs=trainer_config.num_epochs,
        log_every_n_steps=trainer_config.logging_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
        val_check_interval=trainer_config.val_check_interval,
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
