import os
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
from sklearn.model_selection import StratifiedGroupKFold
import wandb

from src.constants.wandb_project import WandbProject
from src.configurations.recall_model_config import RecallModelConfig
from src.configurations.data_config import DataConfig
from src.configurations.trainer_config import TrainerConfig
from src.constants.column_names import QAPairCSVColumns
from src.utils.seed_everything import seed_everything
from src.model_development.recall_model import RecallModel
from src.utils.wandb_artifact import load_dataframe_artifact
from src.pipeline.inference_recall_model import inference
from src.data_preparation.datasets.base_dataset_v2 import BaseDatasetV2
from src.data_preparation.negative_sampler.random_sampler import RandomNegativeSampler


os.environ["TOKENIZERS_PARALLELISM"] = "true"
# torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the recall model.",
        usage="""
        python train_recall_model.py \
            --model_config config/recall_model_config.json \
            --data_config config/data_config.json \
            --trainer_config config/trainer_config.json
        """,
        prog="train_recall_model",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration JSON file.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data configuration JSON file.",
    )
    parser.add_argument(
        "--trainer_config",
        type=str,
        required=True,
        help="Path to the trainer configuration JSON file.",
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

    model_config = RecallModelConfig.from_json(args.model_config)
    data_config = DataConfig.from_json(args.data_config)
    trainer_config = TrainerConfig.from_json(args.trainer_config)

    wandb.init(
        project=WandbProject.PROJECT_NAME,
        job_type="train-recall-model",
        config={
            "model_config": model_config.to_dict(),
            "data_config": data_config.to_dict(),
            "trainer_config": trainer_config.to_dict(),
            "debug": args.debug,
            "seed": args.seed,
        },
        name=f"recall_model-{wandb.util.generate_id()}",
    )

    df = load_dataframe_artifact(
        WandbProject.QA_PAIR_DATASET_NAME,
        data_config.qa_pair_data_version,
    )
    misconception_df = load_dataframe_artifact(
        WandbProject.MISCONCEPTIONS_DATASET_NAME,
        data_config.misconception_data_version,
    )

    skf = StratifiedGroupKFold(
        n_splits=data_config.num_folds,
        shuffle=True,
        random_state=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    all_preds = []
    all_labels = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(
            df,
            df[QAPairCSVColumns.MISCONCEPTION_ID],
            groups=df[QAPairCSVColumns.QUESTION_ID],
        )
    ):
        train_loader, val_loader = get_data_loaders(
            data_config,
            df,
            misconception_df,
            tokenizer,
            train_idx,
            val_idx,
        )

        model_config.fold = fold
        best_model = train_model(
            model_config,
            trainer_config,
            fold,
            train_loader,
            val_loader,
            tokenizer,
        )

        preds, labels = inference(
            best_model,
            misconception_df,
            tokenizer,
            data_config.batch_size,
            data_config.num_workers,
            val_loader,
        )
        all_preds.extend(preds)
        all_labels.extend(labels)

        map_score = best_model.map_calculator.calculate_batch_map(
            actual_indices=np.concatenate(labels),
            rankings_batch=np.concatenate(preds),
        )

        wandb.log({f"map_score_{fold}": map_score})

        if args.debug:
            break

        del best_model, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    map_score = best_model.map_calculator.calculate_batch_map(
        actual_indices=np.concatenate(all_labels),
        rankings_batch=np.concatenate(all_preds),
    )
    wandb.log({"overall_map_score": map_score})


def get_data_loaders(
    data_config: DataConfig,
    df: pd.DataFrame,
    misconception_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    train_idx: List[int],
    val_idx: List[int],
) -> Tuple[DataLoader, DataLoader]:
    """Get the training and validation data loaders.

    Args:
        data_config (DataConfig): Data configuration.
        df (pd.DataFrame): DataFrame containing the dataset.
        misconception_df (pd.DataFrame): DataFrame containing the misconceptions dataset.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
        train_idx (List[int]): Indices of the training data.
        val_idx (List[int]): Indices of the validation data.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    sampler = RandomNegativeSampler(
        sample_size=data_config.negative_sample_size,
        total_misconceptions=misconception_df.shape[0],
    )

    train_dataset = BaseDatasetV2(
        dataframe=train_df,
        misconceptions_df=misconception_df,
        tokenizer=tokenizer,
        negative_sampler=sampler,
        include_meta_data=False,
        question_max_length=data_config.question_max_length,
        misconception_max_length=data_config.misconception_max_length,
    )
    val_dataset = BaseDatasetV2(
        dataframe=val_df,
        misconceptions_df=misconception_df,
        tokenizer=tokenizer,
        negative_sampler=sampler,
        include_meta_data=False,
        question_max_length=data_config.question_max_length,
        misconception_max_length=data_config.misconception_max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=True,
        collate_fn=lambda x: train_dataset.collate_fn(x, False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=False,
        collate_fn=lambda x: val_dataset.collate_fn(x, False),
    )

    return train_loader, val_loader


def train_model(
    model_config: RecallModelConfig,
    trainer_config: TrainerConfig,
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
) -> RecallModel:
    """Train the recall model for a given fold.

    Args:
        model_config (RecallModelConfig): Model configuration.
        trainer_config (TrainerConfig): Trainer configuration.
        fold (int): Fold number.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
    Returns:
        RecallModel: Best model.
    """
    model = RecallModel(model_config, tokenizer)

    checkpoint_callback = ModelCheckpoint(
        monitor=f"val_loss_{fold}",
        mode="min",
        save_top_k=1,
        filename=f"best-checkpoint-{fold}",
    )
    early_stopping_callback = EarlyStopping(
        monitor=f"val_loss_{fold}",
        mode="min",
        patience=trainer_config.patience,
    )
    wandb_logger = WandbLogger(
        project=WandbProject.PROJECT_NAME,
        job_type="train-recall-model",
    )
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=trainer_config.num_epochs,
        log_every_n_steps=trainer_config.logging_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
        val_check_interval=trainer_config.val_check_interval,
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    best_model_path = checkpoint_callback.best_model_path
    # best_model_path = "lightning_logs/version_1/checkpoints/best-checkpoint-0.ckpt"
    best_model = RecallModel.load_from_checkpoint(
        best_model_path,
        config=model_config,
        tokenizer=tokenizer,
    )
    return best_model.eval()


if __name__ == "__main__":
    args = parse_args()
    main(args)
