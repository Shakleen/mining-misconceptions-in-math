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
from src.constants.column_names import ContrastiveCSVColumns
from src.utils.seed_everything import seed_everything
from src.data_preparation.get_dataloader import get_dataloader
from src.model_development.recall_model import RecallModel
from src.utils.wandb_artifact import load_dataframe_artifact
from src.pipeline.inference_recall_model import inference


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("medium")


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

    if args.debug:
        df = pd.read_csv("data/contrastive-datasethu66w3xp.csv")
        misconception_df = pd.read_csv("data/misconceptions-datasetas216_mx.csv")
    else:
        df = load_dataframe_artifact(
            data_config.contrastive_data_name,
            data_config.contrastive_data_version,
        )
        misconception_df = load_dataframe_artifact(
            data_config.misconception_data_name,
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
            df[ContrastiveCSVColumns.LABEL],
            df[ContrastiveCSVColumns.QUESTION_ID],
        )
    ):
        train_loader, val_loader = get_data_loaders(
            data_config,
            df,
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

        print(f"Fold {fold} MAP score: {map_score}")

        if args.debug:
            break

        del best_model, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    map_score = best_model.map_calculator.calculate_batch_map(
        actual_indices=np.concatenate(all_labels),
        rankings_batch=np.concatenate(all_preds),
    )
    print(f"Overall MAP score: {map_score}")


def get_data_loaders(
    data_config: DataConfig,
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    train_idx: List[int],
    val_idx: List[int],
) -> Tuple[DataLoader, DataLoader]:
    """Get the training and validation data loaders.

    Args:
        data_config (DataConfig): Data configuration.
        df (pd.DataFrame): DataFrame containing the dataset.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
        train_idx (List[int]): Indices of the training data.
        val_idx (List[int]): Indices of the validation data.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = get_dataloader(
        train_df,
        tokenizer,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=True,
        include_meta_data=False,
    )
    val_loader = get_dataloader(
        val_df,
        tokenizer,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=False,
        include_meta_data=True,
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
