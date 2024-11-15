import argparse

from src.configurations.recall_model_config import RecallModelConfig
from src.configurations.data_config import DataConfig
from src.configurations.trainer_config import TrainerConfig


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
    return parser.parse_args()


def main(args: argparse.Namespace):
    model_config = RecallModelConfig.from_json(args.model_config)
    data_config = DataConfig.from_json(args.data_config)
    trainer_config = TrainerConfig.from_json(args.trainer_config)

    print(model_config)
    print(data_config)
    print(trainer_config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
