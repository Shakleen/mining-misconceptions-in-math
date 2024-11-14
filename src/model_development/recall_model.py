import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from src.constants.column_names import ContrastiveTorchDatasetColumns
from src.evaluation.map_calculator.map_calculator import MAPCalculator
from src.model_development.loss_functions import info_nce_loss
from src.constants.dll_paths import DLLPaths
from src.configurations.recall_model_config import RecallModelConfig


class RecallModel(pl.LightningModule):
    @property
    def TorchColNames(self):
        return ContrastiveTorchDatasetColumns

    def __init__(self, config: RecallModelConfig):
        super().__init__()
        self.config = config

        self.map_calculator = MAPCalculator(DLLPaths.MAP_CALCULATOR)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model = prepare_model_for_kbit_training(self.model)

        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        self.vector_linear = torch.nn.Linear(
            self.model.config.vocab_size, config.output_dim
        )

    def get_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        features = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        features = F.normalize(self.vector_linear(features), p=2, dim=1)
        return features

    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        misconception_ids: torch.Tensor,
        misconception_mask: torch.Tensor,
    ):
        """Forward pass for the recall model.

        Args:
            question_ids (torch.Tensor): Question IDs. Shape: (batch_size, question_seq_len).
            question_mask (torch.Tensor): Question mask. Shape: (batch_size, question_seq_len).
            misconception_ids (torch.Tensor): Misconception IDs. Shape: (batch_size, num_of_misconceptions, misconception_seq_len).
            misconception_mask (torch.Tensor): Misconception mask. Shape: (batch_size, num_of_misconceptions, misconception_seq_len).

        Returns:
            torch.Tensor: Similarities between the question and misconceptions. Shape: (batch_size, num_of_misconceptions).
        """
        # Reshape misconception inputs to encode all at once
        batch_size, num_misconceptions, seq_length = misconception_ids.shape
        docs = self.get_features(
            input_ids=misconception_ids.view(-1, seq_length),
            attention_mask=misconception_mask.view(-1, seq_length),
        )

        # Reshape back to [batch_size, num_misconceptions, hidden_size]
        docs = docs.view(batch_size, num_misconceptions, -1)

        query = self.get_features(input_ids=question_ids, attention_mask=question_mask)
        # Expand query embeddings to match misconception shape
        # Shape: [batch_size, 1, hidden_size]
        query = query.unsqueeze(1)

        similarities = torch.sum(query * docs, dim=-1)
        return similarities

    def _log_metrics(self, loss: float, accuracy: float, map: float, phase: str):
        """Log metrics for the model.

        Throws an assertion error if the phase is not valid.

        Args:
            loss (float): Loss value.
            accuracy (float): Accuracy value.
            map (float): MAP value.
            phase (str): Phase of the model (train or val).
        """
        assert phase in [
            "train",
            "val",
        ], "Invalid phase. Must be either 'train' or 'val'."

        self.log(f"{phase}_loss_{self.config.fold}", loss)
        self.log(f"{phase}_accuracy_{self.config.fold}", accuracy)
        self.log(f"{phase}_map_{self.config.fold}", map)

        if phase == "train":
            self.log(
                f"learning_rate_{self.config.fold}",
                self.optimizers().param_groups[0]["lr"],
            )

    def training_step(self, batch, batch_idx):
        similarities = self(
            batch[self.TorchColNames.QUESTION_IDS],
            batch[self.TorchColNames.QUESTION_MASK],
            batch[self.TorchColNames.MISCONCEPTION_IDS],
            batch[self.TorchColNames.MISCONCEPTION_MASK],
        )

        # Cross entropy loss
        loss = info_nce_loss(similarities, batch[self.TorchColNames.LABEL])

        # Calculate accuracy
        predictions = torch.argmax(similarities, dim=1)
        accuracy = (predictions == batch[self.TorchColNames.LABEL]).float().mean()

        # Calculate MAP
        map = self.map_calculator.calculate_batch_map(
            actual_indices=batch[self.TorchColNames.LABEL].detach().cpu().numpy(),
            rankings_batch=predictions.detach().cpu().numpy(),
        )

        # Log metrics
        self._log_metrics(loss, accuracy, map, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        similarities = self(
            batch[self.TorchColNames.QUESTION_IDS],
            batch[self.TorchColNames.QUESTION_MASK],
            batch[self.TorchColNames.MISCONCEPTION_IDS],
            batch[self.TorchColNames.MISCONCEPTION_MASK],
        )

        # Cross entropy loss
        loss = info_nce_loss(similarities, batch[self.TorchColNames.LABEL])

        # Calculate accuracy
        predictions = torch.argmax(similarities, dim=1)
        accuracy = (predictions == batch[self.TorchColNames.LABEL]).float().mean()

        # Calculate MAP
        map = self.map_calculator.calculate_batch_map(
            actual_indices=batch[self.TorchColNames.LABEL].detach().cpu().numpy(),
            rankings_batch=predictions.detach().cpu().numpy(),
        )

        self._log_metrics(loss, accuracy, map, "val")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.config.learning_rate,
            max_lr=self.config.learning_rate * 10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
