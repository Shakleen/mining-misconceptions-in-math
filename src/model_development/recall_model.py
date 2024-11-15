import torch
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from src.constants.column_names import ContrastiveTorchDatasetColumns
from src.evaluation.map_calculator.map_calculator import MAPCalculator
from src.model_development.loss_functions import info_nce_loss
from src.constants.dll_paths import DLLPaths
from src.configurations.recall_model_config import RecallModelConfig
from src.model_development.latent_attention_module import LatentAttentionLayer


class RecallModel(pl.LightningModule):
    @property
    def TorchColNames(self):
        return ContrastiveTorchDatasetColumns

    def __init__(self, config: RecallModelConfig, tokenizer: AutoTokenizer = None):
        super().__init__()
        self.config = config

        self.map_calculator = MAPCalculator(DLLPaths.MAP_CALCULATOR)

        if config.use_lora:
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
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                device_map="auto",
                trust_remote_code=True,
            )

        if tokenizer is not None:
            self.model.resize_token_embeddings(len(tokenizer))

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if config.sentence_pooling_method == "attention":
            self.latent_attention_layer = LatentAttentionLayer(
                input_dim=self.model.config.vocab_size,
                hidden_dim=config.hidden_dim,
                num_latents=config.num_latents,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
            )

    def last_token_pool(
        self,
        last_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Pool the last token of the sequence.

        Args:
            last_hidden_states (Tensor): Last hidden states of the model. Shape: (batch_size, seq_length, hidden_size).
            attention_mask (Tensor): Attention mask. Shape: (batch_size, seq_length).

        Returns:
            Tensor: Pooled embeddings. Shape: (batch_size, hidden_size).
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def pool_sentence_embedding(
        self,
        pooling_method: str,
        hidden_state: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Pool the sentence embedding based on the sentence pooling method.

        Args:
            pooling_method (str): Sentence pooling method.
            hidden_state (Tensor): Hidden states of the model. Shape: (batch_size, seq_length, hidden_size).
            mask (Tensor): Attention mask. Shape: (batch_size, seq_length).

        Returns:
            Tensor: Pooled embeddings. Shape: (batch_size, hidden_size).
        """
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]
        elif pooling_method == "last":
            return self.last_token_pool(hidden_state, mask)
        elif pooling_method == "attention":
            return self.latent_attention_layer(hidden_state)

    def get_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Get the features from the model.

        Args:
            input_ids (Tensor): Input IDs. Shape: (batch_size, seq_length).
            attention_mask (Tensor): Attention mask. Shape: (batch_size, seq_length).

        Returns:
            Tensor: Features. Shape: (batch_size, hidden_size).
        """
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        features = self.pool_sentence_embedding(
            pooling_method=self.config.sentence_pooling_method,
            hidden_state=last_hidden_state,
            mask=attention_mask,
        )
        return F.normalize(features, p=2, dim=1)

    def forward(
        self,
        question_ids: Tensor,
        question_mask: Tensor,
        misconception_ids: Tensor,
        misconception_mask: Tensor,
    ):
        """Forward pass for the recall model.

        Args:
            question_ids (Tensor): Question IDs. Shape: (batch_size, question_seq_len).
            question_mask (Tensor): Question mask. Shape: (batch_size, question_seq_len).
            misconception_ids (Tensor): Misconception IDs. Shape: (batch_size, num_of_misconceptions, misconception_seq_len).
            misconception_mask (Tensor): Misconception mask. Shape: (batch_size, num_of_misconceptions, misconception_seq_len).

        Returns:
            Tensor: Similarities between the question and misconceptions. Shape: (batch_size, num_of_misconceptions).
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
        rankings = torch.argsort(similarities, dim=-1, descending=True)
        map = self.map_calculator.calculate_batch_map(
            actual_indices=batch[self.TorchColNames.LABEL].detach().cpu().numpy(),
            rankings_batch=rankings.detach().cpu().numpy(),
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
        rankings = torch.argsort(similarities, dim=-1, descending=True)
        map = self.map_calculator.calculate_batch_map(
            actual_indices=batch[self.TorchColNames.LABEL].detach().cpu().numpy(),
            rankings_batch=rankings.detach().cpu().numpy(),
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
