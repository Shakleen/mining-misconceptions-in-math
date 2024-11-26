import torch
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

from src.constants.column_names import ContrastiveTorchDatasetColumns
from src.evaluation.map_calculator.map_calculator import MAPCalculator
from src.model_development.loss_functions import info_nce_loss
from src.constants.dll_paths import DLLPaths
from src.configurations.recall_model_config import RecallModelConfig
from src.model_development.latent_attention.latent_multi_head_attention import (
    LatentMultiHeadAttention,
)
from src.utils.searcher.similarity_searcher import SimilaritySearcher


class RecallModel(pl.LightningModule):
    @property
    def TorchColNames(self):
        return ContrastiveTorchDatasetColumns

    def __init__(self, config: RecallModelConfig, tokenizer: AutoTokenizer = None):
        super().__init__()
        self.config = config

        self.map_calculator = MAPCalculator(DLLPaths.MAP_CALCULATOR)
        self.searcher = SimilaritySearcher(DLLPaths.SIMILARITY_SEARCH)

        self.model = self.get_model(config, tokenizer)

        if config.sentence_pooling_method == "attention":
            self.latent_attention_layer = LatentMultiHeadAttention(
                input_dim=self.model.config.hidden_size,
                hidden_dim=config.hidden_dim,
                num_latents=config.num_latents,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )

    def get_model(self, config, tokenizer):
        if config.use_lora:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(self.model)

            model = get_peft_model(
                model,
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
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        else:
            model = AutoModel.from_pretrained(
                config.model_path,
                trust_remote_code=True,
            )

        if tokenizer is not None:
            model.resize_token_embeddings(len(tokenizer))

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model

    def set_misconception_dataloader(self, misconception_dataloader: DataLoader):
        """Set the misconception dataloader.

        This must be called before doing validation during training process.

        Args:
            misconception_df (pd.DataFrame): Misconception dataframe.
        """
        self.misconception_dataloader = misconception_dataloader

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

    def _get_features(
        self,
        model: AutoModel,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Get the features from the model.

        Args:
            model (AutoModel): Model to get features from.
            input_ids (Tensor): Input IDs. Shape: (batch_size, seq_length).
            attention_mask (Tensor): Attention mask. Shape: (batch_size, seq_length).

        Returns:
            Tensor: Features. Shape: (batch_size, hidden_size).
        """
        # Shape: (batch_size, seq_length, hidden_size)
        last_hidden_state = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        features = self.pool_sentence_embedding(
            pooling_method=self.config.sentence_pooling_method,
            hidden_state=last_hidden_state,
            mask=attention_mask,
        )
        return F.normalize(features, p=2, dim=1)

    def get_query_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self._get_features(input_ids, attention_mask, model=self.model)

    def get_docs_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self._get_features(input_ids, attention_mask, model=self.model)

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
        docs = self.get_docs_features(
            input_ids=misconception_ids.view(-1, seq_length),
            attention_mask=misconception_mask.view(-1, seq_length),
        )

        # Reshape back to [batch_size, num_misconceptions, hidden_size]
        docs = docs.view(batch_size, num_misconceptions, -1)

        query = self.get_query_features(
            input_ids=question_ids,
            attention_mask=question_mask,
        )
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

    def on_validation_epoch_start(self) -> None:
        self.misconception_embeddings = torch.cat(
            [
                self.get_docs_features(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                .detach()
                .cpu()
                for batch in self.misconception_dataloader
            ],
            dim=0,
        )
        self.map_score_list = []
        return super().on_validation_epoch_start()

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

        # Calculate MAP on negative samples
        rankings = torch.argsort(similarities, dim=-1, descending=True)
        map = self.map_calculator.calculate_batch_map(
            actual_indices=batch[self.TorchColNames.LABEL].detach().cpu().numpy(),
            rankings_batch=rankings.detach().cpu().numpy(),
        )

        self._log_metrics(loss, accuracy, map, "val")

        # Calculate MAP for all misconceptions
        question_embeddings = (
            self.get_query_features(
                input_ids=batch[self.TorchColNames.QUESTION_IDS],
                attention_mask=batch[self.TorchColNames.QUESTION_MASK],
            )
            .detach()
            .cpu()
            .numpy()
        )
        top_k_misconceptions = self.searcher.batch_search(
            question_embeddings,
            self.misconception_embeddings,
            k=25,
        )
        map_score = self.map_calculator.calculate_batch_map(
            actual_indices=batch[self.TorchColNames.LABEL].detach().cpu().numpy(),
            rankings_batch=top_k_misconceptions,
        )
        self.map_score_list.append(map_score)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.misconception_embeddings = None
        self.log(
            f"map_score_{self.config.fold}", torch.tensor(self.map_score_list).mean()
        )
        return super().on_validation_epoch_end()

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
