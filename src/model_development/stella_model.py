import os

import torch
from torch import Tensor
from transformers import AutoModel
import torch.nn.functional as F

from src.model_development.recall_model import RecallModel


class StellaModel(RecallModel):
    def _setup_encoder(self, config):
        self.model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
        )

        self.vector_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=1024,
        )
        vector_linear_dict = {
            k.replace("linear.", ""): v
            for k, v in torch.load(
                os.path.join(config.model_path, "2_Dense_1024/pytorch_model.bin")
            ).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)

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
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        features = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        features = F.normalize(self.vector_linear(features), p=2, dim=1)
        return features
