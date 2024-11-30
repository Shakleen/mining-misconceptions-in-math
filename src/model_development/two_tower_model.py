from torch import Tensor

from src.model_development.recall_model import RecallModel
from src.model_development.latent_attention.latent_multi_head_attention import (
    LatentMultiHeadAttention,
)


class TwoTowerModel(RecallModel):
    def _setup_encoder(self, config):
        self.query_model = self._get_model(config)
        self.docs_model = self._get_model(config)

        if config.sentence_pooling_method == "attention":
            self.query_latent_attention_layer = LatentMultiHeadAttention(
                input_dim=self.query_model.config.hidden_size,
                hidden_dim=config.hidden_dim,
                num_latents=config.num_latents,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )
            self.docs_latent_attention_layer = LatentMultiHeadAttention(
                input_dim=self.docs_model.config.hidden_size,
                hidden_dim=config.hidden_dim,
                num_latents=config.num_latents,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )

    def get_query_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # self.latent_attention_layer = self.query_latent_attention_layer
        return self._get_features(self.query_model, input_ids, attention_mask)

    def get_docs_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # self.latent_attention_layer = self.docs_latent_attention_layer
        return self._get_features(self.docs_model, input_ids, attention_mask)
