from torch import Tensor

from src.model_development.recall_model import RecallModel


class TwoTowerModel(RecallModel):
    def _setup_encoder(self, config):
        self.query_model = self._get_model(config)
        self.docs_model = self._get_model(config)

    def get_query_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self._get_features(self.query_model, input_ids, attention_mask)

    def get_docs_features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self._get_features(self.docs_model, input_ids, attention_mask)
