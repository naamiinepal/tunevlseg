from collections.abc import Mapping
from typing import Any

import torch

from src.models.components.cris_model import CRIS

from .context_learner import ContextLearner


class COOPCRIS(CRIS):
    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        context_learner_cfg: Mapping[str, Any],
    ) -> None:
        super().__init__(**model_cfg)

        self.context_learner = ContextLearner(
            embedding_layer=self.backbone.token_embedding,
            **context_learner_cfg,
        )

    def get_pad_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.context_learner.update_pad_mask_for_context(
            ~(attention_mask.bool()) if attention_mask is not None else input_ids == 0,
        )

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        _self = self.backbone

        x = _self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = self.context_learner.add_context_to_input_embeddings(x)

        x = x + _self.positional_embedding[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = _self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return _self.ln_final(x)
