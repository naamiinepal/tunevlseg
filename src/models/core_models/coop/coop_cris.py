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
        freeze_all: bool = True,
    ) -> None:
        super().__init__(**model_cfg)

        if freeze_all:
            self.eval()
            self.requires_grad_(False)

        self.context_learner = ContextLearner(
            embedding_layer=self.backbone.token_embedding,
            **context_learner_cfg,
        )

    def get_pad_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Get pad mask from the parent method
        pad_mask = super().get_pad_mask(input_ids, attention_mask)

        # Expand the mask to include context
        return self.context_learner.update_pad_mask_for_context(
            pad_mask=pad_mask,
            max_length=self.word_len,
        )

    def encode_text(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _self = self.backbone

        x = _self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = self.context_learner.add_context_to_input_embeddings(
            x,
            max_length=self.word_len,
        )

        x = x + _self.positional_embedding[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = _self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = _self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        # Add indices by context number but clip to the word length
        second_indices = torch.minimum(
            text.argmax(dim=-1) + self.context_learner.num_context,
            torch.tensor(self.word_len - 1),
        )

        pooled_output = x[torch.arange(x.shape[0]), second_indices]

        state = pooled_output @ _self.text_projection

        return x, state
