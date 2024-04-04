from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from src.models.components.cris_model import CRIS

from .context_learner import CoCoOpContextLearner, CoOpContextLearner

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.models.components.cris_model import CLIPImageOutputType


class COOPCRIS(CRIS):
    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        context_learner: type[CoOpContextLearner | CoCoOpContextLearner],
        freeze_all: bool = True,
    ) -> None:
        super().__init__(**model_cfg)

        if freeze_all:
            self.eval()
            self.requires_grad_(False)

        self.context_learner = context_learner(
            visual_dim=self.backbone.visual.output_dim,
            context_dim=self.word_dim,
            embedding_layer=self.backbone.token_embedding,
        )

        # CRIS has vision output in the form of (batch_size, output_dim, H, W)
        # So need to pool it without any learnable params to (batch_size, output_dim)
        # Since image features is not needed for CoOp, no need for an extra computation
        self.image_features_pooler_or_identity = (
            self._pool_4D_tensor
            if context_learner != CoOpContextLearner
            else nn.Identity()
        )

    @staticmethod
    def _pool_4D_tensor(x: torch.Tensor) -> torch.Tensor:
        # Leave batch and channel dimension, pooler from other dims
        return x.mean((2, 3))

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
            max_length=self.max_length,
        )

    def encode_text(
        self,
        text: torch.Tensor,
        image_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _self = self.backbone

        x = _self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = self.context_learner(
            input_embeddings=x,
            max_length=self.max_length,
            image_features=image_features,
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
            torch.tensor(self.max_length - 1),
        )

        pooled_output = x[torch.arange(x.shape[0]), second_indices]

        state = pooled_output @ _self.text_projection

        return x, state

    def get_unimodal_outputs(
        self,
        image_input: torch.Tensor,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[CLIPImageOutputType, torch.Tensor, torch.Tensor]:
        # vis: C3 / C4 / C5
        # input_ids: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(image_input)

        # Get image features from the last element
        image_features = self.image_features_pooler_or_identity(vis[-1])

        input_ids, state = self.encode_text(input_ids, image_features, *args, **kwargs)
        return vis, input_ids, state
