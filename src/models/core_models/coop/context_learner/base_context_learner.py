from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from transformers import PreTrainedTokenizerBase

    EmbeddingLayerType = Callable[[torch.Tensor], torch.Tensor]


class BaseContextLearner(nn.Module):
    def __init__(self, *, num_context: int, context_vectors: torch.Tensor) -> None:
        super().__init__()

        self.num_context = num_context

        self.context_vectors = nn.Parameter(context_vectors)

        print("Shape of context vector", self.context_vectors.shape)

    @staticmethod
    def init_context_vectors(shape: Sequence[int], std: float = 0.02) -> torch.Tensor:
        # Copied from Context Optimization (CoOp).
        # Learning to Prompt for Vision-Language Models
        context_vectors = torch.empty(shape)
        nn.init.normal_(context_vectors, std=std)
        return context_vectors

    @staticmethod
    def get_context_vectors_from_initializer(
        context_initializer: str | list[str],
        embedding_layer: Callable[[torch.Tensor], torch.Tensor],
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        input_ids = tokenizer(
            context_initializer,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            add_special_tokens=False,
        ).input_ids

        with torch.no_grad():
            return embedding_layer(input_ids)

    def _update_mask_for_context(
        self,
        mask: torch.Tensor,
        constructor: str,
        max_length: int | None = None,
    ) -> torch.Tensor:
        # Generate attention mask for context vectors
        attention_mask_for_context_vectors = getattr(torch, constructor)(
            mask.shape[0],
            self.num_context,
            dtype=mask.dtype,
            device=mask.device,
        )

        # Prepend ones to the attention mask.
        return torch.cat((attention_mask_for_context_vectors, mask), dim=1)[
            :,
            :max_length,
        ]

    def update_attention_mask_for_context(
        self,
        attention_mask: torch.Tensor,
        max_length: int | None = None,
    ) -> torch.Tensor:
        return self._update_mask_for_context(attention_mask, "ones", max_length)

    def update_pad_mask_for_context(
        self,
        pad_mask: torch.Tensor,
        max_length: int | None = None,
    ) -> torch.Tensor:
        return self._update_mask_for_context(pad_mask, "zeros", max_length)

    def forward(
        self,
        *,
        input_embeddings: torch.Tensor,
        max_length: int | None = None,
        image_features: torch.Tensor | None = None,
        context_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input_embeddings: (batch, context_length, context_dim)
        # BOS Token
        first_embed = input_embeddings[:, :1, :]

        # Get the embeddings from the middle
        mid_embed_last_idx = (
            -1
            if max_length is None
            else min(max_length - self.num_context, input_embeddings.size(1)) - 1
        )

        # If max_length is provided, reduce the embeddings while preserving the EOS token
        mid_embed = input_embeddings[:, 1:mid_embed_last_idx, :]

        # May or may not be EOS Token, but doing this preserves the last token
        last_embed = input_embeddings[:, -1:, :]

        # May be precomputed, like in the case of cocoop
        if context_vectors is None:
            # (num_context, context_dim) -> (batch, num_context, context_dim)
            context_vectors = self.context_vectors.expand(
                input_embeddings.size(0),
                -1,
                -1,
            )

        return torch.cat(
            (
                first_embed,
                context_vectors,
                mid_embed,
                last_embed,
            ),
            dim=1,
        )