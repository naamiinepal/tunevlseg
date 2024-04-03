from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedTokenizerBase


class ContextLearner(nn.Module):
    def __init__(
        self,
        *,
        num_context: int | None = None,
        context_dim: int | None = None,
        context_initializer: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if context_initializer is None:
            if num_context is None or context_dim is None:
                msg = "`num_context` and `context_dim` must be specified if `context_initializer` is None"
                raise ValueError(msg)

            context_vectors = self.init_context_vectors(
                num_context=num_context,
                context_dim=context_dim,
            )
        else:
            if tokenizer is None or embedding_layer is None:
                msg = "If `context_initializer` is not None, `tokenizer` and `embedding_layer` must be specified"
                raise ValueError(msg)

            context_vectors = self.get_context_vectors_from_initializer(
                context_initializer,
                embedding_layer,
                tokenizer,
            )

        print("Shape of context vector", context_vectors.shape)

        nn.Module.__init__(self)

        self.num_context = len(context_vectors)

        self.context_vectors = nn.Parameter(context_vectors)

    @staticmethod
    def init_context_vectors(num_context: int, context_dim: int) -> torch.Tensor:
        # Copied from Context Optimization (CoOp).
        # Learning to Prompt for Vision-Language Models
        context_vectors = torch.empty(num_context, context_dim)
        nn.init.normal_(context_vectors, std=0.02)
        return context_vectors

    @staticmethod
    def get_context_vectors_from_initializer(
        context_initializer: str,
        embedding_layer: Callable[[torch.Tensor], torch.Tensor],
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        input_ids = tokenizer(
            context_initializer,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            add_special_tokens=False,
        ).input_ids[0]

        with torch.no_grad():
            return embedding_layer(input_ids)

    def add_context_to_input_embeddings(
        self,
        input_embeddings: torch.Tensor,
        max_length: int | None = None,
    ) -> torch.Tensor:
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

        return torch.cat(
            (
                first_embed,
                self.context_vectors.expand(input_embeddings.size(0), -1, -1),
                mid_embed,
                last_embed,
            ),
            dim=1,
        )

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
