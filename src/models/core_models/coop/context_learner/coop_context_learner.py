from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .base_unimodal_learner import BaseUnimodalLearner

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .base_unimodal_learner import EmbeddingLayerType


class CoOpContextLearner(BaseUnimodalLearner):
    def get_context_vectors(
        self,
        num_context: int | None = None,
        context_dim: int | None = None,
        prompt_depth: int = BaseUnimodalLearner.MIN_PROMPT_DEPTH,
        context_initializer: str | list[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        vector_std: float = 0.02,
    ) -> torch.Tensor:
        if context_initializer is None:
            if num_context is None or context_dim is None:
                msg = "`num_context` and `context_dim` must be specified if `context_initializer` is None"
                raise ValueError(msg)

            return self.init_random_context_vectors(
                (prompt_depth, num_context, context_dim), std=vector_std
            )

        if tokenizer is None or embedding_layer is None:
            msg = "If `context_initializer` is not None, `tokenizer` and `embedding_layer` must be specified"
            raise ValueError(msg)

        # Truncate the context to avoid tokenizing and running embedding layer for overflowing tokens
        truncated_context_initializer = (
            context_initializer
            if isinstance(context_initializer, str)
            else context_initializer[:prompt_depth]
        )

        initialized_context_vectors = self.get_context_vectors_from_initializer(
            truncated_context_initializer,
            embedding_layer,
            tokenizer,
        )

        initialized_depth, num_context, context_dim = initialized_context_vectors.shape

        remaining_prompt_depth = prompt_depth - initialized_depth

        if remaining_prompt_depth == 0:
            return initialized_context_vectors

        random_context_vectors = self.init_random_context_vectors(
            (remaining_prompt_depth, num_context, context_dim), std=vector_std
        )

        return torch.cat((initialized_context_vectors, random_context_vectors))

    @staticmethod
    def get_context_vectors_from_initializer(
        context_initializer: str | list[str],
        embedding_layer: EmbeddingLayerType,
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
        constructor: Literal["zeros", "ones"],
        max_length: int | None = None,
    ) -> torch.Tensor:
        # Generate attention mask for context vectors
        attention_mask_for_context_vectors: torch.Tensor = getattr(torch, constructor)(
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

    def get_textual_context(
        self,
        in_context: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        index: int = 0,
    ) -> torch.Tensor:
        return self.context_vectors[index]

    def mutate_text_hidden_states(
        self,
        hidden_states: torch.Tensor,
        index: int,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states[:, 1 : self.num_context + 1] = self.get_textual_context(
            image_features=image_features, index=index
        )

        return hidden_states

    def forward(
        self,
        *,
        input_embeddings: torch.Tensor,
        max_length: int | None = None,
        image_features: torch.Tensor | None = None,
        context_vectors: torch.Tensor | None = None,
        index: int = 0,
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
            context_vectors = self.get_textual_context(
                image_features=image_features, index=index
            ).expand(
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
