from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch
from torch import nn

from .coop_context_learner import CoOpContextLearner

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase

    EmbeddingLayerType = Callable[[torch.Tensor], torch.Tensor]


class MapleContextLearner(CoOpContextLearner):
    MIN_PROMPT_DEPTH: Final = 1

    def __init__(
        self,
        *,
        visual_dim: int,
        max_network_depth: int,
        prompt_depth: int = MIN_PROMPT_DEPTH,
        use_unified_projection: bool = True,
        num_context: int | None = None,
        context_dim: int | None = None,
        context_initializer: str | list[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
    ) -> None:
        self.verify_prompt_depth(
            prompt_depth=prompt_depth,
            max_network_depth=max_network_depth,
        )

        # Shape: (prompt_depth, num_context, context_dim)
        context_vectors = self.get_context_vectors(
            num_context=num_context,
            context_dim=context_dim,
            context_initializer=context_initializer,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            prompt_depth=prompt_depth,
        )

        if context_vectors.ndim != 3:
            msg = "The number of dimensions of `context_vectors` must be 3"
            raise ValueError(msg)

        generated_prompt_depth, num_context, context_dim = context_vectors.shape
        if generated_prompt_depth != prompt_depth:
            msg = "The number of rows of `context_vectors` must be `prompt_depth`"
            raise ValueError(msg)

        print("Shape of context vector", context_vectors.shape)

        super(CoOpContextLearner).__init__()

        self.prompt_depth = prompt_depth
        self.num_context = num_context

        self.context_vectors = nn.Parameter(context_vectors)

        # Use the same internal object resulting in only one layer, for unified projection
        self.projection_layers = (
            (nn.Linear(context_dim, visual_dim),) * prompt_depth
            if use_unified_projection
            else tuple(nn.Linear(context_dim, visual_dim) for _ in range(prompt_depth))
        )

    @classmethod
    def verify_prompt_depth(cls, prompt_depth: int, max_network_depth: int) -> None:
        if prompt_depth < cls.MIN_PROMPT_DEPTH:
            msg = f"{prompt_depth=} must be at least {cls.MIN_PROMPT_DEPTH=}"
            raise ValueError(msg)

        if prompt_depth > max_network_depth:
            msg = f"{prompt_depth=} must be at most {max_network_depth=} for the used network."
            raise ValueError(msg)

    def get_context_vectors(
        self,
        num_context: int | None = None,
        context_dim: int | None = None,
        context_initializer: str | list[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        prompt_depth: int = MIN_PROMPT_DEPTH,
    ) -> torch.Tensor:
        if context_initializer is None:
            if num_context is None or context_dim is None:
                msg = "`num_context` and `context_dim` must be specified if `context_initializer` is None"
                raise ValueError(msg)

            return self.init_context_vectors((prompt_depth, num_context, context_dim))

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

        remaining_prompt_depth = prompt_depth - len(initialized_context_vectors)

        if remaining_prompt_depth == 0:
            return initialized_context_vectors

        if num_context is None or context_dim is None:
            msg = (
                "`num_context` and `context_dim` must be specified if `context_initializer` is not None "
                "and the length of `context_initializer` is less than `prompt_depth`"
            )
            raise ValueError(msg)

        random_context_vectors = self.init_context_vectors(
            (remaining_prompt_depth, num_context, context_dim),
        )

        return torch.cat((initialized_context_vectors, random_context_vectors))

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

    def generate_visual_contexts(
        self,
        text_intermediate_states: Iterable[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            proj_layer(curr_text_prompt[:, 1 : self.num_context + 1])
            for curr_text_prompt, proj_layer in zip(
                text_intermediate_states,
                self.projection_layers,
                strict=False,  # text_intermediate_states is usually longer then the projection layers
            )
        )
