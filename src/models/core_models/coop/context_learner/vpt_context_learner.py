from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base_visual_learner import BaseVisualLearner

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .base_unimodal_learner import EmbeddingLayerType


class VPTContextLearner(BaseVisualLearner):
    def __init__(self, **kwargs) -> None:
        kwargs["context_initializer"] = None
        kwargs["tokenizer"] = None
        kwargs["embedding_layer"] = None

        super().__init__(**kwargs)

    def get_context_vectors(
        self,
        num_context: int | None = None,
        context_dim: int | None = None,
        prompt_depth: int = BaseVisualLearner.MIN_PROMPT_DEPTH,
        context_initializer: str | list[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        vector_std: float = 0.02,
    ) -> torch.Tensor:
        if num_context is None or context_dim is None:
            msg = "`num_context` and `context_dim` must be specified for VPT"
            raise ValueError(msg)

        return self.init_random_context_vectors(
            (prompt_depth, num_context, context_dim), std=vector_std
        )

    def get_visual_context(
        self, in_context: torch.Tensor | None = None, index: int = 0
    ) -> torch.Tensor:
        return self.context_vectors[index]

    def forward(
        self,
        *,
        input_embeddings: torch.Tensor,
        max_length: int | None = None,
        image_features: torch.Tensor | None = None,
        context_vectors: torch.Tensor | None = None,
        index: int = 0,
    ) -> torch.Tensor:
        if context_vectors is None:
            # (num_context, context_dim) -> (batch, num_context, context_dim)
            context_vectors = self.context_vectors[index].expand(
                input_embeddings.size(0),
                -1,
                -1,
            )

        # Add the context to the hidden states at the last position
        return torch.cat((input_embeddings, context_vectors), dim=1)
