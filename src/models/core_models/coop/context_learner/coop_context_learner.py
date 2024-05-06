from __future__ import annotations

from typing import TYPE_CHECKING

from .base_context_learner import BaseContextLearner

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

    from .base_context_learner import EmbeddingLayerType


class CoOpContextLearner(BaseContextLearner):
    def __init__(
        self,
        *,
        num_context: int | None = None,
        context_dim: int | None = None,
        context_initializer: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        vector_std: float = 0.02,
        **kwargs,
    ) -> None:
        context_vectors = self.get_context_vectors(
            num_context=num_context,
            context_dim=context_dim,
            context_initializer=context_initializer,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            vector_std=vector_std,
        )

        super().__init__(
            num_context=len(context_vectors), context_vectors=context_vectors
        )

    def get_context_vectors(
        self,
        num_context: int | None = None,
        context_dim: int | None = None,
        context_initializer: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        vector_std: float = 0.02,
    ) -> torch.Tensor:
        if context_initializer is None:
            if num_context is None or context_dim is None:
                msg = "`num_context` and `context_dim` must be specified if `context_initializer` is None"
                raise ValueError(msg)

            return self.init_context_vectors((num_context, context_dim), std=vector_std)

        if tokenizer is None or embedding_layer is None:
            msg = "If `context_initializer` is not None, `tokenizer` and `embedding_layer` must be specified"
            raise ValueError(msg)

        return self.get_context_vectors_from_initializer(
            context_initializer,
            embedding_layer,
            tokenizer,
        )[0]
