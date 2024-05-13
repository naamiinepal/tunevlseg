from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from transformers import PreTrainedTokenizerBase

    EmbeddingLayerType = Callable[[torch.Tensor], torch.Tensor]


class BaseUnimodalLearner(nn.Module, ABC):
    MIN_PROMPT_DEPTH: Final = 1

    def __init__(
        self,
        *,
        max_network_depth: int,
        prompt_depth: int = MIN_PROMPT_DEPTH,
        num_context: int | None = None,
        context_dim: int | None = None,
        context_initializer: str | list[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        vector_std: float = 0.02,
        **kwargs,
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
            vector_std=vector_std,
        )

        if context_vectors.ndim != 3:
            msg = "The number of dimensions of `context_vectors` must be 3"
            raise ValueError(msg)

        generated_prompt_depth, num_context, context_dim = context_vectors.shape
        if generated_prompt_depth != prompt_depth:
            msg = "The number of rows of `context_vectors` must be `prompt_depth`"
            raise ValueError(msg)

        super().__init__()

        self.prompt_depth = prompt_depth

        self.num_context = num_context
        self.context_dim = context_dim

        self.context_vectors = nn.Parameter(context_vectors)

        print("Shape of context vector", self.context_vectors.shape)

    @classmethod
    def verify_prompt_depth(cls, prompt_depth: int, max_network_depth: int) -> None:
        if prompt_depth < cls.MIN_PROMPT_DEPTH:
            msg = f"{prompt_depth=} must be at least {cls.MIN_PROMPT_DEPTH=}"
            raise ValueError(msg)

        if prompt_depth > max_network_depth:
            msg = f"{prompt_depth=} must be at most {max_network_depth=} for the used network."
            raise ValueError(msg)

    @staticmethod
    def init_random_context_vectors(
        shape: Sequence[int], std: float = 0.02
    ) -> torch.Tensor:
        # Copied from Context Optimization (CoOp).
        # Learning to Prompt for Vision-Language Models
        context_vectors = torch.empty(shape)
        nn.init.normal_(context_vectors, std=std)
        return context_vectors

    @abstractmethod
    def get_context_vectors(
        self,
        num_context: int | None = None,
        context_dim: int | None = None,
        prompt_depth: int = MIN_PROMPT_DEPTH,
        context_initializer: str | list[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embedding_layer: EmbeddingLayerType | None = None,
        vector_std: float = 0.02,
    ) -> torch.Tensor: ...
