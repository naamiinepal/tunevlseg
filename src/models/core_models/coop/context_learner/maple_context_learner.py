from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch
from torch import nn

from .base_context_learner import BaseContextLearner

if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers import PreTrainedTokenizerBase

    from .base_context_learner import EmbeddingLayerType


class MapleContextLearner(BaseContextLearner):
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
        intermediate_dim: int | Iterable[int] | None = None,
        use_proj_norm: bool = False,
        vector_std: float = 0.02,
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

        super().__init__(num_context=num_context, context_vectors=context_vectors)

        self.prompt_depth = prompt_depth

        projection_init_kwargs = {
            "context_dim": context_dim,
            "visual_dim": visual_dim,
            "intermediate_dim": intermediate_dim,
            "use_proj_norm": use_proj_norm,
        }

        # Use the same internal object resulting in only one layer, for unified projection
        self.projection_layers = nn.ModuleList(
            (self.get_text_projection(**projection_init_kwargs),) * prompt_depth
            if use_unified_projection
            else (
                self.get_text_projection(**projection_init_kwargs)
                for _ in range(prompt_depth)
            )
        )

    @staticmethod
    def get_text_projection(
        context_dim: int,
        visual_dim: int,
        intermediate_dim: int | Iterable[int] | None,
        use_proj_norm: bool = False,
    ) -> nn.Module:
        if intermediate_dim is None:
            return nn.Linear(context_dim, visual_dim)

        intermediate_dim = (
            (intermediate_dim,)
            if isinstance(intermediate_dim, int)
            else tuple(intermediate_dim)
        )
        network_layers = [
            nn.Linear(context_dim, intermediate_dim[0]),
            nn.ReLU(inplace=True),
        ]

        for i, o in zip(intermediate_dim, intermediate_dim[1:], strict=False):
            network_layers.extend((nn.Linear(i, o), nn.ReLU(inplace=True)))

        if use_proj_norm:
            network_layers.extend(
                (
                    nn.Linear(
                        intermediate_dim[-1], visual_dim, bias=False
                    ),  # Do not use bias when using normalzation layer
                    nn.LayerNorm(visual_dim),
                )
            )
        else:
            network_layers.append(nn.Linear(intermediate_dim[-1], visual_dim))

        net = nn.Sequential(*network_layers)

        # For every Linear layer, except the last one, init using kaiming_normal
        # This would increase the std of the existing layers weights
        linear_layers = tuple(layer for layer in net if isinstance(layer, nn.Linear))
        for layer in linear_layers[:-1]:
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity="relu")

        return net

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
        vector_std: float = 0.02,
    ) -> torch.Tensor:
        if context_initializer is None:
            if num_context is None or context_dim is None:
                msg = "`num_context` and `context_dim` must be specified if `context_initializer` is None"
                raise ValueError(msg)

            return self.init_context_vectors(
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
            (remaining_prompt_depth, num_context, context_dim), std=vector_std
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

    def generate_visual_contexts(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            proj_layer(curr_ctx_vec)
            for proj_layer, curr_ctx_vec in zip(
                self.projection_layers, self.context_vectors, strict=True
            )
        )
