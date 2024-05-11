from abc import ABC, abstractmethod

import torch

from .base_visual_learner import BaseVisualLearner
from .coop_context_learner import CoOpContextLearner


class BaseSharedLearner(CoOpContextLearner, BaseVisualLearner, ABC):
    @abstractmethod
    def get_transformed_textual_context(
        self, in_context: torch.Tensor | None = None, index: int = 0
    ) -> torch.Tensor: ...

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

        if context_vectors is None:
            context_vectors = self.context_vectors[index]

        transformed_context = self.get_transformed_textual_context(
            in_context=context_vectors,
            index=index,
        )

        return torch.cat(
            (
                first_embed,
                transformed_context.expand(input_embeddings.size(0), -1, -1),
                mid_embed,
                last_embed,
            ),
            dim=1,
        )
