import torch
from torch import nn


class ContextLearnerMixin(nn.Module):
    def __init__(
        self,
        context_initializer: torch.Tensor | None = None,
        num_context: int | None = None,
        context_dim: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        if context_initializer is None:
            if num_context is None or context_dim is None:
                msg = "`num_context` and `context_dim` must be specified if `context_initializer` is None"
                raise ValueError(msg)

            context_initializer = self.init_context_vectors(
                num_context=num_context,
                context_dim=context_dim,
            )

        super().__init__(*args, **kwargs)

        self.num_context = len(context_initializer)

        self.context_vectors = nn.Parameter(context_initializer)

    def init_context_vectors(self, num_context: int, context_dim: int) -> torch.Tensor:
        # Copied from Context Optimization (CoOp).
        # Learning to Prompt for Vision-Language Models
        context_vectors = torch.empty(num_context, context_dim)
        nn.init.normal_(context_vectors, std=0.02)
        return context_vectors

    def add_context_to_input_embeddings(
        self,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # Intercept and add the learnable contexts
        first_embed = input_embeddings[:, :1, :]  # type:ignore
        rest_embed = input_embeddings[:, 1:, :]  # type:ignore

        return torch.cat(
            (first_embed, self.context_vectors, rest_embed),
            dim=1,
        )
