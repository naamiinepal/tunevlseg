import torch
from torch import nn

from .coop_context_learner import CoOpContextLearner


class CoCoOpContextLearner(CoOpContextLearner):
    def __init__(
        self,
        *,
        visual_dim: int,
        intermediate_dim: int | None = None,
        reduction_factor: float | None = 16,
        norm_image_features: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.meta_net = self.get_meta_net(
            visual_dim=visual_dim,
            context_dim=self.context_vectors.size(1),
            intermediate_dim=intermediate_dim,
            reduction_factor=reduction_factor,
        )

        self.image_features_normalizer_or_identity = (
            self._normalize_features if norm_image_features else nn.Identity()
        )

    @staticmethod
    def _normalize_features(
        features: torch.Tensor,
        p: float | str | None = "fro",
        dim: int = -1,
    ) -> torch.Tensor:
        return features / features.norm(p=p, dim=dim, keepdim=True)

    @staticmethod
    def get_meta_net(
        visual_dim: int,
        context_dim: int,
        intermediate_dim: int | None,
        reduction_factor: float | None,
    ) -> nn.Sequential:
        if intermediate_dim is None:
            if reduction_factor is None:
                msg = (
                    "Either `intermediate_dim` or `reduction_factor` must be specified"
                )
                raise ValueError(msg)

            intermediate_dim = int(visual_dim / reduction_factor)

        return nn.Sequential(
            nn.Linear(visual_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, context_dim),
        )

    def forward(
        self,
        *,
        input_embeddings: torch.Tensor,
        max_length: int | None = None,
        image_features: torch.Tensor | None = None,
        context_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_vectors is None:
            if image_features is None:
                msg = "`image_features` must be provided when `context_vectors` is None for CoCoOp"
                raise ValueError(msg)

            # Normalize image features to align with CoCoOp
            image_features = self.image_features_normalizer_or_identity(image_features)

            # image_features: (batch, visual_dim)
            # bias: (batch, context_dim)
            bias: torch.Tensor = self.meta_net(image_features)

            # bias: (batch, 1, context_dim)
            bias = bias.unsqueeze(1)

            # context_vectors: (num_context, context_dim)
            # context_shifted: (batch, num_context, context_dim)
            context_vectors = bias + self.context_vectors

        return super().forward(
            image_features=image_features,
            input_embeddings=input_embeddings,
            max_length=max_length,
            context_vectors=context_vectors,
        )