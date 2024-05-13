import torch
from torch import nn

from .base_projector_learner import BaseProjectorLearner


class CoCoOpContextLearner(BaseProjectorLearner):
    def __init__(
        self,
        *,
        visual_dim: int,
        norm_image_features: bool = True,
        **kwargs,
    ) -> None:
        kwargs["proj_in_dim"] = visual_dim
        kwargs["proj_out_dim"] = None
        kwargs["use_final_bias"] = False

        super().__init__(**kwargs)

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

    def get_textual_context(
        self,
        in_context: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        index: int = 0,
    ) -> torch.Tensor:
        if image_features is None:
            msg = "`image_features` must be provided when `context_vectors` is None for CoCoOp"
            raise ValueError(msg)

        # Normalize image features to align with CoCoOp
        image_features = self.image_features_normalizer_or_identity(image_features)

        # image_features: (batch, visual_dim)
        # bias: (batch, context_dim)
        bias: torch.Tensor = self.get_transformed_context(image_features, index)

        # bias: (batch, 1, context_dim)
        bias = bias.unsqueeze(1)

        if in_context is None:
            in_context = self.context_vectors[index]

        # context_vectors: (num_context, context_dim)
        # context_shifted: (batch, num_context, context_dim)
        return bias + in_context

    def forward(
        self,
        *,
        input_embeddings: torch.Tensor,
        max_length: int | None = None,
        image_features: torch.Tensor | None = None,
        context_vectors: torch.Tensor | None = None,
        index: int = 0,
    ) -> torch.Tensor:
        context_vectors = self.get_textual_context(
            in_context=context_vectors, image_features=image_features, index=index
        )

        return super().forward(
            input_embeddings=input_embeddings,
            max_length=max_length,
            context_vectors=context_vectors,
        )
