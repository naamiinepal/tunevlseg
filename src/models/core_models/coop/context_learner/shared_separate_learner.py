import copy
from collections.abc import Iterable

import torch
from torch import nn

from .base_projector_learner import BaseProjectorLearner
from .base_shared_learner import BaseSharedLearner


class SharedSeparateLearner(BaseSharedLearner):
    def __init__(
        self,
        *,
        textual_dim: int,
        visual_dim: int,
        shared_dim: int = 64,
        prompt_depth: int = BaseSharedLearner.MIN_PROMPT_DEPTH,
        use_unified_projection: bool = True,
        intermediate_dim: int | Iterable[int] | None = None,
        use_proj_norm: bool = False,
        use_lora_proj: bool = False,
        **kwargs,
    ) -> None:
        if use_lora_proj and not isinstance(intermediate_dim, int):
            raise ValueError("Lora projection is only available for a single layer.")

        super().__init__(prompt_depth=prompt_depth, **kwargs)
        projection_layer_getter = (
            BaseProjectorLearner.get_lora_projection
            if use_lora_proj
            else BaseProjectorLearner.get_mlp_projection
        )

        projection_init_kwargs = {
            "in_dim": shared_dim,
            "out_dim": textual_dim,
            "intermediate_dim": intermediate_dim,
            "use_final_norm": use_proj_norm,
        }

        single_text_layer = projection_layer_getter(**projection_init_kwargs)
        self.textual_projection_layers = self.get_projection_layers(
            single_text_layer,
            prompt_depth,
            use_unified_projection,
        )

        projection_init_kwargs["out_dim"] = visual_dim

        single_visual_layer = projection_layer_getter(**projection_init_kwargs)
        self.visual_projection_layers = self.get_projection_layers(
            single_visual_layer,
            prompt_depth,
            use_unified_projection,
        )

    @staticmethod
    def get_projection_layers(
        single_layer: nn.Module,
        prompt_depth: int,
        use_unified_projection,
    ) -> nn.ModuleList:
        # Use the same internal object resulting in only one layer, for unified projection
        return nn.ModuleList(
            (single_layer,) * prompt_depth
            if use_unified_projection
            else (copy.deepcopy(single_layer) for _ in range(prompt_depth))
        )

    def get_transformed_textual_context(
        self, in_context: torch.Tensor | None = None, index: int = 0
    ) -> torch.Tensor:
        if in_context is None:
            in_context = self.context_vectors[index]

        return self.textual_projection_layers[index](in_context)

    def get_transformed_visual_context(
        self, in_context: torch.Tensor | None = None, index: int = 0
    ) -> torch.Tensor:
        if in_context is None:
            in_context = self.context_vectors[index]

        return self.visual_projection_layers[index](in_context)
