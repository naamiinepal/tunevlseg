import itertools
from collections.abc import Iterable

import torch
from torch import nn

from .coop_context_learner import CoOpContextLearner


class BaseProjectorLearner(CoOpContextLearner):
    def __init__(
        self,
        *,
        proj_in_dim: int | None,
        proj_out_dim: int | None,
        prompt_depth: int = CoOpContextLearner.MIN_PROMPT_DEPTH,
        use_unified_projection: bool = True,
        intermediate_dim: int | Iterable[int] | None = None,
        use_proj_norm: bool = False,
        use_lora_proj: bool = False,
        use_final_bias: bool = True,
        **kwargs,
    ) -> None:
        if (
            use_lora_proj
            and intermediate_dim is not None
            and not isinstance(intermediate_dim, int)
        ):
            raise ValueError("Lora projection is only available for a single layer.")

        super().__init__(prompt_depth=prompt_depth, **kwargs)

        projection_init_kwargs = {
            "in_dim": proj_in_dim if proj_in_dim is not None else self.context_dim,
            "out_dim": proj_out_dim if proj_out_dim is not None else self.context_dim,
            "intermediate_dim": intermediate_dim,
            "use_final_norm": use_proj_norm,
            "use_final_bias": use_final_bias,
        }

        projection_layer_getter = (
            self.get_lora_projection
            if use_lora_proj and intermediate_dim is not None
            else self.get_mlp_projection
        )

        # Use the same internal object resulting in only one layer, for unified projection
        self.projection_layers = nn.ModuleList(
            (projection_layer_getter(**projection_init_kwargs),) * prompt_depth
            if use_unified_projection
            else (
                projection_layer_getter(**projection_init_kwargs)
                for _ in range(prompt_depth)
            )
        )

    def get_transformed_context(
        self, in_context: torch.Tensor | None = None, index: int = 0
    ) -> torch.Tensor:
        if in_context is None:
            in_context = self.context_vectors[index]

        return self.projection_layers[index](in_context)

    @staticmethod
    def get_lora_projection(
        in_dim: int,
        out_dim: int,
        intermediate_dim: int,
        use_final_norm: bool,
        use_final_bias: bool = True,
    ) -> nn.Sequential:
        layers = nn.Sequential()

        min_dim = min(out_dim, intermediate_dim)

        layers.append(nn.Linear(in_dim, min_dim, bias=False))

        # If intermediate dim is larger than or equals to out_dim, we don't need
        # to use an intermediate layer
        if intermediate_dim <= out_dim:
            layers.append(
                nn.Linear(
                    intermediate_dim,
                    out_dim,
                    bias=(not use_final_norm) and use_final_bias,
                )
            )

        if use_final_norm:
            layers.append(nn.LayerNorm(out_dim, bias=use_final_bias))

        return layers

    @staticmethod
    def get_mlp_projection(
        in_dim: int,
        out_dim: int,
        intermediate_dim: int | Iterable[int] | None,
        use_final_norm: bool,
        use_final_bias: bool = True,
    ) -> nn.Sequential | nn.Linear:
        if intermediate_dim is None:
            return nn.Linear(in_dim, out_dim)

        intermediate_dim = (
            (intermediate_dim,)
            if isinstance(intermediate_dim, int)
            else tuple(intermediate_dim)
        )
        layers = nn.Sequential(
            nn.Linear(in_dim, intermediate_dim[0]),
            nn.ReLU(inplace=True),
        )

        # for i, o in zip(intermediate_dim, intermediate_dim[1:], strict=False):
        for i, o in itertools.pairwise(intermediate_dim):
            layers.extend((nn.Linear(i, o), nn.ReLU(inplace=True)))

        # For every Linear layer, except the last one, init using kaiming_normal
        # This would increase the std of the existing layers weights
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity="relu")

        layers.append(
            nn.Linear(
                intermediate_dim[-1],
                out_dim,
                bias=(not use_final_norm) and use_final_bias,
            )
        )

        if use_final_norm:
            layers.append(
                nn.LayerNorm(out_dim, bias=use_final_bias),
            )

        return layers
