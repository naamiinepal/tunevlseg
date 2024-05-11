import copy

import torch
from torch import nn

from .base_shared_learner import BaseSharedLearner


class SharedAttnLearner(BaseSharedLearner):
    def __init__(
        self,
        *,
        textual_dim: int,
        visual_dim: int,
        unified_projector: type[nn.TransformerEncoderLayer] | None = None,
        prompt_depth: int = BaseSharedLearner.MIN_PROMPT_DEPTH,
        use_unified_projection: bool = True,
        **kwargs,
    ) -> None:
        if unified_projector is None:
            raise NotImplementedError(
                "You need to provide a transformer encoder layer for the "
                "unified projection layer from the config."
            )

        kwargs["context_initializer"] = None
        kwargs["tokenizer"] = None
        kwargs["embedding_layer"] = None

        super().__init__(prompt_depth=prompt_depth, **kwargs)

        transformer_layer = unified_projector(d_model=textual_dim + visual_dim)  # type:ignore

        self.common_projection_layers = nn.ModuleList(
            (transformer_layer,) * prompt_depth
            if use_unified_projection
            else (copy.deepcopy(transformer_layer) for _ in range(prompt_depth))
        )

        # Make two caches to keep the memory usage minimal
        self._computed_textual_context_cache: dict[int, torch.Tensor] = {}
        self._computed_visual_context_cache: dict[int, torch.Tensor] = {}

    def _get_combined_transformed_context(
        self,
        is_curr_branch_textual: bool,
        in_context: torch.Tensor | None = None,
        index: int = 0,
    ):
        read_cache_dict = (
            self._computed_textual_context_cache
            if is_curr_branch_textual
            else self._computed_visual_context_cache
        )
        transformed_prompt = read_cache_dict.get(index)

        if transformed_prompt is not None:
            # Delete the item from the cache, so only cache cross modality inputs
            del read_cache_dict[index]

            # Move to the correct device before returning the value
            return transformed_prompt.to(self.context_vectors.device)

        if in_context is None:
            in_context = self.context_vectors[index].unsqueeze(0)

        if in_context.ndim != 3:
            raise ValueError(
                "The tensor needs to have 3 dimensions: (batch, context_len, hidden_dim)"
            )

        transformer_shared_prompt: torch.Tensor = self.projection_layers[index](
            in_context
        ).unsqueeze(0)

        if is_curr_branch_textual:
            write_cache_dict = self._computed_visual_context_cache
            # Cache visual context when current branch is textual
            cache_val = transformer_shared_prompt[:, self.textual_dim :]
            return_val = transformer_shared_prompt[:, : self.textual_dim]
        else:
            write_cache_dict = self._computed_textual_context_cache
            # Cache textual context when current branch is visual
            cache_val = transformer_shared_prompt[:, : self.textual_dim]
            return_val = transformer_shared_prompt[:, self.textual_dim :]

        # Cache and move to CPU to reduce memory usage
        write_cache_dict[index] = cache_val.cpu()

        return return_val

    def get_transformed_textual_context(self, *args, **kwargs) -> torch.Tensor:
        return self._get_combined_transformed_context(
            *args, is_curr_branch_textual=True, **kwargs
        )

    def get_visual_context(self, *args, **kwargs) -> torch.Tensor:
        return self._get_combined_transformed_context(
            *args, is_curr_branch_textual=False, **kwargs
        )
