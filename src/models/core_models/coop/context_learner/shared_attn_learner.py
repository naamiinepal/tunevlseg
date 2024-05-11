import copy

import torch
from torch import nn

from .coop_context_learner import CoOpContextLearner


class SharedAttnLearner(CoOpContextLearner):
    def __init__(
        self,
        *,
        textual_dim: int,
        visual_dim: int,
        unified_projector: type[nn.TransformerEncoderLayer] | None = None,
        prompt_depth: int = CoOpContextLearner.MIN_PROMPT_DEPTH,
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

    def get_transformed_context(
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
        return self.get_transformed_context(
            *args, is_curr_branch_textual=True, **kwargs
        )

    def get_transformed_visual_context(self, *args, **kwargs) -> torch.Tensor:
        return self.get_transformed_context(
            *args, is_curr_branch_textual=False, **kwargs
        )

    def mutate_image_hidden_states(
        self, hidden_states: torch.Tensor, index: int
    ) -> None:
        hidden_states[:, -self.num_context :] = self.get_transformed_visual_context(
            index=index
        ).expand(hidden_states.size(0), -1, -1)

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
            context_vectors = self.get_transformed_textual_context(
                in_context=context_vectors,
                index=index,
            )

        return torch.cat(
            (
                first_embed,
                context_vectors.expand(input_embeddings.size(0), -1, -1),
                mid_embed,
                last_embed,
            ),
            dim=1,
        )
