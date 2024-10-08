from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, CLIPModel, SiglipModel

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.siglip.modeling_siglip import (
        SiglipMultiheadAttentionPoolingHead,
    )


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike,
        use_existing_proj: bool,
        freeze_encoders: bool,
        image_size: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Get the tranformer text encoder with its hidden output projected to image provided dimension.

        Args:
        ----
            pretrained_model_name_or_path: The name to the pretrained model or path to the saved model
            use_existing_proj: Use the projection layer provided by `AutoModel`.
            image_size: The size of the image to resize the image position embeddings to.

        """
        super().__init__()

        self.model: CLIPModel | SiglipModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )

        config = self.model.config

        self.text_config = config.text_config
        self.vision_config = config.vision_config

        (
            self.text_projection,
            self.visual_projection,
            self.projection_dim,
        ) = self.get_projections(use_existing_proj)

        if image_size is not None and image_size != self.vision_config.image_size:
            self.resize_image_position_embedding(image_size)

        self.perform_freezing(freeze_encoders, use_existing_proj)

    def perform_freezing(self, freeze_encoders: bool, use_existing_proj: bool) -> None:
        # Compute gradient when not frozen
        self.requires_grad_(not freeze_encoders)

        # Compute the gradient for the newly initialized projection layer
        # Even when model frozen, we still need to compute gradients for this layer
        if not use_existing_proj:
            self.text_projection.requires_grad_(True)

    def get_projections(self, use_existing_proj: bool):
        text_hidden_size = self.text_config.hidden_size
        image_hidden_size = self.vision_config.hidden_size
        if use_existing_proj:
            text_projection = getattr(
                self.model,
                "text_projection",
                getattr(self.model.text_model, "head", nn.Identity()),
            )
            visual_projection = getattr(
                self.model,
                "visual_projection",
                self._siglip_project_image,
            )
            projection_dim: int = getattr(
                self.model.config,
                "projection_dim",
                image_hidden_size,
            )
        else:
            # Make textual projection layer identity if hidden sizes match
            text_projection = (
                nn.Identity()
                if text_hidden_size == image_hidden_size
                else nn.Linear(text_hidden_size, image_hidden_size)
            )
            visual_projection = nn.Identity()

            projection_dim: int = image_hidden_size

        return text_projection, visual_projection, projection_dim

    def _siglip_project_image(
        self,
        hidden_state: torch.FloatTensor,
    ) -> torch.FloatTensor:
        model: SiglipMultiheadAttentionPoolingHead = self.model.vision_model.head

        batch_size = hidden_state.shape[0]
        probe = model.probe.repeat(batch_size, 1, 1)

        hidden_state = model.attention(probe, hidden_state, hidden_state)[0]

        return hidden_state + model.mlp(model.layernorm(hidden_state))

    def get_text_features(self, *args, **kwargs) -> torch.FloatTensor:
        text_outputs: BaseModelOutputWithPooling = self.model.text_model(
            *args,
            **kwargs,
        )

        # shape: (B, N_t, H_t)
        text_embeds = text_outputs.last_hidden_state

        # shape: (B, N_t, H_p)
        return self.text_projection(text_embeds)

    def get_image_features(self, *args, **kwargs) -> torch.FloatTensor:
        visual_outputs: BaseModelOutputWithPooling = self.model.vision_model(
            *args,
            **kwargs,
        )

        # shape: (B, N_t, H_i)
        image_embeds = visual_outputs.last_hidden_state

        # shape: (B, N_t, H_p)
        return self.visual_projection(image_embeds)

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        return self.get_text_features(**text_input), self.get_image_features(
            image_input,
        )

    @torch.no_grad
    def resize_image_position_embedding(
        self,
        new_image_size: int,
    ) -> None:
        # Calculate new number of positions based on the new image size
        new_num_patches = (new_image_size // self.vision_config.patch_size) ** 2
        new_num_positions = new_num_patches + 1

        embeddings = self.model.vision_model.embeddings

        old_position_embedding_weight = embeddings.position_embedding.weight.data

        # Resize position embedding weights using linear interpolation
        # Other available interpolation methods include: nearest, area, nearest-exact
        resized_weights = (
            F.interpolate(
                old_position_embedding_weight.T.unsqueeze(0),
                size=new_num_positions,
                mode="linear",
                align_corners=False,
                antialias=False,
            )
            .squeeze(0)
            .T
        )

        # Update the embeddings attributes
        embeddings.position_embedding.weight.data = resized_weights

        # Update num_positions and position_ids buffer
        embeddings.num_patches = new_num_patches
        embeddings.num_positions = new_num_positions

        embeddings.register_buffer(
            "position_ids",
            torch.arange(
                new_num_positions,
                dtype=embeddings.position_ids.dtype,
                device=embeddings.position_ids.device,
            ).unsqueeze(0),
            persistent=False,
        )
