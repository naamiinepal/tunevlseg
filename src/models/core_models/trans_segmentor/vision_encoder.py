from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModel

if TYPE_CHECKING:
    from pathlib import Path


class TransVisionEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_or_path: str | Path,
        image_size: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model: CLIPVisionModel = CLIPVisionModel.from_pretrained(
            pretrained_model_or_path,
            output_hidden_states=True,
        )  # type:ignore

        self.config = self.model.config  # type:ignore

        if image_size is not None and image_size != self.config.image_size:  # type:ignore
            self.resize_position_embedding(image_size)

    @torch.no_grad
    def resize_position_embedding(
        self,
        new_image_size: int | tuple[int] | list[int],
    ) -> None:
        # Calculate new number of positions based on the new image size
        new_num_patches = (new_image_size // self.config.patch_size) ** 2
        new_num_positions = new_num_patches + 1

        embeddings = self.model.vision_model.embeddings

        old_position_embedding_weight = embeddings.position_embedding.weight.data

        # Resize position embedding weights using linear interpolation
        resized_weights = (
            F.interpolate(
                old_position_embedding_weight.T.unsqueeze(0),
                size=new_num_positions,
                mode="linear",
                align_corners=False,
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
            torch.arange(new_num_positions).unsqueeze(0),
            persistent=False,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)
