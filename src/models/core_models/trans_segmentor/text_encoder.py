from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from transformers import CLIPTextModel

if TYPE_CHECKING:
    from pathlib import Path

    import torch


class TransTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str | Path,
        image_hidden_size: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Get the tranformer text encoder with its hidden output projected to image provided dimension.

        Args:
        ----
            pretrained_model_name_or_path: The name to the pretrained model or path to the saved model
            image_hidden_size: The dimension to project the output of the text encoder.
                If it matches to that of text encoder, no projection layer is used.
        """
        super().__init__(*args, **kwargs)

        # Freeze clip model if needed
        self.model: CLIPTextModel = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
        )  # type:ignore

        self.config = self.model.config  # type:ignore

        text_hidden_size = self.config.hidden_size

        # Make projection layer identity if hidden sizes match
        self.proj_layer = (
            nn.Identity()
            if image_hidden_size is None or text_hidden_size == image_hidden_size
            else nn.Linear(text_hidden_size, image_hidden_size)
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        text_output = self.model(*args, **kwargs)

        # shape: (B, N_t, H_t)
        text_last_hidden_state = text_output.last_hidden_state

        # shape: (B, N_t, H_i)
        return self.proj_layer(text_last_hidden_state)