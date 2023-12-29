# pyright: reportGeneralTypeIssues=false
from pathlib import Path
from typing import Union

from torch import nn
from transformers import CLIPTextModel


class TransTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        freeze_clip: bool,
        image_hidden_size: int,
        *args,
        **kwargs,
    ) -> None:
        """Get the tranformer text encoder with its hidden output projected to image provided dimension.

        Args:
        ----
            pretrained_model_name_or_path: The name to the pretrained model or path to the saved model
            freeze_clip: Whether to freeze the text encoder of CLIP. Freezing disables the gradient of `CLIPTextModel`.
            image_hidden_size: The dimension to project the output of the text encoder.
                If it matches to that of text encoder, no projection layer is used.
        """
        super().__init__(*args, **kwargs)

        # Freeze clip model if needed
        self.text_model = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
        ).requires_grad_(not freeze_clip)

        text_hidden_size = self.text_model.config.hidden_size

        # Make projection layer identity if hidden sizes match
        self.proj_layer = (
            nn.Identity()
            if text_hidden_size == image_hidden_size
            else nn.Linear(text_hidden_size, image_hidden_size)
        )

    def forward(self, *args, **kwargs):
        text_output = self.text_model(*args, **kwargs)

        # shape: (B, N_t, H_t)
        text_last_hidden_state = text_output.last_hidden_state

        # shape: (B, N_t, H_i)
        return self.proj_layer(text_last_hidden_state)