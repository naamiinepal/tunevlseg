from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from transformers import CLIPSegForImageSegmentation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    import torch


class HFCLIPSegWrapper(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike | None = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        model = self.get_pretrained_model(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )

        model.clip.requires_grad_(not freeze_encoder)
        model.decoder.requires_grad_(not freeze_decoder)

        self.model = model

    @staticmethod
    def get_pretrained_model(
        pretrained_model_name_or_path: str | PathLike | None,
        *args,
        **kwargs,
    ) -> CLIPSegForImageSegmentation:
        model = CLIPSegForImageSegmentation.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )

        if not isinstance(model, CLIPSegForImageSegmentation):
            msg = f"Expected `CLIPSegForImageSegmentation` from {pretrained_model_name_or_path}, got {type(model)}"
            raise ValueError(msg)

        return model

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        B, _, H, W = image_input.shape

        # The output is squeezed so it may be (H, W) or (B, H, W)
        outputs = self.model(**text_input, pixel_values=image_input)

        return outputs.logits.view(B, 1, H, W)
