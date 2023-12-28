# pyright: reportGeneralTypeIssues=false
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
from torch import nn
from transformers import CLIPVisionModel

from .decoder import TransDecoder
from .encoder import TransTextEncoder

StrOrPath = Union[str, Path]
StrToAny = Mapping[str, torch.Tensor]


class TransformerSegmentor(nn.Module):
    def __init__(
        self,
        image_pretrained_model_name_or_path: StrOrPath,
        text_pretrained_model_name_or_path: StrOrPath,
        freeze_image_encoder: bool,
        freeze_text_encoder: bool,
        add_pos_enc: bool,
        decoder_layer_kwargs: StrToAny,
        num_decoder_layers: int,
        num_upsampler_layers: int,
        final_image_size: Optional[int] = None,
        num_output_channels: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """The segmentation model that uses text encoder as memory and image as target to output segmentation masks.
            The model uses `num_upsampler_layers` upsampling layers on the top of transformer decoder blocks.

        Args:
        ----
            image_pretrained_model_name_or_path: The name to the pretrained model or path to the saved image model
            text_pretrained_model_name_or_path: The name to the pretrained model or path to the saved text model
            freeze_image_encoder: Whether to freeze the image encoder of CLIP. Freezing disables the gradient of `CLIPVisionModel`.
            freeze_text_encoder: Whether to freeze the text encoder of CLIP. Freezing disables the gradient of `CLIPTextModel`.
            decoder_layer_kwargs: The keyword arguments to the transformer decoder. `n_head` is the required one.
            num_decoder_layers: The number of transformer decoder layers.
            num_upsampler_layers: The number of upsample and conv2d blocks to upsample.
            final_image_size: The final image size to align the output exactly to the input.
            num_output_channels: The number of channels in he output.
                This defaults to `1`  for the binary segmentation task.
        """
        super().__init__(*args, **kwargs)

        # Freeze image encoder if needed
        self.image_encoder = CLIPVisionModel.from_pretrained(
            image_pretrained_model_name_or_path,
        ).requires_grad_(not freeze_image_encoder)

        img_config = self.image_encoder.config

        image_hidden_size = img_config.hidden_size

        self.text_encoder = TransTextEncoder(
            pretrained_model_name_or_path=text_pretrained_model_name_or_path,
            freeze_clip=freeze_text_encoder,
            image_hidden_size=image_hidden_size,
        )

        # Use the image size from the image encoder, if not provided explicitly
        image_size = final_image_size or img_config.image_size

        self.decoder = TransDecoder(
            image_hidden_size=image_hidden_size,
            decoder_layer_kwargs=decoder_layer_kwargs,
            num_decoder_layers=num_decoder_layers,
            patch_size=img_config.patch_size,
            num_upsampler_layers=num_upsampler_layers,
            final_image_size=image_size,
            num_output_channels=num_output_channels,
        )

        self.add_pos_enc_or_identity = (
            self.get_with_pos_enc if add_pos_enc else nn.Identity()
        )

    def forward(self, text_input: StrToAny, image_input: torch.Tensor):
        # shape: (B, N_t, H_i)
        text_proj_output = self.text_encoder(**text_input)

        # image object with pooled output and hidden states
        image_output_obj = self.image_encoder(image_input)

        # shape: (B, N_i, H_i)
        image_last_hidden_state = image_output_obj.last_hidden_state

        # Apply positional encoding, if needed
        text_with_pos_enc = self.add_pos_enc_or_identity(text_proj_output)
        image_with_pos_enc = self.add_pos_enc_or_identity(image_last_hidden_state)

        # shape: (B, num_output_channels, H, W)
        return self.decoder(
            tgt=image_with_pos_enc,
            memory=text_with_pos_enc,
        )

    @staticmethod
    def get_with_pos_enc(x: torch.Tensor):
        # Get the number of tokens and hidden size
        B, N, H = x.shape

        # Get positional encoding
        posenc = TransformerSegmentor.get_posenc(
            d_model=H, token_length=N, device=x.device, dtype=x.dtype
        )

        # Add positional encoding to the input and return it
        return x + posenc

    @staticmethod
    @lru_cache()
    @torch.no_grad()
    def get_posenc(
        d_model: int,
        token_length: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if d_model % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})"
            )

        # Create a tensor of shape (token_length, d_model)
        posenc = torch.zeros(token_length, d_model, device=device, dtype=dtype)

        # Create a tensor of shape (token_length, 1)
        position = torch.arange(token_length, device=device, dtype=dtype).unsqueeze(1)

        # Create a tensor of shape (d_model // 2)
        mul_term = 1e-4 ** (
            torch.arange(0, d_model, 2, device=device, dtype=dtype) / d_model
        )

        # Create a tensor of shape (token_length, d_model // 2)
        angles = position * mul_term

        # Add sin and cos to the posenc tensor
        posenc[:, 0::2] = torch.sin(angles)
        posenc[:, 1::2] = torch.cos(angles)

        return posenc  # (token_length, d_model)
