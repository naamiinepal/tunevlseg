from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .decoder import TransDecoder
from .encoder import MultiModalEncoder

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike


class TransformerSegmentor(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike,
        use_existing_proj: bool,
        freeze_encoders: bool,
        add_pos_enc: bool,
        transformer_decoder: nn.TransformerDecoder,
        num_upsampler_layers: int,
        upsampler_act: nn.Module,
        upsampler_norm: nn.Module | str | None = None,
        upsampler_num_channels_in_group: int = 64,
        image_size: int | None = None,
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
            image_size: The final image size to align the output exactly to the input.
            num_output_channels: The number of channels in he output.
                This defaults to `1`  for the binary segmentation task.

        """
        super().__init__(*args, **kwargs)

        self.encoder = MultiModalEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_existing_proj=use_existing_proj,
            image_size=image_size,
            freeze_encoders=freeze_encoders,
        )

        vision_config = self.encoder.vision_config

        final_image_size: int = image_size or vision_config.image_size
        projection_dim = self.encoder.projection_dim

        if add_pos_enc:
            self.add_pos_embed_or_identity = self.get_with_pos_embed
            if projection_dim % 2 != 0:
                msg = f"Cannot use sin/cos positional encoding with odd dim (got dim={projection_dim})"
                raise ValueError(msg)
        else:
            self.add_pos_embed_or_identity = nn.Identity()

        self.decoder = TransDecoder(
            transformer_decoder=transformer_decoder,
            projection_dim=projection_dim,
            patch_size=vision_config.patch_size,
            num_upsampler_layers=num_upsampler_layers,
            final_image_size=final_image_size,
            upsampler_act=upsampler_act,
            num_output_channels=num_output_channels,
            upsampler_norm=upsampler_norm,
            upsampler_num_channels_in_group=upsampler_num_channels_in_group,
        )

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ) -> torch.Tensor:
        # shape: (B, N_t, H_p)
        text_embeds = self.encoder.get_text_features(**text_input)

        # shape: (B, N_i, H_p)
        image_embeds = self.encoder.get_image_features(image_input)

        # Apply positional encoding, if needed
        text_with_pos_enc = self.add_pos_embed_or_identity(text_embeds)
        image_with_pos_enc = self.add_pos_embed_or_identity(image_embeds)

        # shape: (B, num_output_channels, H, W)
        return self.decoder(
            tgt=image_with_pos_enc,
            memory=text_with_pos_enc,
            memory_mask=text_input.get("attention_mask"),
        )

    @staticmethod
    def get_with_pos_embed(x: torch.Tensor) -> torch.Tensor:
        # Get the number of tokens and hidden size
        _, N, H = x.shape

        # Get positional encoding
        posenc = TransformerSegmentor.generate_pos_embed(
            d_model=H,
            token_length=N,
            device=x.device,
            dtype=x.dtype,
        )

        # Add positional encoding to the input and return it
        return x + posenc

    @staticmethod
    @torch.no_grad()
    def generate_pos_embed(
        d_model: int,
        token_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # Create a tensor of shape (token_length, d_model)
        embeddings = torch.zeros(token_length, d_model, device=device, dtype=dtype)

        # Create a tensor of shape (token_length, 1)
        position = torch.arange(token_length, device=device, dtype=dtype).unsqueeze(1)

        # Create a tensor of shape (d_model // 2)
        mul_term = 1e-4 ** (
            torch.arange(0, d_model, 2, device=device, dtype=dtype) / d_model
        )

        # Create a tensor of shape (token_length, d_model // 2)
        angles = position * mul_term

        # Add sin and cos to the posenc tensor
        embeddings[:, 0::2] = torch.sin(angles)
        embeddings[:, 1::2] = torch.cos(angles)

        return embeddings  # (token_length, d_model)
