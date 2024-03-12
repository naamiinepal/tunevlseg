from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, TypeVar

from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from torch.nn.common_types import _ratio_any_t, _size_2_t, _size_any_t

    StrOrModule = str | nn.Module


class TransDecoder(nn.Module):
    def __init__(
        self,
        transformer_decoder: nn.TransformerDecoder,
        projection_dim: int,
        patch_size: int,
        num_upsampler_layers: int,
        final_image_size: int,
        upsampler_act: nn.Module,
        num_output_channels: int = 1,
        upsampler_norm: StrOrModule | None = None,
        upsampler_num_channels_in_group: int = 64,
        *args,
        **kwargs,
    ) -> None:
        """Segmentation decoder using the encoded representation from text and vision branch.
        It treats the encoded hidden states from vision encoder as the target domain and textual representation as the memory.

        Args:
        ----
            image_hidden_size: The hidden size of the image encoder.
            decoder_layer_kwargs: The keyword arguments to the transformer decoder. `n_head` is the required one.
            num_decoder_layers: The number of transformer decoder layers.
            patch_size: The size of the patch. This is the total scale factor for the transformer output.
            num_upsampler_layers: The number of upsample and conv2d blocks to upsample.
            final_image_size: The final image size to align the output exactly to the input.
            num_output_channels: The number of channels in he output.
                This defaults to `1`  for the binary segmentation task.
            cross_attn_first: Whether to use cross attention before self attention. Defaults to `False`.

        """
        super().__init__(*args, **kwargs)

        self.transformer_decoder = transformer_decoder

        first_transformer_layer: nn.TransformerDecoderLayer = (
            transformer_decoder.layers[0]
        )  # type: ignore

        # Needed to generate attention mask
        self.num_heads: int = first_transformer_layer.self_attn.num_heads

        self.upsampler = self.get_upsampler(
            patch_size=patch_size,
            num_upsampler_layers=num_upsampler_layers,
            projection_dim=projection_dim,
            final_image_size=final_image_size,
            num_output_channels=num_output_channels,
            norm=upsampler_norm,
            group_num_channels=upsampler_num_channels_in_group,
            activation=upsampler_act,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        memory_mask = self.get_memory_mask_from_attention_mask(
            attention_mask, tgt.size(1)
        )

        # shape: (B, N_i + 1, H_i)
        trans_output: torch.Tensor = self.transformer_decoder(
            *args,
            tgt=tgt,
            memory_mask=memory_mask,
            **kwargs,
        )

        img_seq_len = trans_output.size(1)
        sqrt_seq_len = math.isqrt(img_seq_len)

        if (sqrt_seq_len * sqrt_seq_len) != img_seq_len:
            # Remove the pooled output
            # shape: (B, N_i, H_i)
            trans_output = trans_output[:, 1:, :]

        # Move the hidden dimension to make it channel first
        # shape: (B, H_i, N_i)
        channel_first_output = trans_output.movedim(-1, 1)

        hidden_dim = channel_first_output.size(1)

        # Make the output 4D
        # shape: (B, H_i, sqrt(N_i), sqrt(N_i))
        img_channel_first = channel_first_output.view(
            -1, hidden_dim, sqrt_seq_len, sqrt_seq_len
        )

        return self.upsampler(img_channel_first)

    def get_memory_mask_from_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        N_i: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        B, N_t = attention_mask.shape
        n_heads = self.num_heads

        # shape: (B * n_head, N_i, N_t)
        # also negate the attention mask to get the memory_mask
        return ~(
            attention_mask.expand(n_heads * N_i, B, N_t)
            .moveaxis(1, 0)
            .reshape(B * n_heads, N_i, N_t)
            .bool()
        )

    @staticmethod
    def get_upsampler(
        patch_size: int,
        num_upsampler_layers: int,
        projection_dim: int,
        final_image_size: int,
        num_output_channels: int,
        norm: StrOrModule | None,
        group_num_channels: int,
        activation: nn.Module,
    ) -> nn.Sequential:
        """Gets the upsampler block to reduce the hidden channels to `num_output_channels` and increase the spatial dimension of output.

        Args:
        ----
            patch_size: The size of the patch. This is the total scale factor for the transformer output.
            num_upsampler_layers: The number of upsample and conv2d blocks to upsample.
            projection_dim: The hidden size of the image encoder.
            final_image_size: The final image size to align the output exactly to the input.
            num_output_channels: The number of channels in he output.

        Returns:
        -------
            Get sequential mode with `num_upsampler_layers` upsampler layers.

        """
        channel_factor = projection_dim // num_upsampler_layers
        up_factor: float = patch_size ** (1 / num_upsampler_layers)

        layers = []
        in_channels = projection_dim

        current_size = final_image_size // patch_size
        for _ in range(num_upsampler_layers - 1):
            out_channels = in_channels - channel_factor
            num_groups = out_channels // group_num_channels

            # Perform the ceiling since previous layers have more information
            current_size = math.ceil(current_size * up_factor)

            layers.append(
                TransDecoder.get_upsample_block(
                    size=current_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm=norm,
                    norm_kwargs=None if norm != "group" else {"num_groups": num_groups},
                    activation=activation,
                ),
            )

            in_channels = out_channels

        return nn.Sequential(
            *layers,
            TransDecoder.get_upsample_block(
                size=final_image_size,
                in_channels=in_channels,
                out_channels=num_output_channels,
            ),
        )

    @staticmethod
    def get_upsample_block(
        in_channels: int,
        out_channels: int,
        size: _size_any_t | None = None,
        scale_factor: _ratio_any_t | None = None,
        up_mode: str = "bilinear",
        kernel_size: _size_2_t = 3,
        padding: _size_2_t | str = "same",
        padding_mode: str = "replicate",
        norm: StrOrModule | None = None,
        norm_kwargs: Mapping[str, object] | None = None,
        activation: nn.Module | None = None,
    ) -> nn.Sequential:
        """Get the upsample and convolve block.

        Args:
        ----
            in_channels: The number of input channels for convolution.
            out_channels: The number of output channels for convolution.
            size: The image out of the output of the block. You must provide this or `scale_factor`
            scale_factor: The factor to scale the image. You must provide this or `size`
            up_mode: The mode to upsample the image. one of `nearest`, `linear`, `bilinear`, `bicubic` and `trilinear`.
            Default: 'nearest'
            kernel_size: The kernel size for convolution. Defaults to `3`.
            padding: The padding to use. Defaults to `same`.
            padding_mode: The padding mode to use. One of `zeros`, `reflect`, `replicate` or `circular`. Defaults to `zeros`.

        Returns:
        -------
            Tuple of Upsample and 2D Convolution layers

        """
        if size is None and scale_factor is None:
            msg = "Either the `size` of the output image or the `scale_factor` must be provided."
            raise ValueError(msg)

        layers: list[nn.Module] = [
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=up_mode,
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=norm is None,  # Add bias when norm is not None
                padding_mode=padding_mode,
            ),
        ]

        if norm is not None:
            normalized_shape = out_channels
            if norm == "layer":
                if size is None:
                    msg = "Size must be provided when using layer norm."
                    raise ValueError(msg)

                normalized_shape = [out_channels, size, size]

            layers.append(
                __class__.get_norm_module(
                    norm,
                    num_channels=normalized_shape,
                    **(norm_kwargs or {}),
                ),
            )

        if activation is not None:
            # Make a copy of activation since some activation may be learnable
            new_activation = copy.deepcopy(activation)
            layers.append(new_activation)

        return nn.Sequential(*layers)

    @staticmethod
    def get_norm_module(
        norm: StrOrModule,
        num_channels: nn.modules.normalization._shape_t,
        **decoder_kwargs,
    ) -> nn.Module:
        """Get the norm module from string or nn.Module.
        Needed because some parameters of norm needs to be filled dynamically.

        Args:
        ----
            norm: The norm type or the instance of nn.Module itself.
            num_channels: The number of channels or normalized_shape
            **decoder_kwargs: The extra kwargs to pass to the norm.

        Raises:
        ------
            ValueError: Raised when the num_channels is not an int for GroupNorm, or BatchNorm2d
            NotImplementedError: Raised when norm type is not supported

        Returns:
        -------
            Returns a freshly baked nn.Module instance of the norm.

        """
        if isinstance(norm, nn.Module):
            # Make a copy of norm since some norm may be learnable
            return copy.deepcopy(norm)

        if norm == "layer":
            return nn.LayerNorm(num_channels, **decoder_kwargs)

        if not isinstance(num_channels, int):
            msg = (
                f"Please provide num_channels as an int for norm: {norm}. "
                f"Got of type {type(num_channels)}"
            )
            raise ValueError(msg)

        if norm == "instance":
            return nn.InstanceNorm2d(num_channels, **decoder_kwargs)

        if norm == "batch":
            return nn.BatchNorm2d(num_channels, **decoder_kwargs)

        if norm == "group":
            return nn.GroupNorm(num_channels, **decoder_kwargs)

        msg = f"Norm type {norm} not implemented. Please pass as a module itself"
        raise NotImplementedError(msg)
