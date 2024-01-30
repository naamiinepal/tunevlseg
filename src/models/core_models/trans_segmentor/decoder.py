from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    IntorIntTuple = tuple[int, int] | int
    StrToAny = Mapping[str, Any]


class TransDecoder(nn.Module):
    def __init__(
        self,
        image_hidden_size: int,
        decoder_layer_kwargs: StrToAny,
        num_decoder_layers: int,
        patch_size: float,
        num_upsampler_layers: int,
        final_image_size: int,
        num_output_channels: int = 1,
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

        """
        super().__init__(*args, **kwargs)

        self.decoder_layer_kwargs = decoder_layer_kwargs

        self.transformer_decoder = self.get_trans_decoder(
            image_hidden_size,
            decoder_layer_kwargs,
            num_decoder_layers,
        )

        self.upsampler = self.get_upsampler(
            patch_size=patch_size,
            num_upsampler_layers=num_upsampler_layers,
            image_hidden_size=image_hidden_size,
            final_image_size=final_image_size,
            num_output_channels=num_output_channels,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory_mask: torch.Tensor | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if memory_mask is not None:
            memory_mask = self.get_memory_mask(memory_mask, tgt.size(1))

        # shape: (B, N_i + 1, H_i)
        trans_output: torch.Tensor = self.transformer_decoder(
            *args,
            tgt=tgt,
            memory_mask=memory_mask,
            **kwargs,
        )

        # Remove the pooled output
        # shape: (B, N_i, H_i)
        image_output = trans_output[:, 1:, :]

        # Move the hidden dimension to make it channel first
        # shape: (B, H_i, N_i)
        channel_first_output = image_output.movedim(-1, 1)

        B, H, flat_size = channel_first_output.shape

        hid_img_size = math.isqrt(flat_size)

        if (hid_img_size * hid_img_size) != flat_size:
            msg = "The number of tokens besides the pooled output should be a perfect square."
            raise ValueError(
                msg,
            )

        # Make the output 4D
        # shape: (B, H_i, sqrt(N_i), sqrt(N_i))
        img_channel_first = channel_first_output.view(B, H, hid_img_size, hid_img_size)

        return self.upsampler(img_channel_first)

    def get_memory_mask(self, memory_mask: torch.Tensor, N_i: int) -> torch.Tensor:
        B, N_t = memory_mask.shape
        n_head = self.decoder_layer_kwargs["nhead"]

        # needed shape: (B * n_head, N_i, N_t)
        return (
            memory_mask.expand(n_head * N_i, B, N_t)
            .moveaxis(1, 0)
            .reshape(B * n_head, N_i, N_t)
            .bool()
        )

    @staticmethod
    def get_trans_decoder(
        image_hidden_size: int,
        decoder_layer_kwargs: StrToAny,
        num_decoder_layers: int,
    ) -> nn.TransformerDecoder:
        """Get the transformer decoder.

        Args:
        ----
            image_hidden_size: The hidden size of the image encoder.
            decoder_layer_kwargs: The keyword arguments to the transformer decoder. `n_head` is the required one.
            num_decoder_layers: The number of transformer decoder layers.

        Returns:
        -------
            The transformer decoder block with `num_decoder_layers` transformer decoder layers.

        """
        decoder_layer = nn.TransformerDecoderLayer(
            image_hidden_size,
            **decoder_layer_kwargs,
            batch_first=True,
        )
        norm = nn.LayerNorm(image_hidden_size)

        return nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=norm,
        )

    @staticmethod
    def get_upsampler(
        patch_size: float,
        num_upsampler_layers: int,
        image_hidden_size: int,
        final_image_size: int,
        num_output_channels: int,
    ) -> nn.Sequential:
        """Gets the upsampler block to reduce the hidden channels to `num_output_channels` and increase the spatial dimension of output.

        Args:
        ----
            patch_size: The size of the patch. This is the total scale factor for the transformer output.
            num_upsampler_layers: The number of upsample and conv2d blocks to upsample.
            image_hidden_size: The hidden size of the image encoder.
            final_image_size: The final image size to align the output exactly to the input.
            num_output_channels: The number of channels in he output.
                This defaults to `1`  for the binary segmentation task.

        Returns:
        -------
            Get sequential mode with `num_upsampler_layers` upsampler layers.

        """
        up_factor = patch_size ** (1 / num_upsampler_layers)

        channel_factor = image_hidden_size // num_upsampler_layers

        layers = []
        in_channels = image_hidden_size

        for _ in range(num_upsampler_layers - 1):
            out_channels = in_channels - channel_factor
            layers.extend(
                (
                    TransDecoder.get_upsample_block(
                        scale_factor=up_factor,
                        in_channels=in_channels,
                        out_channels=out_channels,
                    ),
                    nn.ReLU(inplace=True),
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
        size: int | None = None,
        scale_factor: float | None = None,
        up_mode: str = "bilinear",
        kernel_size: IntorIntTuple = 3,
        padding: str | IntorIntTuple = "same",
        padding_mode: str = "replicate",
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

        return nn.Sequential(
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
                padding_mode=padding_mode,
            ),
        )
