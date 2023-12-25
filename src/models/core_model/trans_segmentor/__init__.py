# pyright: reportGeneralTypeIssues=false
from pathlib import Path
from typing import Any, Mapping, Union, Optional

from torch import nn
from transformers import CLIPVisionModel

from .encoder import TransTextEncoder
from .decoder import TransDecoder

StrOrPath = Union[str, Path]
StrToAny = Mapping[str, Any]


class TransformerSegmentor(nn.Module):
    def __init__(
        self,
        image_pretrained_model_name_or_path: StrOrPath,
        text_pretrained_model_name_or_path: StrOrPath,
        freeze_image_encoder: bool,
        freeze_text_encoder: bool,
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
            image_pretrained_model_name_or_path
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

    def forward(self, text_input: StrToAny, image_input: StrToAny):
        # shape: (B, N_t, H_i)
        text_proj_output = self.text_encoder(**text_input)

        # image object with pooled output and hidden states
        image_output_obj = self.image_encoder(**image_input)

        # shape: (B, N_i, H_i)
        image_last_hidden_state = image_output_obj.last_hidden_state

        # shape: (B, num_output_channels, H, W)
        decoder_output = self.decoder(
            tgt=image_last_hidden_state, memory=text_proj_output
        )

        return decoder_output
