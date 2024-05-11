from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.models.clipseg.modeling_clipseg import (
    CLIPSegDecoderOutput,
    CLIPSegImageSegmentationOutput,
)

from src.models.components.hf_clipseg_wrapper import HFCLIPSegWrapper

from .context_learner import BaseUnimodalLearner, BaseVisualLearner

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class BaseCLIPSeg(HFCLIPSegWrapper, ABC):
    context_learner: BaseUnimodalLearner

    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        freeze_all: bool = True,
        no_freeze_last_layer: bool = False,
        use_new_last_layer: bool = False,
    ) -> None:
        super().__init__(**model_cfg)

        self.assign_model_learnability(
            freeze_all, no_freeze_last_layer, use_new_last_layer
        )

    def assign_model_learnability(
        self, freeze_all: bool, no_freeze_last_layer: bool, use_new_last_layer: bool
    ):
        if freeze_all:
            self.eval()
            self.requires_grad_(False)

        if use_new_last_layer:
            self.additive_decoder_layer = nn.Sequential(
                nn.Upsample(
                    scale_factor=float(self.model.config.vision_config.patch_size),
                    mode="bilinear",
                ),
                nn.Conv2d(
                    self.model.config.reduce_dim,
                    1,
                    kernel_size=5,
                    padding="same",
                    padding_mode="replicate",
                ),
            )
        elif no_freeze_last_layer:
            self.additive_decoder_layer = None

            trans_conv = self.model.decoder.transposed_convolution
            last_layer = (
                trans_conv[-1]
                if isinstance(trans_conv, torch.nn.Sequential)
                else trans_conv
            )
            last_layer.requires_grad_(True)

    def decoder_forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
    ):
        _self = self.model.decoder

        activations = hidden_states[::-1]

        first_act = activations[0]

        output = torch.zeros(
            self.model.config.reduce_dim,
            dtype=first_act.dtype,
            device=first_act.device,
        )

        all_hidden_states = []
        all_attentions = []

        for i, (activation, layer, reduce) in enumerate(
            zip(activations, _self.layers, _self.reduces, strict=True)
        ):
            # Cannot use += here due to broadcasting
            output = reduce(activation) + output

            if i == _self.conditional_layer:
                output = _self.film_mul(conditional_embeddings) * output.permute(
                    1, 0, 2
                ) + _self.film_add(conditional_embeddings)
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=output_attentions,
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states.append(output)

            if output_attentions:
                all_attentions.append(layer_outputs[1])

        last_seq_index = (
            -self.context_learner.num_context
            if isinstance(self.context_learner, BaseVisualLearner)
            else None
        )

        output = output[
            :, 1:last_seq_index, :
        ].permute(
            0, 2, 1
        )  # remove cls token and prompts and reshape to [batch_size, reduce_dim, seq_len]

        B, H, N = output.shape

        size = math.isqrt(N)

        output = output.view(B, H, size, size)

        logits = _self.transposed_convolution(output)

        if self.additive_decoder_layer is not None:
            logits += self.additive_decoder_layer(output)

        logits = logits.squeeze()

        all_hidden_states = tuple(all_hidden_states) if output_hidden_states else None

        all_attentions = tuple(all_attentions) if output_attentions else None

        if not return_dict:
            return tuple(
                v for v in [logits, all_hidden_states, all_attentions] if v is not None
            )

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    @abstractmethod
    def model_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        conditional_pixel_values: torch.FloatTensor | None = None,
        conditional_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CLIPSegImageSegmentationOutput: ...

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        B, _, H, W = image_input.shape

        # The output is squeezed so it may be (H, W) or (B, H, W)
        outputs = self.model_forward(**text_input, pixel_values=image_input)  # type:ignore

        return outputs.logits.view(B, 1, H, W)
