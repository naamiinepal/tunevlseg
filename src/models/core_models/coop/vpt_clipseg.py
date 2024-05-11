from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clipseg.modeling_clipseg import (
    CLIPSegDecoderOutput,
    CLIPSegImageSegmentationOutput,
)

from src.models.components.hf_clipseg_wrapper import HFCLIPSegWrapper

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .context_learner import VPTContextLearner


class VPTCLIPSeg(HFCLIPSegWrapper):
    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        context_learner: type[VPTContextLearner],
        freeze_all: bool = True,
        no_freeze_last_layer: bool = False,
        use_new_last_layer: bool = False,
    ) -> None:
        super().__init__(**model_cfg)

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

        self.context_learner = context_learner(
            max_network_depth=min(
                self.model.config.text_config.num_hidden_layers,
                self.model.config.vision_config.num_hidden_layers,
            ),
            context_dim=self.model.config.text_config.hidden_size,
        )

    def vision_transformer_model_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        r"""Args:
        ----
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        _self = self.model.clip.vision_model.encoder
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else _self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else _self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        encoder_states = []
        all_attentions = []

        hidden_states = inputs_embeds

        max_layer_idx = max(self.model.extract_layers)

        for idx, encoder_layer in enumerate(_self.layers, 1):
            if output_hidden_states:
                encoder_states.append(hidden_states)

            layer_outputs = (
                _self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
                if _self.gradient_checkpointing and _self.training
                else encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
            )

            hidden_states = layer_outputs[0]

            if idx < self.context_learner.prompt_depth:
                # Change the last tokens, where they were concatenated
                self.context_learner.mutate_image_hidden_states(
                    hidden_states, index=idx
                )

            if output_attentions:
                all_attentions.append(layer_outputs[1])

            # No need to run the vision transformer for more layers
            if idx > max_layer_idx:
                break

        encoder_states = (
            (*encoder_states, hidden_states) if output_hidden_states else None
        )

        all_attentions = tuple(all_attentions) if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    def vision_model_forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        _self = self.model.clip.vision_model

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else _self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else _self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        # shape: (batch_size, sequence_length, hidden_size)
        hidden_states = _self.embeddings(pixel_values)

        # Concat the context vector with the hidden states
        hidden_states = self.context_learner(input_embeddings=hidden_states)

        # Concat before the layernorm, concating after results in unstable training
        hidden_states = _self.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_transformer_model_forward(
            inputs_embeds=hidden_states,  # type:ignore
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last_hidden_state = encoder_outputs[0]
        # pooled_output = last_hidden_state[:, 0, :]
        # pooled_output = _self.post_layernorm(pooled_output)

        if not return_dict:
            return encoder_outputs

        return BaseModelOutput(
            hidden_states=encoder_outputs.hidden_states,  # type:ignore
            attentions=encoder_outputs.attentions,  # type:ignore
        )

    def get_vision_outputs(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None,
        output_hidden_states: bool | None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        BaseModelOutput,
    ]:
        _self = self.model

        vision_outputs: BaseModelOutput = self.vision_model_forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )  # type:ignore
        # pooled_output = _self.clip.visual_projection(vision_outputs[1])

        hidden_states = vision_outputs.hidden_states

        if hidden_states is None:
            msg = "You have to specify output_hidden_states=True if you want to use `CLIPSegForImageSegmentation`"
            raise ValueError(msg)

        # we add +1 here as the hidden states also include the initial embeddings
        activations = tuple(hidden_states[i + 1] for i in _self.extract_layers)

        vision_outputs = BaseModelOutput(
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=vision_outputs.attentions,
        )
        return activations, vision_outputs

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

        output = output[
            :, 1 : -self.context_learner.num_context, :
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
    ) -> CLIPSegImageSegmentationOutput:
        _self = self.model

        return_dict = (
            return_dict if return_dict is not None else _self.config.use_return_dict
        )

        if pixel_values is None:
            msg = (
                "You have to specify pixel_values to use `CLIPSegForImageSegmentation`"
            )
            raise ValueError(msg)

        # step 1: compute conditional embeddings, either from text, images or an own provided embedding
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0],
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                conditional_pixel_values=conditional_pixel_values,
            )
        else:
            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Make sure to pass as many conditional embeddings as there are query images in the batch"
                )
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError(
                    "Make sure that the feature dimension of the conditional embeddings matches"
                    " `config.projection_dim`."
                )

        # step 2: forward the query images through the frozen CLIP vision encoder
        activations, vision_outputs = self.get_vision_outputs(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder_forward(
            activations,
            conditional_embeddings,  # type:ignore
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = decoder_outputs[0]

        loss = (
            None
            if labels is None
            else F.binary_cross_entropy_with_logits(logits, labels)
        )

        return CLIPSegImageSegmentationOutput(
            loss=loss,  # type:ignore
            logits=logits,
            conditional_embeddings=conditional_embeddings,  # type:ignore
            vision_model_output=vision_outputs,  # type:ignore
            decoder_output=decoder_outputs,  # type:ignore
        )

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        B, _, H, W = image_input.shape

        # The output is squeezed so it may be (H, W) or (B, H, W)
        outputs = self.model_forward(**text_input, pixel_values=image_input)  # type:ignore

        return outputs.logits.view(B, 1, H, W)
