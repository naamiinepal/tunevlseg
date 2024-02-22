import math

import torch
from torch import nn
from transformers import CLIPModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from .utils import (
    clip_encoder_common_outputs,
    get_mask_mixed_embed,
)


class CustomHFCLIP(nn.Module):
    def __init__(self, clip_pretrained_path) -> None:
        super().__init__()

        self.model: CLIPModel = CLIPModel.from_pretrained(clip_pretrained_path)  # type:ignore

        vision_config = self.model.config.vision_config  # type:ignore

        self.image_size: int = vision_config.image_size
        self.patch_size: int = vision_config.patch_size

    def get_vision_encoder_output(
        self,
        inputs_embeds: torch.Tensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        model = self.model.vision_model.encoder

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else model.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else model.config.use_return_dict
        )

        encoder_states = ()
        all_attentions = ()

        if pred_masks is None:
            hidden_states = inputs_embeds
            for encoder_layer in model.layers:
                (
                    all_attentions,
                    encoder_states,
                    hidden_states,
                ) = clip_encoder_common_outputs(
                    all_attentions,
                    attention_mask,
                    causal_attention_mask,
                    encoder_layer,
                    encoder_states,
                    hidden_states,
                    model,
                    output_attentions,
                    output_hidden_states,
                )

        else:
            patch_len = inputs_embeds.size(1) - 1
            size = math.isqrt(patch_len)

            if size * size != patch_len:
                msg = "The number of patches in the image must be a perfect square."
                raise ValueError(msg)

            non_masked_blocks: nn.ModuleList = model.layers[:masking_block_idx]  # type:ignore
            masked_blocks: nn.ModuleList = model.layers[masking_block_idx:]  # type:ignore

            hidden_states = inputs_embeds
            for encoder_layer in non_masked_blocks:
                (
                    all_attentions,
                    encoder_states,
                    hidden_states,
                ) = clip_encoder_common_outputs(
                    all_attentions,
                    attention_mask,
                    causal_attention_mask,
                    encoder_layer,
                    encoder_states,
                    hidden_states,
                    model,
                    output_attentions,
                    output_hidden_states,
                )

            # mask_shape: (B, size, size) -> (B, 1, size, size)
            pred_masks = pred_masks.unsqueeze(1)
            num_masks = pred_masks.size(0)

            for encoder_layer in masked_blocks:
                hidden_states = get_mask_mixed_embed(
                    hidden_states,
                    num_masks,
                    patch_len,
                    pred_masks,
                    size,
                )

                (
                    all_attentions,
                    encoder_states,
                    hidden_states,
                ) = clip_encoder_common_outputs(
                    all_attentions,
                    attention_mask,
                    causal_attention_mask,
                    encoder_layer,
                    encoder_states,
                    hidden_states,
                    model,
                    output_attentions,
                    output_hidden_states,
                )

        encoder_states = (
            (*encoder_states, hidden_states) if output_hidden_states else None
        )

        if not output_attentions:
            all_attentions = None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,  # type:ignore
            hidden_states=encoder_states,  # type:ignore
            attentions=all_attentions,  # type:ignore
        )

    def get_vision_model_output(
        self,
        pixel_values: torch.FloatTensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        model = self.model.vision_model

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else model.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else model.config.use_return_dict
        )

        hidden_states = model.embeddings(pixel_values)
        hidden_states = model.pre_layrnorm(hidden_states)

        encoder_outputs = self.get_vision_encoder_output(
            inputs_embeds=hidden_states,
            pred_masks=pred_masks,
            masking_block_idx=masking_block_idx,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.FloatTensor:
        model = self.model
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else model.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else model.config.use_return_dict
        )

        vision_outputs = self.get_vision_model_output(
            pixel_values=pixel_values,
            pred_masks=pred_masks,
            masking_block_idx=masking_block_idx,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        return model.visual_projection(pooled_output)

    def get_text_features(self, *args, **kwargs) -> torch.FloatTensor:
        return self.model.get_text_features(*args, **kwargs)
