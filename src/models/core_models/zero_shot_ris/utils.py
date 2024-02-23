from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers.models.clip.modeling_clip import CLIPEncoder


def get_mask_mixed_embed(
    hidden_states: torch.Tensor,
    num_masks: int,
    patch_len: int,
    pred_masks: torch.Tensor,
    size: int,
):
    # shape: (1, 1, H)
    cls_embed = hidden_states[:, :1, :]

    # shape: (1, N, H)
    patch_embed = hidden_states[:, 1:, :]

    # shape: (1, H, N)
    channel_first_patch_embed = patch_embed.movedim(-1, 1)

    # shape: (1, H, size, size)
    H = channel_first_patch_embed.size(1)
    fourD_channel_first_patch_embed = channel_first_patch_embed.reshape(
        -1,
        H,
        size,
        size,
    )

    # shape: (B, H, size, size)
    mask_mixed_fourD_channel_first_patch_embed = (
        fourD_channel_first_patch_embed * pred_masks
    )

    # shape: (B, H, N)
    mask_mixed_channel_first_patch_embed = (
        mask_mixed_fourD_channel_first_patch_embed.reshape(-1, H, patch_len)
    )

    # shape: (B, N, H)
    mask_mixed_patch_embed = mask_mixed_channel_first_patch_embed.movedim(1, -1)

    # shape: (B, N+1, H)
    return torch.cat(
        (cls_embed.expand(num_masks, -1, -1), mask_mixed_patch_embed),
        dim=1,
    )


def clip_encoder_common_outputs(
    all_attentions: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor | None,
    causal_attention_mask: torch.Tensor | None,
    encoder_layer: nn.Module,
    encoder_states: tuple[torch.Tensor, ...],
    hidden_states: torch.Tensor,
    model: CLIPEncoder,
    output_attentions: bool | None,
    output_hidden_states: bool | None,
):
    if output_hidden_states:
        encoder_states = (*encoder_states, hidden_states)

    args = (hidden_states, attention_mask, causal_attention_mask, output_attentions)
    layer_outputs = (
        model._gradient_checkpointing_func(encoder_layer.__call__, *args)
        if model.gradient_checkpointing and model.training
        else encoder_layer(*args)
    )

    hidden_states = layer_outputs[0]

    if output_attentions:
        all_attentions = (*all_attentions, layer_outputs[1])
    return all_attentions, encoder_states, hidden_states


def get_hf_masked_states(
    inputs_embeds: torch.Tensor,
    pred_masks: torch.Tensor,
    masking_block_idx: int | None,
    model: CLIPEncoder,
    attention_mask: torch.Tensor | None,
    causal_attention_mask: torch.Tensor | None,
    output_attentions: bool | None,
    output_hidden_states: bool | None,
):
    patch_len = inputs_embeds.size(1) - 1
    size = math.isqrt(patch_len)

    if size * size != patch_len:
        msg = "The number of patches in the image must be a perfect square."
        raise ValueError(msg)

    non_masked_blocks: nn.ModuleList = model.layers[:masking_block_idx]  # type:ignore
    masked_blocks: nn.ModuleList = model.layers[masking_block_idx:]  # type:ignore

    all_attentions = ()
    encoder_states = ()
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
    return encoder_states, hidden_states, all_attentions
