import torch
from torch import nn
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
