import math
from typing import Any

import open_clip
import torch
from timm.models import VisionTransformer, checkpoint_seq
from torch import nn

from .utils import get_mask_mixed_embed


class CustomOpenCLIP(nn.Module):
    def __init__(self, clip_pretrained_path) -> None:
        super().__init__()

        self.model: open_clip.CustomTextCLIP = open_clip.create_model(
            clip_pretrained_path,
        )  # type:ignore

        visual = self.model.visual  # type:ignore
        image_size = visual.image_size
        self.image_size: int = (
            image_size if isinstance(image_size, int) else image_size[0]
        )  # type:ignore

        patch_size = visual.trunk.patch_embed.patch_size
        self.patch_size: int = (
            patch_size if isinstance(patch_size, int) else patch_size[0]
        )  # type:ignore

    def get_blocks_output(
        self,
        x: torch.Tensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
    ) -> torch.Tensor:
        blocks: nn.Sequential = self.model.visual.trunk.blocks
        if pred_masks is None:
            return blocks(x)

        patch_len = x.size(1) - 1
        size = math.isqrt(patch_len)

        if size * size != patch_len:
            msg = "The number of patches in the image must be a perfect square."
            raise ValueError(msg)

        non_masked_blocks = blocks[:masking_block_idx]  # type:ignore

        x = non_masked_blocks(x)

        masked_blocks = blocks[masking_block_idx:]  # type:ignore

        # mask_shape: (B, size, size) -> (B, 1, size, size)
        pred_masks = pred_masks.unsqueeze(1)
        num_masks = pred_masks.size(0)

        for block in masked_blocks:
            x = get_mask_mixed_embed(x, num_masks, patch_len, pred_masks, size)
            x = block(x)

        return x

    def get_forward_features(
        self,
        x: torch.Tensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
    ) -> torch.Tensor:
        trunk: VisionTransformer = self.model.visual.trunk
        x = trunk.patch_embed(x)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)
        x = trunk.norm_pre(x)
        if trunk.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.get_blocks_output, x, pred_masks, masking_block_idx)
        else:
            x = self.get_blocks_output(x, pred_masks, masking_block_idx)
        return trunk.norm(x)

    def get_trunk_output(
        self,
        x: torch.Tensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
    ) -> torch.Tensor:
        trunk: VisionTransformer = self.model.visual.trunk
        x = self.get_forward_features(x, pred_masks, masking_block_idx)
        return trunk.forward_head(x)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
    ) -> torch.Tensor:
        x = self.get_trunk_output(pixel_values, pred_masks, masking_block_idx)
        return self.model.visual.head(x)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Any = None,
    ) -> torch.Tensor:
        return self.model.text(input_ids)
