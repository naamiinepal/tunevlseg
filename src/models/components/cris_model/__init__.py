from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .clip import CLIP, build_model
from .layers import FPN, Projector, TransformerDecoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.serialization import FILE_LIKE


class CRIS(nn.Module):
    def __init__(
        self,
        clip_pretrain: FILE_LIKE,
        word_len: int,
        fpn_in: Sequence[int],
        fpn_out: Sequence[int],
        vis_dim: int,
        word_dim: int,
        num_layers: int,
        num_head: int,
        dim_ffn: int,
        dropout: float,
        return_intermediate: bool,
        img_size: int = 416,
        freeze_encoder: bool = True,
        cris_pretrain: FILE_LIKE | None = None,
    ) -> None:
        super().__init__()

        self.img_size = img_size

        self.backbone = self.get_backbone(clip_pretrain, word_len)
        self.backbone.requires_grad_(not freeze_encoder)

        # Multi-Modal FPN
        self.neck = FPN(in_channels=fpn_in, out_channels=fpn_out)

        # Decoder
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=vis_dim,
            nhead=num_head,
            dim_ffn=dim_ffn,
            dropout=dropout,
            return_intermediate=return_intermediate,
        )

        # Projector
        self.proj = Projector(word_dim, vis_dim // 2, 3)

        if cris_pretrain is not None:
            self.load_state_dict(
                torch.load(cris_pretrain, map_location="cpu"),
                strict=True,
            )

    @staticmethod
    def get_backbone(clip_pretrain: FILE_LIKE, word_len: int) -> CLIP:
        # Vision & Text Encoder
        clip_model = torch.jit.load(clip_pretrain, map_location="cpu")
        return build_model(clip_model.state_dict(), word_len).float()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """img: b, 3, h, w
        word: b, words
        word_mask: b, words
        mask: b, 1, h, w.
        """
        # padding mask used in decoder
        pad_mask = (
            ~(attention_mask.bool()) if attention_mask is not None else input_ids == 0
        )

        # vis: C3 / C4 / C5
        # input_ids: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(pixel_values)
        input_ids, state = self.backbone.encode_text(input_ids)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, input_ids, pad_mask)
        fq = fq.view(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        return F.interpolate(pred, self.img_size, mode="bicubic", align_corners=True)