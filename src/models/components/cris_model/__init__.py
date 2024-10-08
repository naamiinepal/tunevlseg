from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .clip import CLIP, build_model
from .layers import FPN, Projector, TransformerDecoder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch.serialization import FILE_LIKE

    from .clip import CLIPImageOutputType


class CRIS(nn.Module):
    max_length = 77

    def __init__(
        self,
        clip_pretrain: FILE_LIKE,
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
        *args,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.img_size = img_size

        self.backbone = self.get_backbone(clip_pretrain)
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
            print("Loading CRIS pre-trained model from:", cris_pretrain)
            self.load_state_dict(
                torch.load(cris_pretrain, map_location="cpu"),
                strict=True,
            )

        self.word_dim = word_dim

    @staticmethod
    def get_backbone(clip_pretrain: FILE_LIKE) -> CLIP:
        # Vision & Text Encoder
        clip_model = torch.jit.load(clip_pretrain, map_location="cpu")
        return build_model(clip_model.state_dict()).float()

    def get_pad_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return (
            ~(attention_mask.bool()) if attention_mask is not None else input_ids == 0
        )

    def get_unimodal_outputs(
        self,
        image_input: torch.Tensor,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[CLIPImageOutputType, torch.Tensor, torch.Tensor]:
        # vis: C3 / C4 / C5
        # input_ids: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(image_input)
        input_ids, state = self.backbone.encode_text(input_ids, *args, **kwargs)
        return vis, input_ids, state

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        """img: b, 3, h, w
        word: b, words
        word_mask: b, words
        mask: b, 1, h, w.
        """
        input_ids = text_input["input_ids"]

        attention_mask = text_input.get("attention_mask")

        # padding mask used in decoder
        pad_mask = self.get_pad_mask(input_ids, attention_mask)

        vis, input_ids, state = self.get_unimodal_outputs(
            image_input, input_ids, key_padding_mask=pad_mask
        )

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, input_ids, pad_mask)
        fq = fq.view(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        return F.interpolate(pred, self.img_size, mode="bicubic", align_corners=True)
