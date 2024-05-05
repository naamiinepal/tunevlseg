from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.nn.common_types import _size_2_t


def conv_layer(
    in_dim: int,
    out_dim: int,
    kernel_size: _size_2_t = 1,
    padding: _size_2_t | str = 0,
    stride: _size_2_t = 1,
):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True),
    )


def linear_layer(in_dim: int, out_dim: int, bias: bool = False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(True),
    )


class CoordConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        padding: _size_2_t = 1,
        stride: _size_2_t = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv_layer(
            in_channels + 2,
            out_channels,
            kernel_size,
            padding,
            stride,
        )

    def add_coord(self, input_tensor: torch.Tensor):
        input_tensor_like_kwargs = {
            "device": input_tensor.device,
            "dtype": input_tensor.dtype,
        }

        b, _, h, w = input_tensor.size()
        x_range = torch.linspace(-1, 1, w, **input_tensor_like_kwargs)
        y_range = torch.linspace(-1, 1, h, **input_tensor_like_kwargs)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return torch.cat([input_tensor, coord_feat], 1)

    def forward(self, x: torch.Tensor):
        x = self.add_coord(x)
        return self.conv1(x)


class Projector(nn.Module):
    def __init__(
        self,
        word_dim: int = 1024,
        in_dim: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.kernel_size = kernel_size

        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode="bilinear"),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1),
        )

        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x: torch.Tensor, word: torch.Tensor):
        """x: b, 512, 26, 26
        word: b, 512.
        """
        x = self.vis(x)
        B, C, H, W = x.size()

        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)

        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)

        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(
            x,
            weight,
            padding=self.kernel_size // 2,
            groups=weight.size(0),
            bias=bias,
        )
        return out.transpose(0, 1)

        # b, 1, 104, 104


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_ffn: int,
        dropout: float,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ffn,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model: int, length: int, input_tensor: torch.Tensor):
        """:param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            msg = (
                "Cannot use sin/cos positional encoding with "
                f"odd dim (got dim={d_model:d})"
            )
            raise ValueError(msg)

        input_tensor_like_kwargs = {
            "device": input_tensor.device,
            "dtype": input_tensor.dtype,
        }

        # Create a tensor of shape (token_length, d_model)
        pe = torch.zeros(length, d_model, **input_tensor_like_kwargs)

        # Create a tensor of shape (token_length, 1)
        position = torch.arange(length, **input_tensor_like_kwargs).unsqueeze(1)

        # Create a tensor of shape (d_model // 2)
        mul_term = 1e-4 ** (
            torch.arange(0, d_model, 2, **input_tensor_like_kwargs) / d_model
        )

        # Create a tensor of shape (token_length, d_model // 2)
        angles = position * mul_term

        # Add sin and cos to the posenc tensor
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model: int, height: int, width: int, input_tensor: torch.Tensor):
        """:param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            msg = (
                "Cannot use sin/cos positional encoding with "
                f"odd dimension (got dim={d_model:d})"
            )
            raise ValueError(msg)

        input_tensor_like_kwargs = {
            "device": input_tensor.device,
            "dtype": input_tensor.dtype,
        }

        # Create a tensor of shape (d_model, height, width)
        pe = torch.zeros(d_model, height, width, **input_tensor_like_kwargs)

        # Each dimension use half of d_model
        d_model //= 2
        mul_term = 1e-4 ** (
            torch.arange(0, d_model, 2, **input_tensor_like_kwargs) / d_model
        )

        # Create a tensor of shape (width, 1)
        pos_w = torch.arange(width, **input_tensor_like_kwargs).unsqueeze(1)
        angles_w = pos_w * mul_term

        pe[:d_model:2, :, :] = (
            torch.sin(angles_w).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(angles_w).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        )

        # Create a tensor of shape (height, 1)
        pos_h = torch.arange(height, **input_tensor_like_kwargs).unsqueeze(1)
        angles_h = pos_h * mul_term

        pe[d_model::2, :, :] = (
            torch.sin(angles_h).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(angles_h).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis: torch.Tensor, txt: torch.Tensor, pad_mask: torch.Tensor):
        """vis: b, 512, h, w
        txt: b, L, 512
        pad_mask: b, L.
        """
        B, C, H, W = vis.size()
        _, L, D = txt.size()

        # position encoding
        vis_pos = self.pos2d(C, H, W, vis)
        txt_pos = self.pos1d(D, L, txt)

        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)

        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)

            if not self.return_intermediate:
                # b, 512, HW
                return output

            intermediate.pop()
            intermediate.append(output)
            # [output1, output2, ..., output_n]
            return intermediate
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 9,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            kdim=d_model,
            vdim=d_model,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, d_model),
        )

        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        vis: torch.Tensor,
        txt: torch.Tensor,
        vis_pos: torch.Tensor,
        txt_pos: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        """vis: 26*26, b, 512
        txt: L, b, 512
        vis_pos: 26*26, 1, 512
        txt_pos: L, 1, 512
        pad_mask: b, L.
        """
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis += self.dropout1(vis2)

        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(
            query=self.with_pos_embed(vis2, vis_pos),
            key=self.with_pos_embed(txt, txt_pos),
            value=txt,
            key_padding_mask=pad_mask,
        )[0]
        vis2 = self.cross_attn_norm(vis2)
        vis += self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        return vis + self.dropout3(vis2)


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (512, 1024, 1024),
        out_channels: Sequence[int] = (256, 512, 1024),
    ) -> None:
        if len(in_channels) < 3:
            msg = "`FPN` module requires `in_channels` of length 3"
            raise ValueError(msg)

        if len(out_channels) < 3:
            msg = "`FPN` module requires `out_channels` of length 3"
            raise ValueError(msg)

        super().__init__()

        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])

        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]), nn.ReLU(True))

        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(
            out_channels[2] + out_channels[1],
            out_channels[1],
            1,
            0,
        )

        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(
            out_channels[0] + out_channels[1],
            out_channels[1],
            1,
            0,
        )

        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)

        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1),
        )

    def forward(
        self,
        imgs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        state: torch.Tensor,
    ):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs

        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)

        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode="bilinear")
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))

        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))

        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)

        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode="bilinear")
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        return self.coordconv(fq)

        # b, 512, 26, 26
