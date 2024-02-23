# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

# -------------------------------------------------------------------------
# Copyright (c) 2019 the AdelaiDet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modified by Xinlong Wang
# -------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..detectron2.layers import ShapeSpec


class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg: Any, input_shape: Iterable[ShapeSpec]) -> None:
        """SOLOv2 Instance Head."""
        super().__init__()
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS
        self.num_embs = 128
        #
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.SOLOV2.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), print(
            "Strides should match the features.",
        )

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, print(
            "Each level must have the same channel!",
        )
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS, print(
            "In channels should equal to tower in channels!",
        )

        self.add_tower_heads(cfg)

        self.cate_pred = nn.Conv2d(
            self.instance_channels,
            self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels,
            self.num_kernels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.emb_pred = nn.Conv2d(
            self.instance_channels,
            self.num_embs,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if cfg.MODEL.SOLOV2.FREEZE:
            # for modules in [self.cate_tower, self.kernel_tower, self.kernel_pred]:
            for modules in [self.cate_tower, self.kernel_tower]:
                for p in modules.parameters():
                    p.requires_grad = False
            print("froze ins head parameters")

        for modules in [
            self.cate_tower,
            self.kernel_tower,
            self.cate_pred,
            self.kernel_pred,
            self.emb_pred,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.SOLOV2.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def add_tower_heads(self, cfg: Any) -> None:
        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM

        head_configs = {
            "cate": (
                cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                False,
            ),
            "kernel": (
                cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                cfg.MODEL.SOLOV2.USE_COORD_CONV,
            ),
        }
        for head, (num_convs, _use_deformable, use_coord) in head_configs.items():
            tower = []
            for i in range(num_convs):
                conv_func = nn.Conv2d

                chn = self.instance_in_channels

                if i == 0 and use_coord:
                    chn += 2

                tower.append(
                    conv_func(
                        chn,
                        self.instance_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=norm is None,
                    ),
                )
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module(f"{head}_tower", nn.Sequential(*tower))

    def forward(self, features: Iterable[torch.Tensor]):
        """Arguments:
        ---------
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns
        -------
            pass.

        """
        cate_pred = []
        kernel_pred = []
        emb_pred = []

        for idx, ins_kernel_feat in enumerate(features):
            # concat coord
            x_range = torch.linspace(
                -1,
                1,
                ins_kernel_feat.shape[-1],
                device=ins_kernel_feat.device,
            )
            y_range = torch.linspace(
                -1,
                1,
                ins_kernel_feat.shape[-2],
                device=ins_kernel_feat.device,
            )
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode="bilinear")
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))

            # emb
            emb_pred.append(self.emb_pred(cate_feat))
        return cate_pred, kernel_pred, emb_pred
