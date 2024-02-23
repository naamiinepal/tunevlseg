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

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence, Sized


class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg: Any, input_shape: Sized) -> None:
        """SOLOv2 Mask Head."""
        super().__init__()
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(
            self.mask_in_features,
        ), "Input shape should match the features."
        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM

        self.convs_all_levels = self.get_convs_all_levels(norm)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels,
                self.num_masks,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=norm is None,
            ),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True),
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def get_convs_all_levels(self, norm: str | None) -> nn.ModuleList:
        convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = []
                conv_tower.append(
                    nn.Conv2d(
                        self.mask_in_channels,
                        self.mask_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=norm is None,
                    ),
                )
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module("conv" + str(i), nn.Sequential(*conv_tower))
                convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = []
                    conv_tower.append(
                        nn.Conv2d(
                            chn,
                            self.mask_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=norm is None,
                        ),
                    )
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module(
                        "conv" + str(j),
                        nn.Sequential(*conv_tower),
                    )
                    upsample_tower = nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    )
                    convs_per_level.add_module("upsample" + str(j), upsample_tower)
                    continue
                conv_tower = []
                conv_tower.append(
                    nn.Conv2d(
                        self.mask_channels,
                        self.mask_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=norm is None,
                    ),
                )
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module("conv" + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
                convs_per_level.add_module("upsample" + str(j), upsample_tower)

            convs_all_levels.append(convs_per_level)
        return convs_all_levels

    def forward(self, features: Sequence[torch.Tensor]):
        """Arguments:
        ---------
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns
        -------
            pass.

        """
        assert len(features) == self.num_levels, (
            "The number of input features should be equal to the supposed level.",
        )

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(
                    -1,
                    1,
                    mask_feat.shape[-1],
                    device=mask_feat.device,
                )
                y_range = torch.linspace(
                    -1,
                    1,
                    mask_feat.shape[-2],
                    device=mask_feat.device,
                )
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level += self.convs_all_levels[i](mask_feat)

        return self.conv_pred(feature_add_all_level)
