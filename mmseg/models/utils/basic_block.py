# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType


class BasicBlock(BaseModule):
    """Basic block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at the
            last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        act_cfg_out: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        if act_cfg_out is None:
            act_cfg_out = {"type": "ReLU", "inplace": True}
        if act_cfg is None:
            act_cfg = {"type": "ReLU", "inplace": True}
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.downsample = downsample
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, "act"):
            out = self.act(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at
            the last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 2

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        act_cfg_out: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        if act_cfg is None:
            act_cfg = {"type": "ReLU", "inplace": True}
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.conv2 = ConvModule(
            channels, channels, 3, stride, 1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.conv3 = ConvModule(
            channels, channels * self.expansion, 1, norm_cfg=norm_cfg, act_cfg=None
        )
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, "act"):
            out = self.act(out)

        return out
