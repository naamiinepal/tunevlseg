# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmseg.registry import MODELS

from ..utils import UpConvBlock, Upsample


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs=2,
        stride=1,
        dilation=1,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        dcn=None,
        plugins=None,
    ):
        if act_cfg is None:
            act_cfg = {"type": "ReLU"}
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__()
        assert dcn is None, "Not implemented yet."
        assert plugins is None, "Not implemented yet."

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


@MODELS.register_module()
class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        with_cp=False,
        norm_cfg=None,
        act_cfg=None,
        *,
        kernel_size=4,
        scale_factor=2,
    ):
        if act_cfg is None:
            act_cfg = {"type": "ReLU"}
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__()

        assert (kernel_size - scale_factor >= 0) and (
            kernel_size - scale_factor
        ) % 2 == 0, (
            f"kernel_size should be greater than or equal to scale_factor "
            f"and (kernel_size - scale_factor) should be even numbers, "
            f"while the kernel size is {kernel_size} and scale_factor is "
            f"{scale_factor}."
        )

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        _norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


@MODELS.register_module()
class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        with_cp=False,
        norm_cfg=None,
        act_cfg=None,
        *,
        conv_cfg=None,
        conv_first=False,
        kernel_size=1,
        stride=1,
        padding=0,
        upsample_cfg=None,
    ):
        if upsample_cfg is None:
            upsample_cfg = {
                "scale_factor": 2,
                "mode": "bilinear",
                "align_corners": False,
            }
        if act_cfg is None:
            act_cfg = {"type": "ReLU"}
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__()

        self.with_cp = with_cp
        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        upsample = Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


@MODELS.register_module()
class UNet(BaseModule):
    """UNet backbone.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=None,
        norm_eval=False,
        dcn=None,
        plugins=None,
        pretrained=None,
        init_cfg=None,
    ):
        if upsample_cfg is None:
            upsample_cfg = {"type": "InterpConv"}
        if act_cfg is None:
            act_cfg = {"type": "ReLU"}
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is a deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = {"type": "Pretrained", "checkpoint": pretrained}
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    {"type": "Kaiming", "layer": "Conv2d"},
                    {
                        "type": "Constant",
                        "val": 1,
                        "layer": ["_BatchNorm", "GroupNorm"],
                    },
                ]
        else:
            raise TypeError("pretrained must be a str or None")

        assert dcn is None, "Not implemented yet."
        assert plugins is None, "Not implemented yet."
        assert len(strides) == num_stages, (
            "The length of strides should be equal to num_stages, "
            f"while the strides is {strides}, the length of "
            f"strides is {len(strides)}, and the num_stages is "
            f"{num_stages}."
        )
        assert len(enc_num_convs) == num_stages, (
            "The length of enc_num_convs should be equal to num_stages, "
            f"while the enc_num_convs is {enc_num_convs}, the length of "
            f"enc_num_convs is {len(enc_num_convs)}, and the num_stages is "
            f"{num_stages}."
        )
        assert len(dec_num_convs) == (num_stages - 1), (
            "The length of dec_num_convs should be equal to (num_stages-1), "
            f"while the dec_num_convs is {dec_num_convs}, the length of "
            f"dec_num_convs is {len(dec_num_convs)}, and the num_stages is "
            f"{num_stages}."
        )
        assert len(downsamples) == (num_stages - 1), (
            "The length of downsamples should be equal to (num_stages-1), "
            f"while the downsamples is {downsamples}, the length of "
            f"downsamples is {len(downsamples)}, and the num_stages is "
            f"{num_stages}."
        )
        assert len(enc_dilations) == num_stages, (
            "The length of enc_dilations should be equal to num_stages, "
            f"while the enc_dilations is {enc_dilations}, the length of "
            f"enc_dilations is {len(enc_dilations)}, and the num_stages is "
            f"{num_stages}."
        )
        assert len(dec_dilations) == (num_stages - 1), (
            "The length of dec_dilations should be equal to (num_stages-1), "
            f"while the dec_dilations is {dec_dilations}, the length of "
            f"dec_dilations is {len(dec_dilations)}, and the num_stages is "
            f"{num_stages}."
        )
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = strides[i] != 1 or downsamples[i - 1]
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2 ** (i - 1),
                        out_channels=base_channels * 2 ** (i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None,
                    )
                )

            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None,
                )
            )
            self.encoder.append(nn.Sequential(*enc_conv_block))
            in_channels = base_channels * 2**i

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) and (w % whole_downsample_rate == 0), (
            f"The input image size {(h, w)} should be divisible by the whole "
            f"downsample rate {whole_downsample_rate}, when num_stages is "
            f"{self.num_stages}, strides is {self.strides}, and downsamples "
            f"is {self.downsamples}."
        )
