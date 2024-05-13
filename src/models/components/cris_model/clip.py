from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from torch.nn.common_types import _size_2_t, _size_any_t

    CLIPImageOutputType = tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: _size_2_t = 1) -> None:
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        is_stride_gt_1 = (
            stride > 1 if isinstance(stride, int) else all(s > 1 for s in stride)
        )

        self.avgpool = nn.AvgPool2d(stride) if is_stride_gt_1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = None
        self.stride = stride

        if is_stride_gt_1 or inplanes != planes * self.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    (
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ),
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return torch.relu(out + identity)


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.spacial_dim = spacial_dim
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5,
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        # residual
        self.connect = nn.Sequential(
            nn.Conv2d(embed_dim, output_dim, 1, stride=1, bias=False),
            nn.BatchNorm2d(output_dim),
        )

    def resize_pos_embed(
        self,
        pos_embed: torch.Tensor,
        input_shpae: _size_any_t,
    ) -> torch.Tensor:
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.

        Args:
        ----
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, C, L_new].

        """
        if pos_embed.ndim != 3:
            msg = "shape of pos_embed must be [B, L, C]"
            raise ValueError(msg)

        pos_h = pos_w = self.spacial_dim
        # cls_token_weight = pos_embed[:, :1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(
            1,
            pos_h,
            pos_w,
            pos_embed.shape[2],
        ).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=input_shpae,
            align_corners=False,
            mode="bicubic",
        )
        # cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        # pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed_weight.transpose(-2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.connect(x)

        B, C, H, W = x.size()
        x = x.view(B, C, -1)  # NC(HW)
        # x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(1+HW)
        pos_embed = self.positional_embedding.unsqueeze(0)
        pos_embed = self.resize_pos_embed(pos_embed, (H, W))  # NC(HW)
        x = x + pos_embed  # NC(HW)
        x = x.permute(2, 0, 1)  # (HW)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        x = x.permute(1, 2, 0).reshape(B, -1, H, W)
        return torch.relu(x + res)


class ModifiedResNet(nn.Module):
    """A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool.
    """

    def __init__(
        self,
        layers: Sequence[int],
        output_dim: int,
        heads: int,
        input_resolution: int = 224,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3,
            width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2,
            width // 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32,
            embed_dim,
            heads,
            output_dim,
        )

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: _size_2_t = 1,
    ) -> nn.Sequential:
        first_layer = Bottleneck(self._inplanes, planes, stride)

        self._inplanes = planes * Bottleneck.expansion
        rest_layers = (Bottleneck(self._inplanes, planes) for _ in range(1, blocks))

        return nn.Sequential(first_layer, *rest_layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def stem(x: torch.Tensor):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = torch.relu(bn(conv(x)))
            return self.avgpool(x)

        x = stem(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x4 = self.attnpool(x4)

        return (x2, x3, x4)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_type = input.dtype
        ret = super().forward(input.float())
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ],
            ),
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.attn(x, x, x, *args, need_weights=False, **kwargs)[0]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), *args, **kwargs)
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        seq_length = x.size(0)
        attn_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.resblocks:
            x = block(x, *args, **kwargs, attn_mask=attn_mask)

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width),
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.view(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            (
                self.class_embedding.expand_as(x),
                x,
            ),
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = self.ln_pre(x + self.positional_embedding)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x[:, 1:, :])

        if self.proj is None:
            return x

        return x @ self.proj


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Sequence[int] | int,
        vision_width: int,
        vision_patch_size: int | None,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, int):
            vision_heads = vision_width // 64

            if vision_patch_size is None:
                msg = "Please set `vision_patch_size` to initialize `VisionTransformer`"
                raise ValueError(msg)

            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )
        else:
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width),
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(-torch.log(torch.tensor(0.07)))

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        attn_std = self.transformer.width**-0.5
        proj_std = attn_std * ((2 * self.transformer.layers) ** -0.5)
        fc_std = 2**-0.5 * attn_std
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=attn_std)

    def encode_image(self, image: torch.Tensor) -> CLIPImageOutputType:
        return self.visual(image)

    def encode_text(
        self,
        text: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, *args, **kwargs)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = x @ self.text_projection
        # state = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return x, state

    # def forward(
    #     self,
    #     image: torch.Tensor,
    #     text: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     image_features = self.encode_image(image)
    #     text_features = self.encode_text(text)
    #
    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #
    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()
    #
    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text
    #


def convert_weights(model: nn.Module) -> None:
    """Convert applicable model parameters to fp16."""

    def _convert_weights_to_fp16(layer: nn.Module) -> None:
        if isinstance(layer, nn.Conv1d | nn.Conv2d | nn.Linear):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            attr = getattr(layer, name, None)
            if attr is not None:
                attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: MutableMapping[str, torch.Tensor]) -> CLIP:
    if "visual.proj" in state_dict:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ],
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5,
        )
        image_resolution = vision_patch_size * grid_size
    else:
        vision_layers = tuple(
            len(
                {
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                },
            )
            for b in range(1, 5)
        )
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5,
        )
        vision_patch_size = None

        if (
            output_width**2 + 1
            != state_dict["visual.attnpool.positional_embedding"].shape[0]
        ):
            msg = "Wrong vision grid size or output width!"
            raise ValueError(msg)

        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        {k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")},
    )

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ("input_resolution", "context_length", "vocab_size"):
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, False)
    return model.eval()
