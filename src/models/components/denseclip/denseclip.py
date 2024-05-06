from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F

from mmengine import ConfigDict
from mmseg.models import builder
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.utils import resize
from mmseg.utils import add_prefix
from mmseg.utils.typing_utils import SampleList

from .untils import tokenize

if TYPE_CHECKING:
    from mmengine.model.base_module import BaseModule
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@builder.SEGMENTORS.register_module()
class DenseCLIP(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self,
        backbone: ConfigDict,
        text_encoder: ConfigDict,
        context_decoder: ConfigDict,
        decode_head: ConfigDict,
        class_names: Iterable[str] | Iterable[Iterable[str]],
        context_length: int,
        context_feature: Literal["attention", "backbone"] = "attention",
        score_concat_index: int = 3,
        text_head: bool = False,
        neck: ConfigDict | None = None,
        tau: float = 0.07,
        auxiliary_head: Iterable[ConfigDict] | ConfigDict | None = None,
        identity_head: ConfigDict | None = None,
        train_cfg: ConfigDict | None = None,
        test_cfg: ConfigDict | None = None,
        pretrained: str | None = None,
        init_cfg: ConfigDict | None = None,
        token_embed_dim: int = 512,
        text_dim: int = 1024,
        **args,
    ):
        super().__init__(init_cfg)
        if pretrained is not None:
            assert (
                backbone.get("pretrained") is None
            ), "both backbone and segmentor set pretrained weight"
            backbone.pretrained = pretrained

            assert (
                text_encoder.get("pretrained") is None
            ), "both text encoder and segmentor set pretrained weight"

            if (
                "RN50" not in pretrained
                and "RN101" not in pretrained
                and "ViT-B" not in pretrained
            ):
                print("not CLIP pre-trained weight, using CLIP ViT-B-16")
                text_encoder.pretrained = "pretrained/ViT-B-16.pt"
            else:
                text_encoder.pretrained = pretrained

        self.backbone: BaseModule = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.context_decoder = builder.build_backbone(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index

        assert context_feature in ["attention", "backbone"]
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head: BaseDecodeHead | None = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat(
            [tokenize(c, context_length=self.context_length) for c in class_names]
        )
        self.num_classes = len(self.texts)

        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigDict):
        """Initialize ``decode_head``"""
        self.decode_head: BaseDecodeHead = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(
        self, auxiliary_head: Iterable[ConfigDict] | ConfigDict | None
    ):
        """Initialize ``auxiliary_head``"""
        self.auxiliary_head: nn.ModuleList | BaseDecodeHead | None
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, Iterable):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_identity_head(self, identity_head: ConfigDict | None):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, inputs: torch.Tensor):
        """Extract features from images."""
        return self.backbone(inputs)

    def after_extract_feat(self, x: torch.Tensor):
        x_orig = list(x[:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        # if self.context_feature == "attention":
        visual_context = torch.cat(
            [global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
            dim=2,
        ).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(
            self.texts.to(global_feat.device), self.contexts
        ).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, _K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum("bchw,bkc->bkhw", visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat(
            [x_orig[self.score_concat_index], score_map], dim=1
        )
        return text_embeddings, x_orig, score_map

    def _forward(self, inputs: torch.Tensor, data_samples: SampleList | None = None):
        x = self.extract_feat(inputs)

        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, _score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        x = [text_embeddings, *x_orig] if self.text_head else x_orig
        # print('text_embedding=', text_embeddings[0])
        out = self.decode_head(x)
        # print('cls_map=', out[0,:,40, 40])

        return resize(
            input=out,
            size=inputs.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

    def predict(self, inputs: torch.Tensor, data_samples: SampleList | None = None):
        x = self.extract_feat(inputs)

        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, _score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        x = [text_embeddings, *x_orig] if self.text_head else x_orig
        # print('text_embedding=', text_embeddings[0])
        return self.decode_head.predict(x, data_samples, self.test_cfg)
        # print('cls_map=', out[0,:,40, 40])

    def loss(
        self,
        inputs: torch.Tensor,
        data_samples: SampleList,
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)
        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, _score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        x = [text_embeddings, *x_orig] if self.text_head else x_orig

        losses = {}

        loss_decode = self.decode_head.loss(x, data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, "decode"))

        if self.identity_head is not None:
            loss_aux = self.identity_head.loss(x, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, "aux_identity"))

        if self.auxiliary_head is not None:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    loss_aux = aux_head.loss(x, data_samples, self.train_cfg)
                    losses.update(add_prefix(loss_aux, f"aux_{idx}"))
            else:
                loss_aux = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, "aux"))
                loss_aux = self._auxiliary_head_forward_train(_x_orig, data_samples)
                losses.update(loss_aux)

        return losses

    def encode_decode(self, inputs: torch.Tensor, batch_data_samples: SampleList):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)

        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, _score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        x = [text_embeddings, *x_orig] if self.text_head else x_orig
        # print('text_embedding=', text_embeddings[0])
        out = self.decode_head(x)
        # print('cls_map=', out[0,:,40, 40])

        return resize(
            input=out,
            size=inputs.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

    # TODO refactor
    def slide_inference(
        self, img: torch.Tensor, batch_data_samples: SampleList, rescale: bool
    ):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        assert self.test_cfg is not None
        h_stride, w_stride = getattr(self.test_cfg, "stride", (1, 1))
        h_crop, w_crop = getattr(self.test_cfg, "crop_size", (1, 1))
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, batch_data_samples)
                preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                device=img.device
            )
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=batch_data_samples[0].gt_sem_seg["ori_shape"][:2],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )
        return preds

    def whole_inference(
        self, img: torch.Tensor, batch_data_samples: SampleList, rescale: bool
    ):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, batch_data_samples)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = batch_data_samples[0].gt_sem_seg["ori_shape"][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )

        if torch.isnan(seg_logit).any():
            print("########### find NAN #############")

        return seg_logit

    def inference(self, img: torch.Tensor, img_meta: Sequence[Mapping], rescale: bool):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg is not None and self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))

        return output

    def simple_test(
        self, img: torch.Tensor, img_meta: Sequence[Mapping], rescale: bool = True
    ):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            return seg_pred.unsqueeze(0)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        return list(seg_pred)

    def aug_test(
        self,
        imgs: Sequence[torch.Tensor],
        img_metas: Sequence[Sequence[Mapping]],
        rescale=True,
    ):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)
