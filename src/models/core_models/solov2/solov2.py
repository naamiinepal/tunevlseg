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
from typing import TYPE_CHECKING, Any, Literal

import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F

from ..detectron2.layers import ShapeSpec
from ..detectron2.modelling import backbone
from ..detectron2.structures import Boxes, ImageList, Instances
from .solov2inshead import SOLOv2InsHead
from .solov2maskhead import SOLOv2MaskHead
from .utils import (
    center_of_mass,
    compute_pairwise_term,
    dice_coefficient,
    imrescale,
    mask_nms,
    matrix_nms,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableSequence, Sequence

    import numpy as np

    TensorIndex = int | slice | torch.Tensor | list | tuple | None


class SOLOv2(nn.Module):
    """SOLOv2 model. Creates FPN backbone, instance branch for kernels and categories prediction,
    mask branch for unified mask features.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        self.scale_ranges = cfg.MODEL.SOLOV2.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.SOLOV2.SIGMA
        # Instance parameters.
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.num_embs = 128

        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS

        # Inference parameters.
        self.max_before_nms = cfg.MODEL.SOLOV2.NMS_PRE
        self.score_threshold = cfg.MODEL.SOLOV2.SCORE_THR
        self.update_threshold = cfg.MODEL.SOLOV2.UPDATE_THR
        self.mask_threshold: float = cfg.MODEL.SOLOV2.MASK_THR
        self.max_per_img: int = cfg.MODEL.SOLOV2.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.SOLOV2.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.SOLOV2.NMS_SIGMA
        self.nms_type = cfg.MODEL.SOLOV2.NMS_TYPE

        # build the backbone.
        backbone_builder = getattr(backbone, cfg.MODEL.BACKBONE.NAME)
        self.backbone = backbone_builder(
            cfg,
            input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)),
        )
        backbone_shape = self.backbone.output_shape()
        self.is_freemask = cfg.MODEL.SOLOV2.IS_FREEMASK

        if not self.is_freemask:
            # build the ins head.
            instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
            self.ins_head = SOLOv2InsHead(cfg, instance_shapes)

            # build the mask head.
            mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
            self.mask_head = SOLOv2MaskHead(cfg, mask_shapes)

        if cfg.MODEL.SOLOV2.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")
            # for p in self.mask_head.parameters():
            #    p.requires_grad = False
            # print("froze mask head parameters")

        # loss
        self.ins_loss_weight: float = cfg.MODEL.SOLOV2.LOSS.DICE_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.SOLOV2.LOSS.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.MODEL.SOLOV2.LOSS.FOCAL_GAMMA
        self.focal_loss_weight = cfg.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT

        # free
        self.bottom_pixels_removed = 10
        self.pairwise_size = 3
        self.pairwise_dilation = 2
        self.pairwise_color_thresh = 0.3
        self._warmup_iters = 1000
        self.register_buffer("_iter", torch.zeros([1]))

        # image transform
        pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(
            3,
            1,
            1,
        )
        pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def preprocess_image(self, batched_inputs: Iterable[Mapping[str, torch.Tensor]]):
        """Normalize, pad and batch the input images."""
        images = [x["image"] for x in batched_inputs]
        # images = [self.normalizer(x) for x in images] # images[0].shape = [1,3,H,W], images= list[tensor]

        return ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
        )  # Class: ImageList, images[0].shape = (batch, channel, H, W)

    @torch.no_grad()
    def get_ground_truth(self, gt_instances: Sequence, mask_feat_size: Sequence[int]):
        if len(gt_instances) and gt_instances[0].has("image_color_similarity"):
            image_color_similarity_list = [
                gt_instance.image_color_similarity for gt_instance in gt_instances
            ]
        else:
            image_color_similarity_list = []

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        cate_soft_label_list = []
        for img_idx in range(len(gt_instances)):
            (
                cur_ins_label_list,
                cur_cate_label_list,
                cur_ins_ind_label_list,
                cur_grid_order_list,
                cur_cate_soft_label_list,
            ) = self.get_ground_truth_single(
                img_idx,
                gt_instances,
                mask_feat_size=mask_feat_size,
            )
            ins_label_list.append(cur_ins_label_list)
            cate_label_list.append(cur_cate_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
            cate_soft_label_list.append(cur_cate_soft_label_list)

        return (
            ins_label_list,
            cate_label_list,
            ins_ind_label_list,
            grid_order_list,
            cate_soft_label_list,
            image_color_similarity_list,
        )

    def get_ins_label(
        self,
        cate_label: torch.Tensor,
        emb_label: torch.Tensor,
        grid_order: MutableSequence[int],
        gt_bboxes_raw: torch.Tensor,
        gt_embs_raw: torch.Tensor,
        gt_labels_raw: torch.Tensor,
        gt_masks_raw: torch.Tensor,
        hit_indices: TensorIndex,
        ins_ind_label: MutableSequence[bool] | torch.Tensor,
        mask_feat_size: Sequence[int],
        num_grid: int,
    ):
        gt_bboxes = gt_bboxes_raw[hit_indices]
        gt_labels = gt_labels_raw[hit_indices]
        gt_masks = gt_masks_raw[hit_indices, ...]
        gt_embs = gt_embs_raw[hit_indices]

        half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
        half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

        # mass center
        center_ws, center_hs = center_of_mass(gt_masks)
        valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

        device = gt_labels_raw.device

        output_stride = 4
        np_gt_masks: np.ndarray = torch.as_tensor(
            gt_masks.permute(1, 2, 0),
            dtype=torch.uint8,
            device="cpu",
        ).numpy()
        np_gt_masks = imrescale(np_gt_masks, scale=1.0 / output_stride)  # type:ignore
        if len(np_gt_masks.shape) == 2:
            np_gt_masks = np_gt_masks[..., None]
        gt_masks = torch.as_tensor(
            np_gt_masks,
            dtype=torch.uint8,
            device=device,
        ).permute(2, 0, 1)

        ins_label = []

        for (
            seg_mask,
            gt_label,
            gt_emb,
            half_h,
            half_w,
            center_h,
            center_w,
            valid_mask_flag,
        ) in zip(
            gt_masks,
            gt_labels,
            gt_embs,
            half_hs,
            half_ws,
            center_hs,
            center_ws,
            valid_mask_flags,
            strict=False,
        ):
            if not valid_mask_flag:
                continue
            upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
            coord_w = int((center_w / upsampled_size[1]) // (1.0 / num_grid))
            coord_h = int((center_h / upsampled_size[0]) // (1.0 / num_grid))

            # left, top, right, down
            top_box = max(
                0,
                int(((center_h - half_h) / upsampled_size[0]) // (1.0 / num_grid)),
            )
            down_box = min(
                num_grid - 1,
                int(((center_h + half_h) / upsampled_size[0]) // (1.0 / num_grid)),
            )
            left_box = max(
                0,
                int(((center_w - half_w) / upsampled_size[1]) // (1.0 / num_grid)),
            )
            right_box = min(
                num_grid - 1,
                int(((center_w + half_w) / upsampled_size[1]) // (1.0 / num_grid)),
            )

            top = max(top_box, coord_h - 1)
            down = min(down_box, coord_h + 1)
            left = max(coord_w - 1, left_box)
            right = min(right_box, coord_w + 1)

            cate_label[top : (down + 1), left : (right + 1)] = gt_label
            emb_label[top : (down + 1), left : (right + 1)] = gt_emb
            # cate_label[coord_h, coord_w] = gt_label
            # cate_soft_label[coord_h, coord_w] = gt_soft_label
            for i in range(top, down + 1):
                for j in range(left, right + 1):
                    label = int(i * num_grid + j)

                    cur_ins_label = torch.zeros(
                        [mask_feat_size[0], mask_feat_size[1]],
                        dtype=torch.uint8,
                        device=device,
                    )
                    cur_ins_label[: seg_mask.shape[0], : seg_mask.shape[1]] = seg_mask
                    ins_label.append(cur_ins_label)
                    ins_ind_label[label] = True
                    grid_order.append(label)

        if len(ins_label) == 0:
            return torch.zeros(
                [0, mask_feat_size[0], mask_feat_size[1]],
                dtype=torch.uint8,
                device=device,
            )

        return torch.stack(ins_label, 0)

    def get_ground_truth_single(
        self,
        img_idx: int,
        gt_instances: Sequence,
        mask_feat_size: Sequence[int],
    ):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        gt_masks_raw = gt_instances[img_idx].gt_masks
        device = gt_labels_raw.device
        if hasattr(gt_instances[img_idx], "gt_embs"):
            gt_embs_raw = gt_instances[img_idx].gt_embs
        else:  # empty soft labels
            gt_embs_raw = torch.zeros(
                [gt_labels_raw.shape[0], self.num_embs],
                device=device,
            )
        if not torch.is_tensor(gt_masks_raw):
            gt_masks_raw = gt_masks_raw.tensor

        # ins
        gt_areas = torch.sqrt(
            (gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0])
            * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]),
        )

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        emb_label_list = []
        for (lower_bound, upper_bound), _, num_grid in zip(
            self.scale_ranges,
            self.strides,
            self.num_grids,
            strict=False,
        ):
            hit_indices = (
                ((gt_areas >= lower_bound) & (gt_areas <= upper_bound))
                .nonzero()
                .flatten()
            )
            num_ins = len(hit_indices)

            grid_order = []
            cate_label = torch.zeros(
                [num_grid, num_grid],
                dtype=torch.int64,
                device=device,
            )
            cate_label = torch.fill_(cate_label, self.num_classes)
            ins_ind_label = torch.zeros([num_grid**2], dtype=torch.bool, device=device)
            emb_label = torch.zeros([num_grid, num_grid, self.num_embs], device=device)

            ins_label = (
                torch.zeros(
                    [0, mask_feat_size[0], mask_feat_size[1]],
                    dtype=torch.uint8,
                    device=device,
                )
                if num_ins == 0
                else self.get_ins_label(
                    cate_label,
                    emb_label,
                    grid_order,
                    gt_bboxes_raw,
                    gt_embs_raw,
                    gt_labels_raw,
                    gt_masks_raw,
                    hit_indices,
                    ins_ind_label,
                    mask_feat_size,
                    num_grid,
                )
            )

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
            emb_label_list.append(emb_label)
        return (
            ins_label_list,
            cate_label_list,
            ins_ind_label_list,
            grid_order_list,
            emb_label_list,
        )

    def get_paired_losses(
        self,
        ins_pred_list: Iterable[torch.Tensor | None],
        ins_labels: Iterable[torch.Tensor],
        image_color_similarity: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor | Literal[0]]:
        # dice loss
        loss_ins = []
        loss_ins_max = []
        loss_pairwise = []
        for input, target, cur_image_color_similarity in zip(
            ins_pred_list,
            ins_labels,
            image_color_similarity,
            strict=False,
        ):
            if input is None:
                continue
            input_scores = torch.sigmoid(input)

            mask_losses_y = dice_coefficient(
                input_scores.max(dim=1, keepdim=True)[0],
                target.max(dim=1, keepdim=True)[0],
            )
            mask_losses_x = dice_coefficient(
                input_scores.max(dim=2, keepdim=True)[0],
                target.max(dim=2, keepdim=True)[0],
            )
            loss_ins_max.append((mask_losses_y + mask_losses_x).mean())

            mask_losses_y = dice_coefficient(
                input_scores.mean(dim=1, keepdim=True),
                target.float().mean(dim=1, keepdim=True),
            )
            mask_losses_x = dice_coefficient(
                input_scores.mean(dim=2, keepdim=True),
                target.float().mean(dim=2, keepdim=True),
            )
            loss_ins.append((mask_losses_y + mask_losses_x).mean())

            pairwise_losses = compute_pairwise_term(
                input[:, None, ...],
                self.pairwise_size,
                self.pairwise_dilation,
            )
            box_target = target.max(dim=1, keepdim=True)[0].expand(
                -1,
                target.shape[1],
                -1,
            ) * target.max(dim=2, keepdim=True)[0].expand(-1, -1, target.shape[2])

            # weights = (image_color_similarity >= self.pairwise_color_thresh).float()
            weights = (
                cur_image_color_similarity >= self.pairwise_color_thresh
            ).float() * box_target[:, None, ...].float()
            cur_loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(
                min=1.0,
            )
            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
            cur_loss_pairwise = cur_loss_pairwise * warmup_factor
            loss_pairwise.append(cur_loss_pairwise)

        if not loss_ins_max:
            loss_ins_max = 0
        else:
            loss_ins_max = torch.stack(loss_ins_max).mean()
            loss_ins_max = loss_ins_max * self.ins_loss_weight * 1.0

        if not loss_ins:
            loss_ins = 0
        else:
            loss_ins = torch.stack(loss_ins).mean()
            loss_ins = loss_ins * self.ins_loss_weight * 0.1

        if not loss_pairwise:
            loss_pairwise = 0
        else:
            loss_pairwise = torch.stack(loss_pairwise).mean()
            loss_pairwise = 1.0 * loss_pairwise

        return {
            "loss_ins": loss_ins,
            "loss_ins_max": loss_ins_max,
            "loss_pairwise": loss_pairwise,
        }

    def loss(
        self,
        cate_preds: Iterable[torch.Tensor],
        kernel_preds: Iterable[Iterable[torch.Tensor]],
        emb_preds: Iterable[torch.Tensor],
        ins_pred: torch.Tensor,
        targets: Iterable[Sequence],
        pseudo: bool = False,
    ):
        self._iter += 1
        (
            ins_label_list,
            cate_label_list,
            ins_ind_label_list,
            grid_order_list,
            emb_label_list,
            image_color_similarity_list,
        ) = targets

        # ins
        ins_labels = [
            torch.cat(list(ins_labels_level), 0)
            for ins_labels_level in zip(*ins_label_list, strict=False)
        ]
        if len(image_color_similarity_list):
            image_color_similarity = []
            for level_idx in range(len(ins_label_list[0])):
                level_image_color_similarity = []
                for img_idx in range(len(ins_label_list)):
                    num = ins_label_list[img_idx][level_idx].shape[0]
                    cur_image_color_sim = image_color_similarity_list[img_idx][
                        [0]
                    ].expand(num, -1, -1, -1)
                    level_image_color_similarity.append(cur_image_color_sim)
                image_color_similarity.append(torch.cat(level_image_color_similarity))
        else:
            image_color_similarity = ins_labels.copy()

        kernel_preds = [
            [
                kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[
                    :,
                    grid_orders_level_img,
                ]
                for kernel_preds_level_img, grid_orders_level_img in zip(
                    kernel_preds_level,
                    grid_orders_level,
                    strict=False,
                )
            ]
            for kernel_preds_level, grid_orders_level in zip(
                kernel_preds,
                zip(*grid_order_list, strict=False),
                strict=False,
            )
        ]
        # generate masks
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                _N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(
                    -1,
                    H,
                    W,
                )
                b_mask_pred.append(cur_ins_pred)
            b_mask_pred = None if len(b_mask_pred) == 0 else torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat(
                [
                    ins_ind_labels_level_img.flatten()
                    for ins_ind_labels_level_img in ins_ind_labels_level
                ],
            )
            for ins_ind_labels_level in zip(*ins_ind_label_list, strict=False)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        losses = self.get_paired_losses(
            ins_pred_list,
            ins_labels,
            image_color_similarity,
        )

        # cate
        cate_labels = [
            torch.cat(
                [
                    cate_labels_level_img.flatten()
                    for cate_labels_level_img in cate_labels_level
                ],
            )
            for cate_labels_level in zip(*cate_label_list, strict=False)
        ]
        flatten_cate_labels = torch.cat(cate_labels)
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        # prepare one_hot
        pos_inds = torch.nonzero(
            (flatten_cate_labels != self.num_classes) & (flatten_cate_labels != -1),
        ).squeeze(1)
        num_ins = len(pos_inds)

        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1

        if pseudo:
            flatten_cate_labels_oh = flatten_cate_labels_oh[pos_inds]
            flatten_cate_preds = flatten_cate_preds[pos_inds]

        if len(flatten_cate_preds):
            loss_cate = (
                self.focal_loss_weight
                * sigmoid_focal_loss_jit(
                    flatten_cate_preds,
                    flatten_cate_labels_oh,
                    gamma=self.focal_loss_gamma,
                    alpha=self.focal_loss_alpha,
                    reduction="sum",
                )
                / (num_ins + 1)
            )
        else:
            loss_cate = 0 * flatten_cate_preds.sum()

        emb_labels = [
            torch.cat(
                [
                    emb_labels_level_img.reshape(-1, self.num_embs)
                    for emb_labels_level_img in emb_labels_level
                ],
            )
            for emb_labels_level in zip(*emb_label_list, strict=False)
        ]
        flatten_emb_labels = torch.cat(emb_labels)
        emb_preds = [
            emb_pred.permute(0, 2, 3, 1).reshape(-1, self.num_embs)
            for emb_pred in emb_preds
        ]
        flatten_emb_preds = torch.cat(emb_preds)

        if num_ins:
            flatten_emb_labels = flatten_emb_labels[pos_inds]
            flatten_emb_preds = flatten_emb_preds[pos_inds]
            flatten_emb_preds = flatten_emb_preds / flatten_emb_preds.norm(
                dim=1,
                keepdim=True,
            )
            flatten_emb_labels = flatten_emb_labels / flatten_emb_labels.norm(
                dim=1,
                keepdim=True,
            )
            loss_emb = 1 - (flatten_emb_preds * flatten_emb_labels).sum(dim=-1)
            loss_emb = loss_emb.mean() * 4.0
        else:
            loss_emb = 0 * flatten_emb_preds.sum()

        return {
            "loss_emb": flatten_emb_preds.sum() * 0.0,
            "loss_cate": loss_cate,
            **losses,
        }

    @staticmethod
    def split_feats(feats: Sequence[torch.Tensor]):
        return (
            F.interpolate(feats[0], scale_factor=0.5, mode="bilinear"),
            feats[1],
            feats[2],
            feats[3],
            F.interpolate(feats[4], size=feats[3].shape[-2:], mode="bilinear"),
        )

    def inference(
        self,
        pred_cates: Sequence[torch.Tensor],
        pred_kernels: torch.Tensor,
        pred_embs: torch.Tensor,
        pred_masks: torch.Tensor,
        cur_sizes: Sequence[tuple[int, int]],
        images: Iterable[Mapping[str, int]],
        keep_train_size: bool = False,
    ):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx, ori_img in enumerate(images):
            # image size.
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [
                pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                for i in range(num_ins_levels)
            ]
            pred_kernel = [
                pred_kernels[i][img_idx]
                .permute(1, 2, 0)
                .view(-1, self.num_kernels)
                .detach()
                for i in range(num_ins_levels)
            ]
            pred_emb = [
                pred_embs[i][img_idx].view(-1, self.num_embs).detach()
                for i in range(num_ins_levels)
            ]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)
            pred_emb = torch.cat(pred_emb, dim=0)

            # inference for single image.
            result = self.inference_single_image(
                pred_cate,
                pred_kernel,
                pred_emb,
                pred_mask,
                cur_sizes[img_idx],
                ori_size,
                keep_train_size,
            )
            results.append({"instances": result})
        return results

    def get_results(
        self,
        cur_size: Sequence[int],
        cate_labels: torch.Tensor,
        cate_scores: torch.Tensor,
        emb_preds: torch.Tensor,
        keep: TensorIndex,
        keep_train_size: bool,
        maskness: torch.Tensor,
        ori_size: tuple[int, int],
        scores: torch.Tensor,
        seg_preds: torch.Tensor,
    ):
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.shape[-2:]
        ratio = max(math.ceil(h / f_h), math.ceil(w / f_w))
        upsampled_size_out = (int(f_h * ratio), int(f_w * ratio))

        seg_preds = seg_preds[keep, :, :]

        scores = scores[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        maskness = maskness[keep]
        emb_preds = emb_preds[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[: self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        scores = scores[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        maskness = maskness[sort_inds]
        emb_preds = emb_preds[sort_inds]

        # reshape to original size.
        seg_preds: torch.Tensor = F.interpolate(
            seg_preds.unsqueeze(0),
            size=upsampled_size_out,
            mode="bilinear",
        )[:, :, :h, :w]

        # keep_train_size for self-training
        seg_masks = (
            seg_preds
            if keep_train_size
            else F.interpolate(
                seg_preds,
                size=ori_size,
                mode="bilinear",
            )
        ).squeeze(0)

        seg_masks = seg_masks > self.mask_threshold

        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > 0
        scores = scores[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        seg_masks = seg_masks[keep]
        maskness = maskness[keep]
        emb_preds = emb_preds[keep]

        results = Instances(seg_masks.shape[1:])  # type:ignore unable to find the shape during type check
        results.pred_classes = cate_labels
        results.scores = scores
        results.category_scores = cate_scores
        results.maskness = maskness
        results.pred_masks = seg_masks
        # normalize the embeddings
        results.pred_embs = emb_preds / emb_preds.norm(dim=-1, keepdim=True)

        width_proj = seg_masks.max(1)[0]
        height_proj = seg_masks.max(2)[0]
        width, height = width_proj.sum(1), height_proj.sum(1)
        center_ws, _ = center_of_mass(width_proj[:, None, :])
        _, center_hs = center_of_mass(height_proj[:, :, None])
        pred_boxes = torch.stack(
            [
                center_ws - 0.5 * width,
                center_hs - 0.5 * height,
                center_ws + 0.5 * width,
                center_hs + 0.5 * height,
            ],
            1,
        )
        results.pred_boxes = Boxes(pred_boxes)
        return results

    def inference_single_image(
        self,
        cate_preds: torch.Tensor,
        kernel_preds: torch.Tensor,
        emb_preds: torch.Tensor,
        seg_preds: torch.Tensor,
        cur_size: tuple[int, int],
        ori_size: tuple[int, int],
        keep_train_size: bool = False,
    ):
        # process.
        inds = cate_preds > self.score_threshold
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            results = Instances(ori_size)
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            results.scores = torch.tensor([])
            results.category_scores = torch.tensor([])
            results.maskness = torch.tensor([])
            results.pred_embs = torch.tensor([])
            return results

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]
        emb_preds = emb_preds[inds[:, 0]]

        if keep_train_size:  # used in self-training
            # sort and keep top nms_pre
            sort_inds = torch.argsort(cate_scores, descending=True)
            max_pseudo_labels = self.max_before_nms
            if len(sort_inds) > max_pseudo_labels:
                sort_inds = sort_inds[:max_pseudo_labels]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]
            kernel_preds = kernel_preds[sort_inds]
            emb_preds = emb_preds[sort_inds]
            inds = inds[sort_inds]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[: size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1] : size_trans[ind_]] *= self.instance_strides[
                ind_
            ]
        strides = strides[inds[:, 0]]

        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks: torch.Tensor = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            results.scores = torch.tensor([])
            results.category_scores = torch.tensor([])
            results.maskness = torch.tensor([])
            results.pred_embs = torch.tensor([])
            return results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        emb_preds = emb_preds[keep, :]

        # maskness.
        seg_masks = seg_preds > self.mask_threshold
        maskness = (seg_preds * seg_masks.float()).sum((1, 2)) / seg_masks.sum((1, 2))

        scores = cate_scores * maskness

        # sort and keep top nms_pre
        sort_inds = torch.argsort(scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[: self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        scores = scores[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        maskness = maskness[sort_inds]
        emb_preds = emb_preds[sort_inds]

        if self.nms_type == "matrix":
            # matrix nms & filter.
            scores = matrix_nms(
                cate_labels,
                seg_masks,
                sum_masks,
                scores,
                sigma=self.nms_sigma,
                kernel=self.nms_kernel,
            )
            keep = scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(
                cate_labels,
                seg_masks,
                sum_masks,
                scores,
                nms_thr=self.mask_threshold,
            )
        else:
            raise NotImplementedError

        if keep.sum() == 0:
            results = Instances(ori_size)
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            results.scores = torch.tensor([])
            results.category_scores = torch.tensor([])
            results.maskness = torch.tensor([])
            results.pred_embs = torch.tensor([])
            return results

        return self.get_results(
            cur_size,
            cate_labels,
            cate_scores,
            emb_preds,
            keep,
            keep_train_size,
            maskness,
            ori_size,
            scores,
            seg_preds,
        )
