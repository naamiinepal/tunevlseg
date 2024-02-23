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
import torch.nn.functional as F
from skimage import color

from ..detectron2.structures import ImageList
from .solov2 import SOLOv2
from .utils import get_images_color_similarity, point_nms

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class PseudoSOLOv2(SOLOv2):
    def forward(
        self,
        batched_inputs: Sequence[Mapping[str, Any]],
        branch: object = "supervised",
        given_proposals: object = None,
        val_mode: object = False,
    ):
        original_images = [x["image"] for x in batched_inputs]
        images = self.preprocess_image(
            batched_inputs,
        )  # return ImageList class, Images[0].shape = [batch, channel, H, W]
        # print(images.tensor.shape)
        features = self.backbone(images.tensor)
        if self.is_freemask:
            return [features]

        contains_gt = "instances" in batched_inputs[0]
        if contains_gt:
            gt_instances = [x["instances"] for x in batched_inputs]
            original_image_masks = [
                torch.ones_like(x[0], dtype=torch.float32) for x in original_images
            ]
            # mask out the bottom area where the COCO dataset probably has wrong annotations
            for i in range(len(original_image_masks)):
                im_h = batched_inputs[i]["height"]
                pixels_removed = int(
                    self.bottom_pixels_removed
                    * float(original_images[i].size(1))
                    / float(im_h),
                )
                if pixels_removed > 0:
                    original_image_masks[i][-pixels_removed:, :] = 0

            original_images = ImageList.from_tensors(
                original_images,
                self.backbone.size_divisibility,
            )
            original_image_masks = ImageList.from_tensors(
                original_image_masks,
                self.backbone.size_divisibility,
                pad_value=0.0,
            )
            self.add_bitmasks_from_boxes(
                gt_instances,
                original_images.tensor,
                original_image_masks.tensor,
            )
        else:
            gt_instances = None

        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        cate_pred, kernel_pred, emb_pred = self.ins_head(ins_features)
        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        if not self.training:
            # point nms.
            cate_pred = [
                point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                for cate_p in cate_pred
            ]
            # do inference for results.
            return self.inference(
                cate_pred,
                kernel_pred,
                emb_pred,
                mask_pred,
                images.image_sizes,
                batched_inputs,
            )

        if contains_gt and branch == "supervised":
            mask_feat_size = mask_pred.size()[-2:]
            targets = self.get_ground_truth(gt_instances, mask_feat_size)  # type:ignore
            return self.loss(cate_pred, kernel_pred, emb_pred, mask_pred, targets)
        return None

    def add_bitmasks_from_boxes(
        self,
        instances: Iterable,
        images: torch.Tensor,
        image_masks: torch.Tensor,
    ) -> None:
        stride = 4
        start = int(stride // 2)

        assert (
            images.size(2) % stride == 0
        ), f"The third dimension of image: {images.size(2)} must be divisible by stride: {stride}"
        assert (
            images.size(3) % stride == 0
        ), f"The fourth dimension of image: {images.size(3)} must be divisible by stride: {stride}"

        downsampled_images = F.avg_pool2d(
            images.float(),
            kernel_size=stride,
            stride=stride,
            padding=0,
        )
        image_masks = image_masks[:, start::stride, start::stride]

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(
                downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy(),
            )
            images_lab = torch.as_tensor(
                images_lab,
                device=downsampled_images.device,
                dtype=torch.float32,
            )
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = get_images_color_similarity(
                images_lab,
                image_masks[im_i],
                self.pairwise_size,
                self.pairwise_dilation,
            )

            # _h, _w = per_im_gt_inst.image_size
            # per_im_gt_inst.gt_masks = torch.stack(per_im_bitmasks_full, dim=0)[:, :h, :w]
            if len(per_im_gt_inst) > 0:
                per_im_gt_inst.image_color_similarity = torch.cat(
                    [images_color_similarity for _ in range(len(per_im_gt_inst))],
                    dim=0,
                )
