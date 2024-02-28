from typing import Any

import numpy as np
import torch
from torch import nn
from torch.serialization import FILE_LIKE
from torchvision.transforms import Normalize

from ..detectron2.structures.boxes import Boxes
from ..solov2 import PseudoSOLOv2


class CustomFreeSOLO(nn.Module):
    def __init__(
        self,
        solo_config: object,
        solo_state_dict_path: FILE_LIKE,
        normalizer_inplace: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = self.load_model(solo_config, solo_state_dict_path, *args, **kwargs)

    @staticmethod
    def load_model(
        solo_config: object,
        solo_state_dict_path: FILE_LIKE,
        load_strict: bool = True,
        *load_args,
        **load_kwargs,
    ):
        solo = PseudoSOLOv2(solo_config)

        # Load state_dict to cpu first
        state_dict = torch.load(solo_state_dict_path, "cpu", *load_args, **load_kwargs)

        solo.load_state_dict(state_dict, strict=load_strict)

        return solo

    def forward(self, image_input: torch.Tensor) -> tuple[Boxes, torch.BoolTensor]:
        image_height = image_input.size(1)
        image_width = image_input.size(2)

        batched_images = [
            {
                "image": image_input,
                "height": image_height,
                "width": image_width,
            },
        ]

        solo_pred_instances = self.model(batched_images)[0]["instances"]

        pred_boxes: Boxes = solo_pred_instances.pred_boxes
        pred_masks: torch.BoolTensor = solo_pred_instances.pred_masks
        return pred_boxes, pred_masks
