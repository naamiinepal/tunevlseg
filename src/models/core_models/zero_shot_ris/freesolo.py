import torch
from torch import nn

from ..solov2 import PseudoSOLOv2


class CustomFreeSOLO(nn.Module):
    def __init__(self, solo_config, solo_state_dict_path) -> None:
        super().__init__()
        self.model = self.load_model(solo_config, solo_state_dict_path)

    @staticmethod
    def load_model(solo_config, solo_state_dict_path):
        solo = PseudoSOLOv2(solo_config)

        # Load state_dict to cpu first
        state_dict = torch.load(solo_state_dict_path, map_location="cpu")

        solo.load_state_dict(state_dict)

        return solo

    def forward(self, image_input: torch.Tensor):
        image_height = image_input.size(1)
        image_width = image_input.size(2)

        batched_images = [
            {
                "image": image_input,
                "height": image_height,
                "width": image_width,
            },
        ]

        solo_pred = self.model(batched_images)[0]

        pred_masks = solo_pred["instances"].pred_masks
        pred_boxes = solo_pred["instances"].pred_boxes
        return pred_boxes, pred_masks
