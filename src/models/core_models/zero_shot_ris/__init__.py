from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision.transforms import functional as TF

from .custom_openclip import CustomOpenCLIP
from .freesolo import CustomFreeSOLO
from .hfclip import CustomHFCLIP

if TYPE_CHECKING:
    from torch.serialization import FILE_LIKE

    from .baseclip import BaseCLIP

    InterpolationModeConvertible = TF.InterpolationMode | str | int


class ZeroShotRIS(nn.Module):
    def __init__(
        self,
        clip_pretrained_path: str,
        is_hf_model: bool,
        clip_interpolation_mode: InterpolationModeConvertible,
        solo_config: object,
        solo_state_dict_path: FILE_LIKE,
        masking_block_idx: int | None = -3,
        alpha: float = 0.95,
        beta: float = 0.5,
        *clip_args,
        **clip_kwargs,
    ) -> None:
        super().__init__()

        self.clip: BaseCLIP = (
            CustomHFCLIP(clip_pretrained_path, *clip_args, **clip_kwargs)
            if is_hf_model
            else CustomOpenCLIP(clip_pretrained_path, *clip_args, **clip_kwargs)
        )

        self.clip_interpolation_mode = self.get_torchvision_interpolation_mode(
            clip_interpolation_mode,
        )

        self.freesolo = CustomFreeSOLO(solo_config, solo_state_dict_path)

        self.masking_block_idx = masking_block_idx
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def get_torchvision_interpolation_mode(mode) -> TF.InterpolationMode:
        if isinstance(mode, TF.InterpolationMode):
            return mode

        if isinstance(mode, str):
            return TF.InterpolationMode(mode)

        if isinstance(mode, int):
            return TF._interpolation_modes_from_int(mode)

        msg = f"Unsupported interpolation mode: {mode}"
        raise ValueError(msg)

    def get_cropped_tensor(
        self,
        image_input: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_masks: torch.Tensor,
    ):
        # Fill the image outside of the mask but inside the bounding box with
        # the following pixel_mean
        # shape: (3, 1, 1)
        pixel_mean = image_input.mean((1, 2), keepdim=True)

        # pixel_mean = torch.tensor(
        #     [0.485, 0.456, 0.406],
        #     device=image_input.device,
        #     dtype=image_input.dtype,
        # ).reshape(3, 1, 1)

        # pred_masks: (B, H, W)
        channeled_pred_mask = pred_masks.unsqueeze(1)
        # image_input: (3, H, W)
        masked_images = (
            image_input * channeled_pred_mask + ~channeled_pred_mask * pixel_mean
        )

        clip_image_size = self.clip.image_size
        cropped_imgs = []
        for pred_box, masked_image in zip(pred_boxes, masked_images, strict=True):
            x1, y1, x2, y2 = pred_box.tolist()

            masked_image = TF.resized_crop(
                masked_image,
                y1,
                x1,
                (y2 - y1),
                (x2 - x1),
                [clip_image_size, clip_image_size],
                antialias=False,
                interpolation=self.clip_interpolation_mode,
            )

            cropped_imgs.append(masked_image)

        return torch.stack(cropped_imgs)

    def get_cropped_features(
        self,
        image_input: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_masks: torch.Tensor,
    ):
        # Separated in a different function to invoke gc faster
        cropped_tensor = self.get_cropped_tensor(image_input, pred_boxes, pred_masks)

        return self.clip.get_image_features(cropped_tensor)

    def get_text_ensemble(self, text_input: dict[str, torch.Tensor]):
        batched_text_features = self.clip.get_text_features(
            text_input["input_ids"][0],
            text_input["attention_mask"][0],
        )

        phrase_features, class_features = batched_text_features

        return self.beta * phrase_features + (1 - self.beta) * class_features

    def get_max_index(self, text_ensemble: torch.Tensor, visual_feature: torch.Tensor):
        # normalized features
        image_embeds = visual_feature / visual_feature.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_ensemble / text_ensemble.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

        return logits_per_image.argmax()

    def get_mask_features(self, image_input: torch.Tensor, pred_masks):
        clip_image_size = self.clip.image_size

        resized_image = TF.resize(
            image_input,
            [clip_image_size, clip_image_size],
            antialias=False,
            interpolation=self.clip_interpolation_mode,
        )

        patch_size = self.clip.patch_size
        mask_size = clip_image_size // patch_size

        # Add channel dimension to the pred_masks
        resized_masks = torch.stack(
            [
                TF.resize(mask, [mask_size, mask_size], antialias=False)
                for mask in pred_masks.unsqueeze(1)
            ],
        ).squeeze(1)

        return self.clip.get_image_features(
            pixel_values=resized_image.unsqueeze(0),
            pred_masks=resized_masks,
            masking_block_idx=self.masking_block_idx,
        )

    def get_visual_feature(self, image_input: torch.Tensor, pred_boxes, pred_masks):
        mask_features = self.get_mask_features(image_input, pred_masks)

        crop_features = self.get_cropped_features(image_input, pred_boxes, pred_masks)

        return self.alpha * mask_features + (1 - self.alpha) * crop_features

    def forward(self, image_input: torch.Tensor, text_input: dict[str, torch.Tensor]):
        """Steps to implement:

        1. image = Load an image in the original size (without normalization).
        2. freesolo_resized = Resize for freesolo, if it doesn't resize itself
        3. mask_resize = resize the to the size of sqrt(num_patches)
        4. Crop and resize the images using masks.
        5. clip_resize = Resize and normalize for original image for clip.
        6. Get classname and description to pass to the clip_text encoder.
        """
        if image_input.ndim == 4:
            if image_input.size(0) != 1:
                print("The image input should be of a single batch size.")
            image_input = image_input[0]

        # Get boxes from the output
        pred_boxes, pred_masks = self.freesolo(image_input)

        if len(pred_masks) == 0:
            print("No pred masks given. No image features are returned.")
            return torch.zeros(
                (1, *image_input.shape[1:]),
                dtype=image_input.dtype,
                device=image_input.device,
            )

        # Convert to integer type and move to cpu
        pred_boxes = pred_boxes.tensor.to(dtype=torch.int, device="cpu")

        visual_feature = self.get_visual_feature(image_input, pred_boxes, pred_masks)

        text_ensemble = self.get_text_ensemble(text_input)

        max_index = self.get_max_index(text_ensemble, visual_feature)

        selected_mask: torch.Tensor = pred_masks[max_index]

        return selected_mask[None, None, ...].float()
