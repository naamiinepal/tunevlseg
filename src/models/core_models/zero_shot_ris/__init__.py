from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TF

from .custom_openclip import CustomOpenCLIP
from .freesolo import CustomFreeSOLO
from .hfclip import CustomHFCLIP

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        cache_dir: Path | str | None = None,
        read_cache: bool = False,
        write_cache: bool = False,
        cache_object_glob: str = "*.npz",
        num_masks: int = 1,
        return_similarity: bool = False,
        force_no_load_models: bool = False,
        *clip_args,
        **clip_kwargs,
    ) -> None:
        if num_masks < 1:
            msg = "The number of masks to be returned should be >= 1."
            raise ValueError(msg)

        if cache_dir is None and (read_cache or write_cache):
            msg = "Cache directory must be specified if caching is enabled."
            raise ValueError(msg)

        super().__init__()

        self.clip_interpolation_mode = self.get_torchvision_interpolation_mode(
            clip_interpolation_mode,
        )

        if not force_no_load_models:
            self.clip: BaseCLIP = (
                CustomHFCLIP(clip_pretrained_path, *clip_args, **clip_kwargs)
                if is_hf_model
                else CustomOpenCLIP(clip_pretrained_path, *clip_args, **clip_kwargs)
            )

            self.freesolo = CustomFreeSOLO(solo_config, solo_state_dict_path)

        self.cache_prefix = clip_pretrained_path.replace("/", "_")

        if cache_dir is None:
            self.cache_dir = None
        else:
            # Create cache directory and its parents if not already created
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = cache_dir

            if read_cache:
                self.existing_cached_files = set(cache_dir.glob(cache_object_glob))

        self.read_cache = read_cache
        self.write_cache = write_cache

        self.masking_block_idx = masking_block_idx
        self.alpha = alpha
        self.beta = beta

        self.num_masks = num_masks
        self.return_similarity = return_similarity

    @staticmethod
    def get_torchvision_interpolation_mode(mode) -> TF.InterpolationMode:
        if isinstance(mode, TF.InterpolationMode):
            return mode

        if isinstance(mode, str):
            return TF.InterpolationMode(mode)

        if isinstance(mode, int):
            return TF._interpolation_modes_from_int(mode)

        msg = f"Unsupported interpolation mode: {mode} of type: {type(mode)}"
        raise ValueError(msg)

    def get_cropped_tensor(
        self,
        image_input: torch.Tensor,
        pred_boxes: torch.IntTensor,
        pred_masks: torch.BoolTensor,
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
        pred_boxes: torch.IntTensor,
        pred_masks: torch.BoolTensor,
    ):
        # Separated in a different function to invoke gc faster
        cropped_tensor = self.get_cropped_tensor(image_input, pred_boxes, pred_masks)

        return self.clip.get_image_features(cropped_tensor)

    def get_text_ensemble(
        self,
        text_input: dict[str, torch.Tensor],
        current_base_cache_filename: Path | None,
        image_like_kwargs: Mapping[str, Any],
    ):
        textual_feature_cache_filename = self.get_cache_path(
            current_base_cache_filename,
            f"{self.cache_prefix}_textual_feature",
        )

        if (
            textual_feature_cache_filename is not None
            and self.read_cache
            and textual_feature_cache_filename in self.existing_cached_files
        ):
            data = np.load(textual_feature_cache_filename)

            zero_tensor = torch.zeros(1, **image_like_kwargs)

            # Load numpy only if needed
            if self.beta == 0:
                phrase_features = zero_tensor
            else:
                np_phrase_features = data["phrase_features"]

                phrase_features = torch.as_tensor(
                    np_phrase_features,
                    **image_like_kwargs,
                )  # type:ignore

            # Load numpy only if needed
            if self.beta == 1:
                class_features = zero_tensor
            else:
                np_class_features = data["class_features"]
                class_features = torch.as_tensor(np_class_features, **image_like_kwargs)  # type:ignore
        else:
            batched_text_features = self.clip.get_text_features(
                input_ids=text_input["input_ids"][0],
                attention_mask=text_input["attention_mask"][0],
            )

            phrase_features, class_features = batched_text_features

            if textual_feature_cache_filename is not None and self.write_cache:
                np.savez_compressed(
                    textual_feature_cache_filename,
                    phrase_features=phrase_features.cpu().numpy(),
                    class_features=class_features.cpu().numpy(),
                )

        return self.beta * phrase_features + (1 - self.beta) * class_features

    def get_max_index(self, text_ensemble: torch.Tensor, visual_feature: torch.Tensor):
        # normalized features
        image_embeds = visual_feature / visual_feature.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_ensemble / text_ensemble.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

        if self.num_masks == 1:
            max_sim, max_idx = logits_per_image.max(dim=-1)
            if not self.return_similarity:
                return max_idx
            return max_idx, max_sim

        topk_sim, topk_idx = logits_per_image.topk(self.num_masks)

        if not self.return_similarity:
            return topk_idx
        return topk_idx, topk_sim

    def get_mask_features(
        self,
        image_input: torch.Tensor,
        pred_masks: torch.BoolTensor,
    ):
        clip_image_size = self.clip.image_size

        resized_image = TF.resize(
            image_input,
            [clip_image_size, clip_image_size],
            antialias=False,
            interpolation=self.clip_interpolation_mode,
        )

        # Add batch dimension to the resized image
        resized_image = resized_image.unsqueeze(0)

        patch_size = self.clip.patch_size
        mask_size = clip_image_size // patch_size

        # pred_masks: shape (B, H, W)
        resized_masks = TF.resize(
            pred_masks,
            size=[mask_size, mask_size],
            interpolation=TF.InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )

        return self.clip.get_image_features(
            pixel_values=resized_image,
            pred_masks=resized_masks,
            masking_block_idx=self.masking_block_idx,
        )

    def get_visual_feature(
        self,
        image_input: torch.Tensor,
        pred_boxes: torch.IntTensor,
        pred_masks: torch.BoolTensor,
        current_base_cache_filename: Path | None,
    ):
        image_like_kwargs = {
            "dtype": image_input.dtype,
            "device": image_input.device,
        }

        visual_feature_cache_filename = self.get_cache_path(
            current_base_cache_filename,
            f"{self.cache_prefix}_visual_feature",
        )

        if (
            visual_feature_cache_filename is not None
            and self.read_cache
            and visual_feature_cache_filename in self.existing_cached_files
        ):
            data = np.load(visual_feature_cache_filename)
            zero_tensor = torch.zeros(1, **image_like_kwargs)

            # Load numpy only if needed
            if self.alpha == 0:
                mask_features = zero_tensor
            else:
                np_mask_features = data["mask_features"]
                mask_features = torch.as_tensor(np_mask_features, **image_like_kwargs)  # type:ignore

            # Load numpy only if needed
            if self.alpha == 1:
                crop_features = zero_tensor
            else:
                np_crop_features = data["crop_features"]
                crop_features = torch.as_tensor(np_crop_features, **image_like_kwargs)  # type:ignore

        elif visual_feature_cache_filename is not None and self.write_cache:
            # Need to calculate both features if writing to cache
            mask_features = self.get_mask_features(image_input, pred_masks)
            crop_features = self.get_cropped_features(
                image_input,
                pred_boxes,
                pred_masks,
            )

            np.savez_compressed(
                visual_feature_cache_filename,
                mask_features=mask_features.cpu().numpy(),
                crop_features=crop_features.cpu().numpy(),
            )
        else:
            zero_tensor = torch.zeros(1, **image_like_kwargs)
            mask_features = (
                self.get_mask_features(image_input, pred_masks)
                if self.alpha != 0
                else zero_tensor
            )
            crop_features = (
                self.get_cropped_features(image_input, pred_boxes, pred_masks)
                if self.alpha != 1
                else zero_tensor
            )

        return self.alpha * mask_features + (1 - self.alpha) * crop_features

    @staticmethod
    def get_cache_path(current_base_cache_filename: Path | None, stem_postfix: str):
        if current_base_cache_filename is None:
            return None
        return current_base_cache_filename.with_stem(
            f"{current_base_cache_filename.stem}_{stem_postfix}",
        )

    def get_freesolo_predictions(
        self,
        current_base_cache_filename: Path | None,
        image_input: torch.Tensor,
    ):
        image_like_kwargs = {
            "dtype": image_input.dtype,
            "device": image_input.device,
        }

        freesolo_cache_filename = self.get_cache_path(
            current_base_cache_filename,
            "freesolo",
        )

        if (
            freesolo_cache_filename is not None
            and self.read_cache
            and freesolo_cache_filename in self.existing_cached_files
        ):
            data = np.load(freesolo_cache_filename)
            np_masks = data["masks"]

            if len(np_masks) == 0:
                return None

            np_boxes = data["boxes"]

            pred_masks = torch.as_tensor(np_masks, device=image_like_kwargs["device"])  # type:ignore

            pred_boxes: torch.IntTensor = torch.as_tensor(np_boxes, dtype=torch.int16)  # type:ignore
        else:
            # Get boxes from the output
            pred_masks: torch.BoolTensor
            pred_boxes_raw, pred_masks = self.freesolo(image_input)

            if len(pred_masks) == 0:
                print("No pred masks given. No image features are returned.")

                if freesolo_cache_filename is not None and self.write_cache:
                    np.savez_compressed(freesolo_cache_filename, masks=[])

                return None

            # Convert to integer type and move to cpu
            pred_boxes = pred_boxes_raw.tensor.to(dtype=torch.int16, device="cpu")

            if freesolo_cache_filename is not None and self.write_cache:
                np.savez_compressed(
                    freesolo_cache_filename,
                    boxes=pred_boxes.numpy(),
                    masks=pred_masks.cpu().numpy(),
                )

        return pred_boxes, pred_masks

    def forward(self, image_input: torch.Tensor, text_input: dict[str, Any]):
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

        if self.cache_dir is None:
            current_base_cache_filename = None
        else:
            # contains extension
            cache_name: str = text_input["cache_name"][0]
            current_base_cache_filename = (self.cache_dir / cache_name).with_suffix(
                ".npz",
            )

        image_like_kwargs = {
            "dtype": image_input.dtype,
            "device": image_input.device,
        }

        freesolo_predictions = self.get_freesolo_predictions(
            current_base_cache_filename,
            image_input,
        )

        if freesolo_predictions is None:
            print("No pred masks could be extracted from the image.")
            return torch.zeros(
                (1, 1, *image_input.shape[1:]),
                **image_like_kwargs,
            )

        pred_boxes, pred_masks = freesolo_predictions

        visual_feature = self.get_visual_feature(
            image_input,
            pred_boxes,
            pred_masks,
            current_base_cache_filename,
        )

        text_ensemble = self.get_text_ensemble(
            text_input,
            current_base_cache_filename,
            image_like_kwargs,
        )

        max_output = self.get_max_index(text_ensemble, visual_feature)

        H, W = pred_masks.shape[1:]
        resulting_mask_size = (-1, 1, H, W)

        # If not returning similarity
        if not isinstance(max_output, tuple):
            return (
                pred_masks[max_output]
                .reshape(resulting_mask_size)
                .to(dtype=image_like_kwargs["dtype"])
            )

        indices, values = max_output

        return pred_masks[indices].reshape(resulting_mask_size).to(
            dtype=image_like_kwargs["dtype"],
        ), values
