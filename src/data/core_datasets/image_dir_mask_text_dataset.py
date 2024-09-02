from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .basedataset import BaseImageTextMaskDataset

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .basedataset import StrOrPath


class ImageDirTextMaskDataset(BaseImageTextMaskDataset):
    def __init__(
        self,
        *,
        image_dir: StrOrPath,
        mask_dir: StrOrPath,
        image_suffix: str,
        mask_suffix: str,
        insert_stop_at_last: bool = False,
        **kwargs,
    ) -> None:
        if image_suffix[0] != ".":
            raise ValueError(f"image_suffix must start with a period: {image_suffix=}")

        if mask_suffix[0] != ".":
            raise ValueError(f"mask_suffix must start with a period: {mask_suffix=}")

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        tasks = self.get_tasks()

        self.insert_stop_at_last = insert_stop_at_last

        super().__init__(tasks=tasks, **kwargs)

    def get_tasks(self) -> list[Mapping[str, str]]:
        # Throw error in the early stage if no directories exist
        num_classes = len(tuple(p for p in self.mask_dir.iterdir() if p.is_dir()))

        if not num_classes:
            raise ValueError(f"No directories found in {self.mask_dir}")

        tasks: list[Mapping[str, str]] = []

        for mask_path in self.mask_dir.glob(f"*/*{self.mask_suffix}"):
            mask_name = mask_path.name
            class_name = mask_path.parent.name

            tasks.append({"class_name": class_name, "mask_name": mask_name})

        return tasks

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]
        class_name = str(task["class_name"])

        curr_prompt = (
            f"{class_name}."
            if self.insert_stop_at_last and class_name[-1] != "."
            else class_name
        )

        text_inputs = self.tokenizer(curr_prompt)

        mask_name = Path(task["mask_name"])

        image_path = self.image_dir / mask_name.with_suffix(self.image_suffix)

        image = self.load_image(
            path=image_path,
            imread_flags=cv2.IMREAD_COLOR,
            cvtColor_code=cv2.COLOR_BGR2RGB,
        )

        # Update mask_name to save in this way later
        mask_name = class_name / mask_name
        mask_path = self.mask_dir / mask_name

        mask = (
            self.load_image(
                path=mask_path,
                imread_flags=cv2.IMREAD_GRAYSCALE,
            ).astype(np.float32)
            / 255
        )

        # Need to calculate here, may be changed below
        mask_shape = np.array(mask.shape)

        # Add the final channel layer to mask
        mask = mask[..., None]

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Metadata are needed to save the image for the predict step
        return {
            "image": image,
            "mask": mask,
            "mask_shape": mask_shape,  # Needed to collate properly
            "mask_name": str(mask_name),
            "prompt": curr_prompt,
            **text_inputs,
        }
