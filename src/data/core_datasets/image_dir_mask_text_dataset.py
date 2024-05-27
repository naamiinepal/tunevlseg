from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .basedataset import BaseImageTextMaskDataset

if TYPE_CHECKING:
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

    def get_tasks(self) -> list[Path]:
        class_names = {
            str(p.relative_to(self.mask_dir))
            for p in self.mask_dir.iterdir()
            if p.is_dir()
        }

        num_classes = len(class_names)

        if not num_classes:
            raise ValueError(f"No directories found in {self.mask_dir}")

        image_files = tuple(self.image_dir.glob(f"*{self.image_suffix}"))

        if not image_files:
            raise ValueError(f"No files found in {self.image_dir}")

        tasks: list[Path] = []

        # Cache for faster access
        mask_suffix = self.mask_suffix
        image_dir = self.image_dir
        for image_path in image_files:
            image_name = image_path.relative_to(image_dir).with_suffix(mask_suffix)
            tasks.extend(class_name / image_name for class_name in class_names)

        return tasks

    def __getitem__(self, index: int) -> dict[str, Any]:
        mask_name = Path(self.tasks[index])

        curr_prompt = str(mask_name.parents[-1])

        if self.insert_stop_at_last and curr_prompt[-1] != ".":
            curr_prompt += "."

        text_inputs = self.tokenizer(curr_prompt)

        image_path = self.image_dir / mask_name.with_suffix(self.image_suffix)

        image = self.load_image(
            path=image_path,
            imread_flags=cv2.IMREAD_COLOR,
            cvtColor_code=cv2.COLOR_BGR2RGB,
        )

        mask_path = self.mask_dir / mask_name

        mask = (
            self.load_image(
                path=mask_path,
                imread_flags=cv2.IMREAD_GRAYSCALE,
            ).astype(np.float32)
            / 255
        )

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
            "mask_shape": np.array(mask.shape[:-1]),  # Needed to collate properly
            "mask_name": mask_name,
            "prompt": curr_prompt,
            **text_inputs,
        }
