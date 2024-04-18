from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .basedataset import BaseImageTextMaskDataset

if TYPE_CHECKING:
    from .basedataset import (
        StrOrPath,
    )

    PromptType = str | Sequence[str]
    PromptMappingType = Mapping[str, PromptType]


class ImageTextMaskDataset(BaseImageTextMaskDataset):
    def __init__(
        self,
        *,
        image_dir: StrOrPath,
        mask_dir: StrOrPath,
        task_path: StrOrPath,
        prompt_index: int,
        override_prompt: str | None = None,
        insert_stop_at_last: bool = False,
        **kwargs,
    ) -> None:
        tasks = self.get_tasks(task_path)

        super().__init__(tasks=tasks, **kwargs)

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.prompt_map_index = f"p{prompt_index}" if prompt_index >= 0 else "random"

        self.override_prompt = override_prompt

        self.insert_stop_at_last = insert_stop_at_last

    @staticmethod
    def get_tasks(task_path: StrOrPath) -> list[dict[str, str | PromptMappingType]]:
        with open(task_path, encoding="locale") as fp:
            return json.load(fp)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]

        # Get img_name and wrap in str to make linter happy
        img_name = str(task["img_name"])
        image = self.load_image(
            path=self.image_dir / img_name,
            imread_flags=cv2.IMREAD_COLOR,
            cvtColor_code=cv2.COLOR_BGR2RGB,
        )

        # Get mask_name and wrap in str to make linter happy
        mask_name = str(task["mask_name"])

        # Mask needs to be of type float32
        mask = (
            self.load_image(self.mask_dir / mask_name, cv2.IMREAD_GRAYSCALE).astype(
                np.float32,
            )
            / 255
        )

        # Add the final channel layer to mask
        mask = mask[..., None]

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        curr_prompt = self.get_curr_prompt(task)

        if self.insert_stop_at_last and curr_prompt[-1] != ".":
            curr_prompt += "."

        text_inputs = self.tokenizer(curr_prompt)

        # Metadata are needed to save the image for the predict step
        return {
            "image": image,
            "mask": mask,
            "mask_shape": np.array(mask.shape),  # Needed to collate properly
            "mask_name": mask_name,
            "prompt": curr_prompt,
            **text_inputs,
        }

    def get_curr_prompt(self, task: Mapping[str, Any]) -> str:
        prompts: PromptMappingType = task["prompts"]

        if not isinstance(prompts, Mapping):
            msg = f"Expected `prompts` to be a `Mapping` but got: {type(prompts)} instead."
            raise TypeError(msg)
        # Use overrided prompt if provided
        if self.override_prompt is not None:
            return self.override_prompt

        if self.prompt_map_index == "random":
            # sort the keys of the mapping without leading `p`
            # and converting the remainder to int
            # This implementation doesn't assume the order in the dict
            possible_keys = sorted(prompts, key=lambda x: int(x[1:]))

            # Randomly select a prompt except the first one i.e., p0
            map_index = random.choice(possible_keys[1:])
        else:
            map_index = self.prompt_map_index

        curr_prompt = prompts[map_index]

        if isinstance(curr_prompt, str):
            return curr_prompt

        # Randomly choose from a list of prompts
        return random.choice(curr_prompt)
