from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import torch
    from cv2.typing import MatLike

    from .basedataset import ReturnTensorsType, StrPath

    PromptType = Literal[
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
    ]
    PromptTypeOrRandom = PromptType | Literal["random"]

    PromptMappingType = Mapping[PromptType, str | Sequence[str]]


class ZeroShotDataset(Dataset):
    def __init__(
        self,
        images_dir: StrPath,
        masks_dir: StrPath,
        caps_file: StrPath,
        tokenizer_pretrained_path: StrPath,
        prompt_type: PromptTypeOrRandom,
        object_class: str,
        transforms: Callable | None = None,
        context_length: int | None = None,
        return_tensors: ReturnTensorsType = None,
        override_prompt: str | None = None,
    ) -> None:
        super().__init__()

        with open(caps_file, encoding="locale") as fp:
            self.imgs_captions = json.load(fp)

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_pretrained_path,
            **({} if context_length is None else {"model_max_length": context_length}),
        )

        # LSP couldn't infer the type for the instance methods
        self.prompt_type: PromptTypeOrRandom = prompt_type

        self.object_class = object_class
        self.transforms = transforms
        self.return_tensors = return_tensors
        self.override_prompt = override_prompt

    def __len__(self) -> int:
        return len(self.imgs_captions)

    def __getitem__(self, index) -> dict[str, Any]:
        cap = self.imgs_captions[index]

        image = self.load_image(self.images_dir / cap["img_name"], cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_name = cap["mask_name"]
        mask = self.load_image(self.masks_dir / mask_name, cv2.IMREAD_GRAYSCALE)

        # Convert mask to floating point and add the final channel
        mask = mask[..., None].astype(np.float32) / 255

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        text_inputs = self.get_text_output(cap["prompts"])

        mask_shape = image.shape[:-1]

        # Metadata are needed to save the image for the predict step
        return {
            "image": image,
            "mask": mask,
            "mask_shape": np.array(mask_shape),
            "mask_name": mask_name,
            **text_inputs,
        }

    def get_curr_prompt(self, prompt_mapping: PromptMappingType) -> str:
        # Use overrided prompt if provided
        if self.override_prompt is not None:
            return self.override_prompt

        if self.prompt_type == "random":
            # Randomly select a prompt except the first one i.e., p0
            possible_choices = tuple(prompt_mapping.values())[1:]
            prompt = random.choice(possible_choices)
        else:
            prompt = prompt_mapping[self.prompt_type]

        if isinstance(prompt, str):
            return prompt

        # Randomly choose from a list of prompts
        return random.choice(prompt)

    def get_text_output(
        self,
        prompt_mapping: PromptMappingType,
    ) -> dict[str, list[int]] | dict[str, np.ndarray] | dict[str, torch.Tensor]:
        """Get a format randomly if prompt_format_list has more than one entries
        else, if the format is fixed, the only entry is fetched making it deterministic.

        Args:
        ----
            phrase: The phrase to be used for the prompt.

        Returns:
        -------
            The formatted prompt.

        """
        prompt = self.get_curr_prompt(prompt_mapping)

        return self.tokenizer(
            [prompt, self.object_class],
            truncation=True,
            padding=True,
            return_tensors=self.return_tensors,
        )

    @staticmethod
    def load_image(path: StrPath, flags: int = cv2.IMREAD_UNCHANGED) -> MatLike:
        """Load an image from a path.

        Args:
        ----
            path: The path to the image.
            flags: The flags to be passed to `cv2.imread` function while loading.

        Raises:
        ------
            ValueError: If image is not found in the path.

        Returns:
        -------
            Image loaded from the path as a numpy-like array.

        """
        img = cv2.imread(str(path), flags)

        if img is None:
            msg = f"Image not found in the path: {path}"
            raise ValueError(msg)

        return img
