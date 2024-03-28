from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.data.core_datasets.basedataset import BaseImageTextMaskDataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import torch

    from src.data.core_datasets.basedataset import (
        CollateFnType,
        ReturnTensorsType,
        StrOrPath,
        TransformsType,
    )

    JSONMapping = Mapping[str, Any]
    PromptMethodType = Literal["fixed", "shuffle", "shuffle+"]
    Str2SetInt = Mapping[str, set[int]]

    ImageLike = TypeVar("ImageLike", np.ndarray, torch.Tensor)


class OpenDomainBaseDataset(BaseImageTextMaskDataset, ABC):
    def __init__(
        self,
        task_json_path: StrOrPath,
        image_dir: StrOrPath,
        mask_dir: StrOrPath,
        tokenizer_pretrained_path: StrOrPath,
        transforms: TransformsType = None,
        return_tensors: ReturnTensorsType = None,
        prompt_method: PromptMethodType = "fixed",
        neg_prob: float = 0,
        neg_sample_tries: int = 1000,
        filter_tasks: bool = False,
        collate_fn: CollateFnType = None,
    ) -> None:
        tasks = self.load_tasks(task_json_path, filter_tasks)

        # Needed only for negative sampling and is expensive
        if neg_prob > 0:
            phrase2image_ids = self.get_phrase2image_ids(tasks)
            self.unique_phrases = tuple(phrase2image_ids)
            self.phrase2image_ids = phrase2image_ids
        else:
            # For the sake of types, keep them empty
            self.phrase2image_ids: Str2SetInt = {}
            self.unique_phrases: tuple[str, ...] = ()

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.neg_prob = neg_prob
        self.neg_sample_tries = neg_sample_tries

        self.prompt_format_choices = self.get_prompt_list(prompt_method)

        super().__init__(
            tasks=tasks,
            tokenizer_pretrained_path=tokenizer_pretrained_path,
            transforms=transforms,
            return_tensors=return_tensors,
            collate_fn=collate_fn,
        )

        self.tasks: tuple[dict[str, Any], ...]

    @staticmethod
    @abstractmethod
    def load_tasks(
        json_path: StrOrPath,
        filter_tasks: bool,
    ) -> tuple[dict[str, Any], ...]:
        """Load tasks from a json file and filter out tasks that have phrase length less than 2.
        Also, exclude images with ids in invalid_img_ids (copied from clipseg).

        Args:
        ----
            json_path: The path to the json file containing task_id and phrase.
            filter_tasks: Whether to filter tasks using length and img_ids.

        Returns:
        -------
            The filtered tasks.

        """
        ...

    @staticmethod
    @abstractmethod
    def get_phrase2image_ids(tasks: Iterable[JSONMapping]) -> Str2SetInt:
        """Get a mapping of phrase to a set of image ids.

        Args:
        ----
            tasks: The tasks to get the mapping from.

        Returns:
        -------
            A mapping of phrase to a set of image ids.

        """
        ...

    @staticmethod
    def get_prompt_list(prompt_method: PromptMethodType) -> tuple[str, ...]:
        """Get a collection of prompt formats based on prompt_method.

        Args:
        ----
            prompt_method: The prompt method.

        Returns:
        -------
            The list of prompt formats.

        """
        prompt_format_list = ["a photo of {}."]

        # For shuffle and shuffle+
        if prompt_method != "fixed":
            prompt_format_list.extend(
                (
                    "a photograph of {}.",
                    "a picture of {}.",
                    "an image of {}.",
                    "{}.",
                ),
            )

        if prompt_method == "shuffle+":
            prompt_format_list.extend(
                (
                    "a cropped photo of {}.",
                    "a good photo of {}.",
                    "a bad photo of {}.",
                    "a cropped photograph of {}.",
                    "a good photograph of {}.",
                    "a bad photograph of {}.",
                    "a cropped image of {}.",
                    "a good image of {}.",
                    "a bad image of {}.",
                    "a cropped snap of {}.",
                    "a good snap of {}.",
                    "a bad snap of {}.",
                ),
            )

        return tuple(prompt_format_list)

    @staticmethod
    @abstractmethod
    def get_mask_name(task: JSONMapping) -> str:
        """Get mask name from task.

        Args:
        ----
            task: The task to extract mask name from.

        Returns:
        -------
            The mask name.

        """
        ...

    @abstractmethod
    def get_image_id_image_path(self, task: JSONMapping) -> tuple[int, Path]:
        """Get image id and image path from task.

        Args:
        ----
            task: The task to extract image id and image path from.

        Returns:
        -------
            The image id and image path.

        """
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task = self.tasks[idx]

        image_id, image_path = self.get_image_id_image_path(task)

        image = self.load_image(
            path=image_path,
            imread_flags=cv2.IMREAD_COLOR,
            cvtColor_code=cv2.COLOR_BGR2RGB,
        )

        phrase = str(task["phrase"])
        new_phrase_or_none = self.get_neg_phrase(
            curr_phrase=phrase,
            curr_image_id=image_id,
        )

        # Needed for metadata
        mask_shape = image.shape[:-1]
        mask_name = self.get_mask_name(task)

        if new_phrase_or_none is not None:
            # Replace phrase and make mask zeros
            phrase = new_phrase_or_none
            mask = np.zeros(mask_shape, np.float32)
        else:
            # If new_phrase was not assiged, load mask from the disk
            mask = (
                self.load_image(
                    self.mask_dir / mask_name,
                    cv2.IMREAD_GRAYSCALE,
                ).astype(np.float32)
                / 255
            )

        # Add the final channel layer to mask
        mask = mask[..., None]

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        prompt_format = random.choice(self.prompt_format_choices)

        prompt = prompt_format.format(phrase)

        text_inputs = self.get_text_output(prompt)

        # Metadata are needed to save the image for the predict step
        return {
            "image": image,
            "mask": mask,
            "mask_shape": np.array(mask_shape),
            "mask_name": mask_name,
            **text_inputs,
        }

    def get_neg_phrase(self, curr_phrase: str, curr_image_id: int) -> str | None:
        """Get a new phrase that is not the current phrase and is not present in the dataset for the same image.
        The corresponding mask for this phrase would be all zeros.

        Args:
        ----
            curr_phrase: The corresponding phrase for the current task.
            curr_image_id: The id of the image for the current task.

        Returns:
        -------
            New phrase or None if no new phrase is sampled.

        """
        # Use zero mask if neg_prob is greater than 1
        # Do not use mask if the neg_prob is less than 0
        if self.neg_prob >= 1 or (
            self.neg_prob > 0 and random.random() < self.neg_prob
        ):
            for _ in range(self.neg_sample_tries):
                new_phrase = random.choice(self.unique_phrases)

                # Can not sample the same phrase
                if new_phrase == curr_phrase:
                    continue

                new_image_ids = self.phrase2image_ids[new_phrase]

                # Do not get the phrase if it is present in the dataset for the same image
                if curr_image_id not in new_image_ids:
                    return new_phrase
        return None

    @staticmethod
    def plot_img_mask_cut(
        img: np.ndarray,
        phrase: str,
        mask: np.ndarray,
        figsize: tuple[int, int] = (15, 5),
    ) -> None:
        """Plot an image with mask and cut of the image where the mask is.

        Args:
        ----
            img: The image to be plotted.
            phrase: The phrase corresponding to the mask.
            mask: The mask to be plotted.
            figsize: The figure size.

        """
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        ax1.imshow(__class__.img_normalize(img))
        ax1.set_title("Image")

        mask = mask.astype(bool)

        ax2.imshow(mask)
        ax2.set_title(f"Mask: {phrase}")

        # Darken the image where the mask is
        img_cut = img.copy().astype(np.float32)

        alpha = 0.7

        img_cut[mask] *= 1 - alpha

        img_cut = __class__.img_normalize(img_cut)

        ax3.imshow(img_cut)
        ax3.set_title("Image with Mask")

    @staticmethod
    def img_normalize(img: ImageLike) -> ImageLike:
        """Normalize the image between 0 and 1 by dividing the image by its range.

        Args:
        ----
            img: The image to be normalized.

        Returns:
        -------
            The normalized image.

        """
        img_min: ImageLike = img.min()  # type: ignore
        img_max: ImageLike = img.max()  # type: ignore
        return (img - img_min) / (img_max - img_min)
