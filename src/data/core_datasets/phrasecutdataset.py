from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch
    from cv2.typing import MatLike

    StrPath = str | Path

    TaskType = Mapping[Literal["task_id", "phrase"], str]
    PromptMethodType = Literal["fixed", "shuffle", "shuffle+"]
    Str2SetInt = Mapping[str, set[int]]

    ImageLike = TypeVar("ImageLike", np.ndarray, torch.Tensor)


class PhraseCutDataset(Dataset):
    def __init__(
        self,
        data_root: StrPath,
        task_json_path: StrPath,
        tokenizer_pretrained_path: StrPath,
        image_dir: StrPath = "images",
        mask_dir: StrPath = "masks",
        transforms: Callable | None = None,
        return_tensors: Literal["tf", "pt", "np"] | None = None,
        prompt_method: PromptMethodType = "fixed",
        neg_prob: float = 0,
        neg_sample_tries: int = 1000,
        filter_tasks: bool = False,
    ) -> None:
        super().__init__()

        self.tasks = self.load_tasks(Path(data_root, task_json_path), filter_tasks)

        # Needed only for negative sampling and is expensive
        if neg_prob > 0:
            phrase2image_ids = self.get_phrase2image_ids()
            self.unique_phrases = tuple(phrase2image_ids)
            self.phrase2image_ids = phrase2image_ids
        else:
            # For the sake of types, keep them empty
            self.phrase2image_ids: Str2SetInt = {}
            self.unique_phrases: tuple[str, ...] = ()

        self.image_path = Path(data_root, image_dir)
        self.mask_path = Path(data_root, mask_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_path)
        self.return_tensors = return_tensors

        self.transforms = transforms

        self.neg_prob = neg_prob
        self.neg_sample_tries = neg_sample_tries

        self.prompt_format_choices = self.get_prompt_list(prompt_method)

    @staticmethod
    def load_tasks(json_path: Path, filter_tasks: bool) -> tuple[TaskType, ...]:
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
        with json_path.open() as f:
            tasks = json.load(f)

        if not filter_tasks:
            return tuple(tasks)

        # img_ids where the size in the annotations does not match actual size
        # **copied from clipseg**
        invalid_img_ids = (
            61530,
            61564,
            150333,
            150344,
            150417,
            150516,
            285665,
            285743,
            285761,
            285814,
            286065,
            286093,
            498010,
            498042,
            498187,
            498246,
            498269,
        )

        return tuple(
            task
            for task in tasks
            if len(task["phrase"]) > 1
            and PhraseCutDataset.get_image_id_from_task_id(task["task_id"])
            not in invalid_img_ids
        )

    def get_phrase2image_ids(self) -> Str2SetInt:
        """Get a mapping of phrase to a set of image ids.

        Returns
        -------
            A mapping of phrase to a set of image ids.
        """
        # Map phrase to a list of images
        phrase2image_ids: defaultdict[str, list[int]] = defaultdict(list)

        for task in self.tasks:
            phrase = task["phrase"]
            task_id = task["task_id"]

            image_id = self.get_image_id_from_task_id(task_id)

            phrase2image_ids[phrase].append(image_id)

        return {k: set(v) for k, v in phrase2image_ids.items()}

    @staticmethod
    def get_image_id_from_task_id(task_id: str) -> int:
        """Get image id from task id.

        Args:
        ----
            task_id: The task id.

        Returns:
        -------
            Image id.
        """
        img_id, _ = task_id.split("__", 1)
        return int(img_id)

    @staticmethod
    def get_prompt_list(prompt_method: PromptMethodType) -> tuple[str, ...]:
        """Get a collecttion of prompt formats based on prompt_method.

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

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task = self.tasks[idx]

        task_id = task["task_id"]

        image_id = self.get_image_id_from_task_id(task_id)

        image = self.load_image(self.image_path / f"{image_id}.jpg", cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        phrase = task["phrase"]
        new_phrase_or_none = self.get_neg_phrase(
            curr_phrase=phrase,
            curr_image_id=image_id,
        )

        mask_shape = image.shape[:-1]

        safe_phrase = phrase.replace("/", "\\")

        mask_name = f"{task_id}-{safe_phrase}.png"

        if new_phrase_or_none is not None:
            # Replace phrase and make mask zeros
            phrase = new_phrase_or_none
            mask = np.zeros(mask_shape, np.float32)
        else:
            # If new_phrase was not assiged, load mask from the disk
            mask = (
                self.load_image(
                    self.mask_path / mask_name,
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

        text_inputs = self.get_text_output(phrase)

        # Metadata are needed to save the image for the predict step
        return {
            "image": image,
            "mask": mask,
            "mask_shape": np.array(mask_shape),
            "mask_name": mask_name,
            **text_inputs,
        }

    def get_text_output(
        self,
        phrase: str,
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
        prompt_format = random.choice(self.prompt_format_choices)

        prompt = prompt_format.format(phrase)

        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            return_tensors=self.return_tensors,
        )

        # Remove the first (batch) dimension if self.return_tensors is not None
        if self.return_tensors is None:
            return text_inputs

        return {k: v[0] for k, v in text_inputs.items()}

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

        ax1.imshow(PhraseCutDataset.img_normalize(img))
        ax1.set_title("Image")

        mask = mask.astype(bool)

        ax2.imshow(mask)
        ax2.set_title(f"Mask: {phrase}")

        # Darken the image where the mask is
        img_cut = img.copy().astype(np.float32)

        alpha = 0.7

        img_cut[mask] *= 1 - alpha

        img_cut = PhraseCutDataset.img_normalize(img_cut)

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
