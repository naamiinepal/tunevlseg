# pyright: reportGeneralTypeIssues=false
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Mapping, Optional, Tuple, Union

import cv2
import cv2.typing
import ijson
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer

StrPath = Union[str, Path]

PolygonType = Iterable[Iterable[Iterable[Tuple[int, int]]]]
TaskType = Mapping[str, Union[str, PolygonType]]
PromptMethodType = Literal["fixed", "shuffle", "shuffle+"]


class PhraseCutDataset(Dataset):
    def __init__(
        self,
        data_root: StrPath,
        task_json_path: StrPath,
        tokenizer_pretrained_path: StrPath,
        image_dir: StrPath = "images",
        transforms: Optional[Callable] = None,
        return_tensors: Optional[Literal["tf", "pt", "np"]] = None,
        prompt_method: PromptMethodType = "fixed",
        neg_prob: float = 0.0,
        neg_sample_tries: int = 1000,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)

        self.tasks = self.get_tasks(data_root / task_json_path)

        self.phrase2image_ids = self.get_phrase2image_ids()
        self.unique_phrases = tuple(self.phrase2image_ids)

        self.image_path = data_root / image_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_path)
        self.return_tensors = return_tensors

        self.transforms = transforms

        self.neg_prob = neg_prob
        self.neg_sample_tries = neg_sample_tries

        self.prompt_format_choices = self.get_prompt_list(prompt_method)

    def get_phrase2image_ids(self):
        # Map phrase to a list of images
        phrase2image_ids: Mapping[str, List[str]] = defaultdict(list)

        for task in self.tasks:
            phrase = task["phrase"]
            image_id = task["image_id"]

            phrase2image_ids[phrase].append(image_id)

        return {k: set(v) for k, v in phrase2image_ids.items()}

    @staticmethod
    def get_tasks(json_path: StrPath):
        task_list: List[TaskType] = []
        required_keys = ("phrase", "Polygons", "image_id")

        # Load json iteratively
        with open(json_path, "rb") as f:
            for item in ijson.items(f, "item", use_float=True):
                task: TaskType = {k: v for k, v in item.items() if k in required_keys}
                task_list.append(task)

        return tuple(task_list)

    @staticmethod
    def get_prompt_list(prompt_method: PromptMethodType):
        prompt_format_list = ["a photo of a {}."]

        # For shuffle and shuffle+
        if prompt_method != "fixed":
            prompt_format_list.extend(
                (
                    "a photograph of a {}.",
                    "an image of a {}.",
                    "{}.",
                )
            )

        if prompt_method == "shuffle+":
            prompt_format_list.extend(
                (
                    "a cropped photo of a {}.",
                    "a good photo of a {}.",
                    "a photo of one {}.",
                    "a bad photo of a {}.",
                    "a photo of the {}.",
                )
            )

        return tuple(prompt_format_list)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int):
        task = self.tasks[idx]

        image_id = task["image_id"]

        image = self.load_image(self.image_path / f"{image_id}.jpg", cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        phrase = task["phrase"]
        new_phrase_or_none = self.get_neg_phrase(
            curr_phrase=phrase, curr_image_id=image_id
        )

        mask_shape = image.shape[:-1]
        if new_phrase_or_none is not None:
            # Replace phrase and make mask zeros
            phrase = new_phrase_or_none
            mask = np.zeros((*mask_shape, 1), np.float32)
        else:
            # Get polygon from mask if new_phrase is not extracted
            mask = self.polygon_to_mat(mask_shape, task["Polygons"])

        # Add the final channel layer to mask
        mask = mask[..., None]

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        text_inputs = self.get_text_output(phrase)

        return {"image": image, "mask": mask, **text_inputs}

    def get_text_output(self, phrase: str):
        # Get a format randomly if prompt_format_list has more than one entries
        prompt_format = (
            self.prompt_format_choices[0]
            if len(self.prompt_format_choices)
            else random.choice(self.prompt_format_choices)
        )

        prompt = prompt_format.format(phrase)

        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            return_tensors=self.return_tensors,
        )

        # Remove the first (batch) dimension
        return {k: v[0] for k, v in text_inputs.items()}

    def get_neg_phrase(self, curr_phrase: str, curr_image_id: str) -> Optional[str]:
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
    def load_image(path: StrPath, flags: int = cv2.IMREAD_UNCHANGED):
        str_path = str(path)

        img = cv2.imread(str_path, flags)

        if img is None:
            msg = "Image not found in the path:"
            raise ValueError(msg, str_path)

        return img

    # @staticmethod
    # def polygon_to_pil(
    #     img_size: Tuple[int, int],
    #     polygons: PolygonType,
    # ):
    #     # Create mask from polygon
    #     mask = Image.new("1", img_size)
    #     img_draw = ImageDraw.Draw(mask)
    #
    #     # Loop to add multiple polygons to the mask
    #     for poly in polygons:
    #         for p in poly:
    #             int_p = np.around(p).astype(int)
    #             img_draw.polygon(int_p.flatten().tolist(), fill=1, outline=1)
    #
    #     return mask

    @staticmethod
    def polygon_to_mat(img_size: Tuple[int, int], polygons: PolygonType):
        if len(img_size) != 2:
            msg = "The image img_size be of two dimensions"
            raise ValueError(msg)

        # Create an empty mask for the polygon
        # Its type can be either float32 or uint8
        mask = np.zeros(img_size, np.float32)

        # Loop to add multiple polygons to the mask
        for poly in polygons:
            pts = [np.around(p).astype(np.int32) for p in poly]
            cv2.fillPoly(mask, pts, 1.0)

        return mask

    @staticmethod
    def plot_img_mask_cut(
        img: np.ndarray,
        phrase: str,
        mask: np.ndarray,
        figsize: Tuple[int, int] = (15, 5),
    ) -> None:
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
    def img_normalize(img: np.ndarray) -> np.ndarray:
        return (img - img.min()) / (img.max() - img.min())
