# pyright: reportGeneralTypeIssues=false
import json
import random
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Union

import cv2
from cv2.typing import MatLike
from torch.utils.data import Dataset
from transformers import AutoTokenizer

StrOrPath = Union[str, Path]
PromptType = Union[str, List[str]]
PromptMappingType = Mapping[str, PromptType]
TransformType = Callable[[MatLike, MatLike], Mapping[str, MatLike]]


class ImageTextDataset(Dataset):
    def __init__(
        self,
        img_root: StrOrPath,
        mask_root: StrOrPath,
        task_path: StrOrPath,
        pretrained_model_name_or_path: StrOrPath,
        prompt_index: int,
        transform: Optional[TransformType] = None,
    ) -> None:
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        self.transform = transform

        self.prompt_map_index = f"p{prompt_index}"

        with open(task_path) as fp:
            self.tasks: List[Mapping[str, Union[str, PromptMappingType]]] = json.load(
                fp
            )

    def __getitem__(self, index: int):
        task = self.tasks[index]

        img_name: str = task["img_name"]
        mask_name: str = task["mask_name"]
        prompts: PromptType = task["prompts"][self.prompt_map_index]

        img_path = self.img_root / img_name
        mask_path = self.mask_root / mask_name

        curr_prompt = prompts if isinstance(prompts, str) else random.choice(prompts)

        img = self.load_image(img_path)

        mask = self.load_image(mask_path)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        text = self.tokenizer(curr_prompt)

        # Also return the metadata to save the image
        return {
            "text": text,
            "img": img,
            "mask": mask,
            "img_name": img_name,
            "mask_name": mask_name,
            "img_shape": img.shape,
            "mask_shape": mask.shape,
        }

    @staticmethod
    def load_image(img_path: Path):
        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError("Image is not found in the directory: ", img_path)

        return img
