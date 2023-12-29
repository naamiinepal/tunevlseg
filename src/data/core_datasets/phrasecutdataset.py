# pyright: reportGeneralTypeIssues=false
import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Optional, Tuple, Union

import cv2
import cv2.typing
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer

StrPath = Union[str, Path]

PolygonType = Iterable[Iterable[Iterable[Tuple[int, int]]]]


class PhraseCutDataset(Dataset):
    def __init__(
        self,
        data_root: StrPath,
        task_json_path: StrPath,
        tokenizer_pretrained_path: StrPath,
        image_dir: StrPath = "images",
        transforms: Optional[Callable] = None,
        return_tensors: Literal["tf", "pt", "np"] = "np",
    ) -> None:
        super().__init__()

        data_root = Path(data_root)

        with (data_root / task_json_path).open() as f:
            self.tasks: Tuple[Mapping[str, Union[str, PolygonType]], ...] = tuple(
                json.load(f),
            )

        self.image_path = data_root / image_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_path)
        self.return_tensors = return_tensors

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int):
        task = self.tasks[idx]

        image = self.load_image(
            self.image_path / f"{task['image_id']}.jpg",
            cv2.IMREAD_COLOR,
        )
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        text_output = self.tokenizer(
            task["phrase"],
            truncation=True,
            padding="max_length",
            return_tensors=self.return_tensors,
        )

        mask = self.polygon_to_mat(image.shape[:-1], task["Polygons"])

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {"image": image, "mask": mask, **text_output}

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
