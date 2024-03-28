from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import OpenDomainBaseDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.utils.data.dataloader import _collate_fn_t

    from . import JSONMapping, PromptMethodType, ReturnTensorsType, StrOrPath


class PhraseCutDataset(OpenDomainBaseDataset):
    def __init__(
        self,
        data_root: StrOrPath,
        task_json_path: StrOrPath,
        tokenizer_pretrained_path: StrOrPath,
        image_dir: StrOrPath = "images",
        mask_dir: StrOrPath = "masks",
        transforms: Callable | None = None,
        return_tensors: ReturnTensorsType = None,
        prompt_method: PromptMethodType = "fixed",
        neg_prob: float = 0,
        neg_sample_tries: int = 1000,
        filter_tasks: bool = False,
        collate_fn: _collate_fn_t[JSONMapping] | None = None,
    ) -> None:
        task_json_path = Path(data_root, task_json_path)
        image_dir = Path(data_root, image_dir)
        mask_dir = Path(data_root, mask_dir)
        super().__init__(
            task_json_path=task_json_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
            tokenizer_pretrained_path=tokenizer_pretrained_path,
            transforms=transforms,
            return_tensors=return_tensors,
            prompt_method=prompt_method,
            neg_prob=neg_prob,
            neg_sample_tries=neg_sample_tries,
            filter_tasks=filter_tasks,
            collate_fn=collate_fn,
        )

    @staticmethod
    def load_tasks(json_path: StrOrPath, filter_tasks: bool):
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
        with open(json_path, encoding="locale") as f:
            tasks: list[dict[str, Any]] = json.load(f)

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

    @staticmethod
    def get_phrase2image_ids(tasks: Iterable[JSONMapping]):
        # Map phrase to a list of images
        phrase2image_ids: defaultdict[str, list[int]] = defaultdict(list)

        for task in tasks:
            phrase = task["phrase"]
            task_id = task["task_id"]

            image_id = __class__.get_image_id_from_task_id(task_id)

            phrase2image_ids[phrase].append(image_id)

        return {k: set(v) for k, v in phrase2image_ids.items()}

    @staticmethod
    def get_image_id_from_task_id(task_id: str):
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

    def get_image_id_image_path(self, task: JSONMapping):
        task_id: str = task["task_id"]

        image_id = self.get_image_id_from_task_id(task_id)

        image_path = self.image_dir / f"{image_id}.jpg"

        return image_id, image_path

    @staticmethod
    def get_mask_name(task: JSONMapping) -> str:
        phrase: str = task["phrase"]
        safe_phrase = phrase.replace("/", "\\")

        task_id = task["task_id"]
        return f"{task_id}-{safe_phrase}.png"
