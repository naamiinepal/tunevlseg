from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

from . import OpenDomainBaseDataset

if TYPE_CHECKING:
    from collections.abc import Iterable

    from . import JSONMapping, StrOrPath

# {
#   "image_id": 380440,
#   "image_name": "COCO_train2014_000000380440.jpg",
#   "ann_id": 491042,
#   "sent_id": 8,
#   "phrase": "the man in yellow coat"
# }


class RefCOCODataset(OpenDomainBaseDataset):
    @staticmethod
    def load_tasks(json_path: StrOrPath, filter_tasks: bool):
        with open(json_path, encoding="locale") as f:
            tasks = json.load(f)

        if filter_tasks:
            tasks = (task for task in tasks if len(task["phrase"]) > 1)

        return tuple(tasks)

    @staticmethod
    def get_phrase2image_ids(tasks: Iterable[JSONMapping]):
        # Map phrase to a list of images
        phrase2image_ids: defaultdict[str, list[int]] = defaultdict(list)

        for task in tasks:
            phrase: str = task["phrase"]  # type: ignore
            image_id: int = task["image_id"]  # type: ignore

            phrase2image_ids[phrase].append(image_id)

        return {k: set(v) for k, v in phrase2image_ids.items()}

    def get_image_id_image_path(self, task: JSONMapping):
        image_id: int = task["image_id"]

        image_name: str = task["image_name"]
        image_path = self.image_dir / image_name

        return image_id, image_path

    @staticmethod
    def get_mask_name(task: JSONMapping) -> str:
        image_id = task["image_id"]
        ann_id = task["ann_id"]
        sent_id = task["sent_id"]
        return f"{image_id}-{ann_id}-{sent_id}.png"
