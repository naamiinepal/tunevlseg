from typing import Any

from .image_text_mask_dataset import ImageTextMaskDataset


class ZeroShotDataset(ImageTextMaskDataset):
    def __init__(self, object_class: str, *args, **kwargs) -> None:
        super().__init__(*args, *kwargs)

        self.object_class = object_class

    def __getitem__(self, index: int) -> dict[str, Any]:
        ds_output = super().__getitem__(index)

        curr_prompt = ds_output["prompt"]

        if not isinstance(curr_prompt, str):
            msg = "The value for key: `prompt` must be a string for `ImageTextMaskDataset`."
            raise TypeError(msg)

        text_inputs = self.get_text_output([curr_prompt, self.object_class])

        return {**ds_output, **text_inputs}
