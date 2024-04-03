from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import cv2
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from os import PathLike

    import torch
    from cv2.typing import MatLike
    from torch.utils.data.dataloader import _collate_fn_t

    NDArray = MatLike | torch.Tensor

    StrOrPath = str | PathLike
    TransformsType = Callable[..., Mapping[str, NDArray]] | None
    ReturnTensorsType = Literal["tf", "pt", "np"] | None
    CollateFnType = _collate_fn_t[Mapping[str, Any]] | None


class BaseImageTextMaskDataset(Dataset, ABC):
    def __init__(
        self,
        tasks: Sequence,
        tokenizer_pretrained_path: StrOrPath,
        transforms: TransformsType,
        return_tensors: ReturnTensorsType,
        collate_fn: CollateFnType,
        *tokenizer_loader_args,
        **tokenizer_loader_kwargs,
    ) -> None:
        self.tasks = tasks

        self.tokenizer = self.get_pretrained_tokenizer(
            tokenizer_pretrained_path,
            *tokenizer_loader_args,
            **tokenizer_loader_kwargs,
        )

        self.transforms = transforms

        self.return_tensors = return_tensors
        self.collate_fn = collate_fn

    @staticmethod
    def get_pretrained_tokenizer(
        pretrained_model_name_or_path: StrOrPath,
        *args,
        **kwargs,
    ) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            msg = (
                "Expected `tokenizer` to be an instance of "
                f"`PreTrainedTokenizerBase` but got: {type(tokenizer)} instead."
            )
            raise TypeError(msg)

        return tokenizer

    @staticmethod
    def load_image(
        path: StrOrPath,
        imread_flags: int = cv2.IMREAD_UNCHANGED,
        cvtColor_code: int | None = None,
    ) -> MatLike:
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
        img = cv2.imread(str(path), imread_flags)

        if img is None:
            msg = f"Image not found in the path: {path}"
            raise ValueError(msg)

        if cvtColor_code is None:
            return img

        return cv2.cvtColor(img, cvtColor_code)

    def get_text_output(
        self,
        prompt: str | list[str],
        *args,
        **kwargs,
    ) -> Mapping[str, list[int] | NDArray]:
        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            *args,
            truncation=True,
            padding=True,
            return_tensors=self.return_tensors,
            return_attention_mask=True,  # Explicitly enable the attn_mask
            **kwargs,
        )

        # Remove the first (batch) dimension if self.return_tensors is not None
        if self.return_tensors is None:
            return text_inputs

        return {k: v[0] for k, v in text_inputs.items()}

    def __len__(self) -> int:
        return len(self.tasks)

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...
