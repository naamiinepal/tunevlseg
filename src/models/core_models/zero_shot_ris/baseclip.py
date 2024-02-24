from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class BaseCLIP(nn.Module, ABC):
    def __init__(self, model: Callable, image_size: int, patch_size: int) -> None:
        super().__init__()

        self.model = model
        self.image_size = image_size
        self.patch_size = patch_size

    @abstractmethod
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pred_masks: torch.Tensor | None = None,
        masking_block_idx: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Any = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        ...
