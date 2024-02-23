from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class BaseCLIP(nn.Module, ABC):
    model: object
    image_size: int
    patch_size: int

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
