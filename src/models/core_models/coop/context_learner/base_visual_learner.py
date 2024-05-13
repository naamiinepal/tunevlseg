from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .base_unimodal_learner import BaseUnimodalLearner

if TYPE_CHECKING:
    import torch


class BaseVisualLearner(BaseUnimodalLearner):
    @abstractmethod
    def get_visual_context(
        self, in_context: torch.Tensor | None = None, index: int = 0
    ) -> torch.Tensor: ...

    def mutate_image_hidden_states(
        self, hidden_states: torch.Tensor, index: int
    ) -> torch.Tensor:
        hidden_states[:, -self.num_context :] = self.get_visual_context(index=index)

        return hidden_states
