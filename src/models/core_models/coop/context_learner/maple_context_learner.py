import torch

from .base_projector_learner import BaseProjectorLearner
from .base_visual_learner import BaseVisualLearner


class MapleContextLearner(BaseProjectorLearner, BaseVisualLearner):
    def __init__(
        self,
        *,
        visual_dim: int,
        **kwargs,
    ) -> None:
        kwargs["proj_in_dim"] = None  # Uses context dim if None is provided
        kwargs["proj_out_dim"] = visual_dim

        super().__init__(**kwargs)

    def get_visual_context(self, *args, **kwargs) -> torch.Tensor:
        return self.get_transformed_context(*args, **kwargs)
