import torch

from .base_projector_learner import BaseProjectorLearner


class MapleContextLearner(BaseProjectorLearner):
    def __init__(
        self,
        *,
        visual_dim: int,
        **kwargs,
    ) -> None:
        kwargs["proj_in_dim"] = self.context_dim
        kwargs["proj_out_dim"] = visual_dim

        super().__init__(**kwargs)

    def mutate_image_hidden_states(
        self, hidden_states: torch.Tensor, index: int
    ) -> None:
        hidden_states[:, -self.num_context :] = self.get_transformed_context(
            index=index
        ).expand(hidden_states.size(0), -1, -1)
