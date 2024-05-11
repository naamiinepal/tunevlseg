from __future__ import annotations

from typing import TYPE_CHECKING

from .base_multimodal_clipseg import BaseMultimodalCLIPSeg

if TYPE_CHECKING:
    from .context_learner import SharedSeparateLearner


class SharedSeparateCLIPSeg(BaseMultimodalCLIPSeg):
    def __init__(
        self, context_learner: type[SharedSeparateLearner], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.context_learner = context_learner(
            textual_dim=self.model.config.text_config.hidden_size,
            visual_dim=self.model.config.vision_config.hidden_size,
            max_network_depth=min(
                self.model.config.text_config.num_hidden_layers,
                self.model.config.vision_config.num_hidden_layers,
            ),
            context_dim=self.model.config.text_config.hidden_size,
            embedding_layer=self.model.clip.text_model.embeddings.token_embedding,
        )
