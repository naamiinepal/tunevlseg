from .base_visual_learner import BaseVisualLearner
from .coop_context_learner import CoOpContextLearner


class BaseSharedLearner(CoOpContextLearner, BaseVisualLearner):
    def __init__(self, **kwargs):
        kwargs["context_initializer"] = None
        kwargs["tokenizer"] = None
        kwargs["embedding_layer"] = None

        super().__init__(**kwargs)
