from .bloom import BloomModelInferPolicy
from .chatglm2 import ChatGLM2InferPolicy
from .llama import LlamaModelInferPolicy

model_policy_map = {
    "llama": LlamaModelInferPolicy,
    "bloom": BloomModelInferPolicy,
    "chatglm": ChatGLM2InferPolicy,
}

__all__ = [
    "BloomModelInferPolicy",
    "ChatGLM2InferPolicy",
    "LlamaModelInferPolicy",
    "model_polic_map",
]
