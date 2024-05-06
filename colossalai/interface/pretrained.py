from torch.nn import Module

__all__ = [
    "get_pretrained_path",
    "set_pretrained_path",
]


def get_pretrained_path(model: Module) -> str | None:
    return getattr(model, "_pretrained", None)


def set_pretrained_path(model: Module, path: str) -> None:
    model._pretrained = path
