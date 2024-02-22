# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass


@dataclass
class ShapeSpec:
    """A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: int | None = None
    height: int | None = None
    width: int | None = None
    stride: int | None = None
