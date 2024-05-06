# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmflow."""

from collections.abc import Sequence
from typing import Optional, Union

import torch

from mmengine.config import ConfigDict
from mmseg.structures import SegDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, Sequence[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

SampleList = Sequence[SegDataSample]
OptSampleList = Optional[SampleList]

# Type hint of Tensor
TensorDict = dict[str, torch.Tensor]
TensorList = Sequence[torch.Tensor]

ForwardResults = Union[
    dict[str, torch.Tensor], list[SegDataSample], tuple[torch.Tensor], torch.Tensor
]
