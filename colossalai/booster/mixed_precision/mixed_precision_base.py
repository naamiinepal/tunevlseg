from abc import ABC, abstractmethod
from collections.abc import Callable

import torch.nn as nn
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper


class MixedPrecision(ABC):
    """
    An abstract class for mixed precision training.
    """

    @abstractmethod
    def configure(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        criterion: Callable | None = None,
    ) -> tuple[nn.Module, OptimizerWrapper, Callable]:
        # TODO: implement this method
        pass
