# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from mmengine.structures import InstanceData


class BaseAssigner(ABC):
    """Base assigner that assigns masks to ground truth class labels."""

    @abstractmethod
    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs,
    ):
        """Assign masks to either a ground truth class label or a negative
        label."""
