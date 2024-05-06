# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections.abc import Callable

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from mmengine.device import get_device
from mmengine.dist import init_dist, is_distributed, master_only
from mmengine.model import convert_sync_batchnorm, is_model_wrapper
from mmengine.registry import MODEL_WRAPPERS, STRATEGIES

from .single_device import SingleDeviceStrategy


@STRATEGIES.register_module()
class DDPStrategy(SingleDeviceStrategy):
    """Distribution strategy for distributed data parallel training.

    Args:
        model_wrapper (dict): Dict for model wrapper. Defaults to None.
        sync_bn (str): Type of sync batch norm. Defaults to None.
            Options are 'torch' and 'mmcv'.
        **kwargs: Other arguments for :class:`BaseStrategy`.
    """

    def __init__(
        self,
        *,
        model_wrapper: dict | None = None,
        sync_bn: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_wrapper = model_wrapper
        self.sync_bn = sync_bn

    def _setup_distributed(  # type: ignore
        self,
        launcher: str = "pytorch",
        backend: str = "nccl",
        **kwargs,
    ):
        """Setup distributed environment.

        Args:
            launcher (str): Way to launcher multi processes. Supported
                launchers are 'pytorch', 'mpi' and 'slurm'.
            backend (str): Communication Backends. Supported backends are
                'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
            **kwargs: Other arguments for :func:`init_dist`.
        """
        if not is_distributed():
            init_dist(launcher, backend, **kwargs)

    def convert_model(self, model: nn.Module) -> nn.Module:
        """convert all ``BatchNorm`` layers in the model to ``SyncBatchNorm``
        (SyncBN) or ``mmcv.ops.sync_bn.SyncBatchNorm`` (MMSyncBN) layers.

        Args:
            model (nn.Module): Model to be converted.

        Returns:
            nn.Module: Converted model.
        """
        if self.sync_bn is not None:
            try:
                model = convert_sync_batchnorm(model, self.sync_bn)
            except ValueError as e:
                self.logger.error(
                    'cfg.sync_bn should be "torch" or '
                    f'"mmcv", but got {self.sync_bn}'
                )
                raise e

        return model

    def _wrap_model(self, model: nn.Module) -> DistributedDataParallel:
        """Wrap the model to :obj:``MMDistributedDataParallel`` or other custom
        distributed data-parallel module wrappers.

        Args:
            model (nn.Module): Model to be wrapped.

        Returns:
            nn.Module or DistributedDataParallel: nn.Module or subclass of
            ``DistributedDataParallel``.
        """
        if is_model_wrapper(model):
            return model

        model = model.to(get_device())

        model = self.convert_model(model)

        if self.model_wrapper is None:
            # set broadcast_buffers as False to keep compatibility with
            # OpenMMLab repos
            self.model_wrapper = {
                "type": "MMDistributedDataParallel",
                "broadcast_buffers": False,
            }

        default_args = {
            "type": "MMDistributedDataParallel",
            "module": model,
            "device_ids": [int(os.environ["LOCAL_RANK"])],
        }
        return MODEL_WRAPPERS.build(self.model_wrapper, default_args=default_args)

    @master_only
    def save_checkpoint(
        self,
        filename: str,
        *,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        extra_ckpt: dict | None = None,
        callback: Callable | None = None,
    ) -> None:
        super().save_checkpoint(
            filename=filename,
            save_optimizer=save_optimizer,
            save_param_scheduler=save_param_scheduler,
            extra_ckpt=extra_ckpt,
            callback=callback,
        )
