# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import shutil
import tempfile
import time
from unittest import TestCase
from uuid import uuid4

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group
from torch.utils.data import Dataset

import mmengine.hooks  # F401
import mmengine.optim  # noqa F401
from mmengine.config import Config
from mmengine.dist import is_distributed
from mmengine.evaluator import BaseMetric
from mmengine.logging import MessageHub, MMLogger
from mmengine.model import BaseModel
from mmengine.registry import DATASETS, METRICS, MODELS, DefaultScope
from mmengine.runner import Runner
from mmengine.visualization import Visualizer


class ToyModel(BaseModel):
    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_samples, list):
            data_samples = torch.stack(data_samples)
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)

        if mode == "tensor":
            return outputs
        if mode == "loss":
            loss = (data_samples - outputs).sum()
            return {"loss": loss}
        if mode == "predict":
            return outputs
        return None


class ToyDataset(Dataset):
    METAINFO = {}  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return {"inputs": self.data[index], "data_samples": self.label[index]}


class ToyMetric(BaseMetric):
    def __init__(self, collect_device="cpu", dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {"acc": 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return {"acc": 1}


class RunnerTestCase(TestCase):
    """A test case to build runner easily.

    `RunnerTestCase` will do the following things:

    1. Registers a toy model, a toy metric, and a toy dataset, which can be
       used to run the `Runner` successfully.
    2. Provides epoch based and iteration based cfg to build runner.
    3. Provides `build_runner` method to build runner easily.
    4. Clean the global variable used by the runner.
    """

    dist_cfg = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": 29600,
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
    }

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        # Prevent from registering module with the same name by other unit
        # test. These registries will be cleared in `tearDown`
        MODELS.register_module(module=ToyModel, force=True)
        METRICS.register_module(module=ToyMetric, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        epoch_based_cfg = {
            "work_dir": self.temp_dir.name,
            "model": {"type": "ToyModel"},
            "train_dataloader": {
                "dataset": {"type": "ToyDataset"},
                "sampler": {"type": "DefaultSampler", "shuffle": True},
                "batch_size": 3,
                "num_workers": 0,
            },
            "val_dataloader": {
                "dataset": {"type": "ToyDataset"},
                "sampler": {"type": "DefaultSampler", "shuffle": False},
                "batch_size": 3,
                "num_workers": 0,
            },
            "val_evaluator": [{"type": "ToyMetric"}],
            "test_dataloader": {
                "dataset": {"type": "ToyDataset"},
                "sampler": {"type": "DefaultSampler", "shuffle": False},
                "batch_size": 3,
                "num_workers": 0,
            },
            "test_evaluator": [{"type": "ToyMetric"}],
            "optim_wrapper": {"optimizer": {"type": "SGD", "lr": 0.1}},
            "train_cfg": {"by_epoch": True, "max_epochs": 2, "val_interval": 1},
            "val_cfg": {},
            "test_cfg": {},
            "default_hooks": {"logger": {"type": "LoggerHook", "interval": 1}},
            "custom_hooks": [],
            "env_cfg": {"dist_cfg": {"backend": "nccl"}},
            "experiment_name": "test1",
        }
        self.epoch_based_cfg = Config(epoch_based_cfg)

        # prepare iter based cfg.
        self.iter_based_cfg: Config = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = {
            "dataset": {"type": "ToyDataset"},
            "sampler": {"type": "InfiniteSampler", "shuffle": True},
            "batch_size": 3,
            "num_workers": 0,
        }
        self.iter_based_cfg.log_processor = {"by_epoch": False}

        self.iter_based_cfg.train_cfg = {"by_epoch": False, "max_iters": 12}
        self.iter_based_cfg.default_hooks = {
            "logger": {"type": "LoggerHook", "interval": 1},
            "checkpoint": {"type": "CheckpointHook", "interval": 12, "by_epoch": False},
        }

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        Visualizer._instance_dict.clear()
        DefaultScope._instance_dict.clear()
        MessageHub._instance_dict.clear()
        MODELS.module_dict.pop("ToyModel", None)
        METRICS.module_dict.pop("ToyMetric", None)
        DATASETS.module_dict.pop("ToyDataset", None)
        self.temp_dir.cleanup()
        if is_distributed():
            destroy_process_group()

    def build_runner(self, cfg: Config):
        cfg.experiment_name = self.experiment_name
        return Runner.from_cfg(cfg)

    @property
    def experiment_name(self):
        # Since runners could be built too fast to have a unique experiment
        # name(timestamp is the same), here we use uuid to make sure each
        # runner has the unique experiment name.
        return f"{self._testMethodName}_{time.time()} + " f"{uuid4()}"

    def setup_dist_env(self):
        self.dist_cfg["MASTER_PORT"] += 1
        os.environ["MASTER_PORT"] = str(self.dist_cfg["MASTER_PORT"])
        os.environ["MASTER_ADDR"] = self.dist_cfg["MASTER_ADDR"]
        os.environ["RANK"] = self.dist_cfg["RANK"]
        os.environ["WORLD_SIZE"] = self.dist_cfg["WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = self.dist_cfg["LOCAL_RANK"]

    def clear_work_dir(self):
        logging.shutdown()
        for filename in os.listdir(self.temp_dir.name):
            filepath = os.path.join(self.temp_dir.name, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath)
