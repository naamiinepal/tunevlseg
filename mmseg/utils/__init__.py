# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .class_names import (
    ade_classes,
    ade_palette,
    bdd100k_classes,
    bdd100k_palette,
    cityscapes_classes,
    cityscapes_palette,
    cocostuff_classes,
    cocostuff_palette,
    dataset_aliases,
    get_classes,
    get_palette,
    isaid_classes,
    isaid_palette,
    loveda_classes,
    loveda_palette,
    potsdam_classes,
    potsdam_palette,
    stare_classes,
    stare_palette,
    synapse_classes,
    synapse_palette,
    vaihingen_classes,
    vaihingen_palette,
    voc_classes,
    voc_palette,
)

# yapf: enable
from .collect_env import collect_env
from .get_templates import get_predefined_templates
from .io import datafrombytes
from .misc import add_prefix, stack_batch
from .set_env import register_all_modules
from .tokenizer import tokenize
from .typing_utils import (
    ConfigType,
    ForwardResults,
    MultiConfig,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
    TensorDict,
    TensorList,
)

# isort: off
from .mask_classification import MatchMasks, seg_data_to_instance_data

__all__ = [
    "ConfigType",
    "ForwardResults",
    "MatchMasks",
    "MultiConfig",
    "OptConfigType",
    "OptMultiConfig",
    "OptSampleList",
    "SampleList",
    "TensorDict",
    "TensorList",
    "add_prefix",
    "ade_classes",
    "ade_palette",
    "bdd100k_classes",
    "bdd100k_palette",
    "cityscapes_classes",
    "cityscapes_palette",
    "cocostuff_classes",
    "cocostuff_palette",
    "collect_env",
    "datafrombytes",
    "dataset_aliases",
    "get_classes",
    "get_palette",
    "get_predefined_templates",
    "isaid_classes",
    "isaid_palette",
    "loveda_classes",
    "loveda_palette",
    "potsdam_classes",
    "potsdam_palette",
    "register_all_modules",
    "seg_data_to_instance_data",
    "stack_batch",
    "stare_classes",
    "stare_palette",
    "synapse_classes",
    "synapse_palette",
    "tokenize",
    "vaihingen_classes",
    "vaihingen_palette",
    "voc_classes",
    "voc_palette",
]
