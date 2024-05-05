from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import rich
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.tree import Tree

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

if TYPE_CHECKING:
    from collections.abc import Sequence


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    STYLE = "dim"

    tree = Tree("CONFIG", style=STYLE, guide_style=STYLE)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing...",
        )

    # add all the other fields to queue (not specified in `print_order`)
    queue.extend(field for field in cfg if field not in queue)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=STYLE, guide_style=STYLE)

        config_group = cfg[field]

        branch_content = (
            OmegaConf.to_yaml(config_group, resolve=resolve)
            if isinstance(config_group, DictConfig)
            else str(config_group)
        )

        branch.add(Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with Path(cfg.paths.output_dir, "config_tree.log").open("w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        hydra_config_cfg = HydraConfig().cfg
        if hydra_config_cfg is not None and "id" in hydra_config_cfg.hydra.job:  # type:ignore
            msg = "Specify tags before launching a multirun!"
            raise ValueError(msg)

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [_t for _t in (t.strip() for t in tags.split(",")) if _t]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with Path(cfg.paths.output_dir, "tags.log").open("w") as file:
            rich.print(cfg.tags, file=file)
