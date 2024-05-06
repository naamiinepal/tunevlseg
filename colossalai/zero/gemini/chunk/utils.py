from time import time

import torch
import torch.distributed as dist
import torch.nn as nn

from .manager import ChunkManager
from .search_utils import search_chunk_configuration


def safe_div(a, b):
    if a == 0:
        return 0
    return a / b


def init_chunk_manager(
    model: nn.Module,
    init_device: torch.device | None = None,
    hidden_dim: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> ChunkManager:
    if hidden_dim:
        search_interval = hidden_dim
    else:
        search_interval = 1024  # defaults to 1024
    kwargs["search_interval"] = search_interval

    dist.barrier()
    begin = time()

    config_dict, total_size, wasted_size = search_chunk_configuration(model, **kwargs)

    dist.barrier()
    end = time()
    span_s = end - begin
    mega_unit = 1024**2
    total_size /= mega_unit
    wasted_size /= mega_unit

    if verbose and dist.get_rank() == 0:
        print(
            f"searching chunk configuration is completed in {span_s:.2f} s.\n",
            f"used number: {total_size:.2f} * 2^20, wasted number: {wasted_size:.2f} * 2^20\n",
            f"total wasted percentage is {100 * safe_div(wasted_size, total_size + wasted_size):.2f}%",
            sep="",
            flush=True,
        )
    dist.barrier()

    return ChunkManager(config_dict, init_device)
