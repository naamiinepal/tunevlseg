import torch

from ..registry import meta_profiler_function


@meta_profiler_function.register(torch.nn.functional.instance_norm)
def torch_nn_func_instancenorm(
    input: torch.Tensor,
    running_mean: torch.Tensor | None = None,
    running_var: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.nn.functional.group_norm)
def torch_nn_func_groupnorm(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[int, int]:
    has_affine = weight is not None
    flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.nn.functional.layer_norm)
def torch_nn_func_layernorm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[int, int]:
    has_affine = weight is not None
    flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.nn.functional.batch_norm)
def torch_nn_func_batchnorm(
    input: torch.Tensor,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> tuple[int, int]:
    has_affine = weight is not None
    if training:
        flops = input.numel() * (2 if has_affine else 1)
    else:
        flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs
