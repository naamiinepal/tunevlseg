# -------------------------------------------------------------------------
# Copyright (c) 2019 the AdelaiDet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn

if TYPE_CHECKING:
    import numpy as np

    NDTensor = TypeVar("NDTensor", torch.Tensor, np.ndarray)


def dice_loss(input: torch.Tensor, target: torch.Tensor):
    input = input.reshape(input.size(0), -1)
    target = target.reshape(target.size(0), -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d


def reduce_loss(loss: NDTensor, reduction: str) -> NDTensor:
    """Reduce loss as specified.

    Args:
    ----
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
    ------
        Tensor: Reduced loss tensor.

    """
    if reduction == "sum":
        return loss.sum()  # type:ignore
    if reduction.endswith("mean"):
        return loss.mean()  # type:ignore
    return loss


def weight_reduce_loss(
    loss: NDTensor,
    weight: float | NDTensor | None = None,
    reduction: str = "mean",
    avg_factor: float | NDTensor | None = None,
):
    """Apply element-wise weight and reduce loss.

    Args:
    ----
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
    -------
        Tensor: Processed loss values.

    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss *= weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        return reduce_loss(loss, reduction)
    if reduction == "sum":
        msg = 'avg_factor can not be used with reduction="sum"'
        raise ValueError(msg)
    if reduction == "mean":
        return loss.sum() / avg_factor  # type:ignore numpy returns the sum as Any
    # if reduction is 'none', then do nothing
    return loss


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: float | torch.Tensor | None = None,
):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = sigmoid_focal_loss_jit(pred, target, gamma=gamma, alpha=alpha)
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@dataclass
class FocalLoss(nn.Module):
    gamma: float = 2
    alpha: float = 0.25
    reduction: str = "mean"
    loss_weight: float | torch.Tensor = 1.0

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in {None, "none", "mean", "sum"}
        reduction = reduction_override or self.reduction
        return self.loss_weight * sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
        )
