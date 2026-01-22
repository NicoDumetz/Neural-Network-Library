##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Smoothed NLL Loss Backward
##
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor


class SmoothedNLLLossBackward(UnaryBackwardOp):
    def __init__(self, smooth_targets: np.ndarray, factor: float) -> None:
        self.smooth_targets = smooth_targets
        self.factor = factor

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        scalar_grad = float(np.array(out_grad, dtype=a.data.dtype).reshape(-1)[0])
        grad_weight = -self.factor * scalar_grad
        grad_to_propagate = grad_weight * self.smooth_targets
        return grad_to_propagate
