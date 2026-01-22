##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Negative log-likelihood backward operator
##

# **************************************************************************** #
#                                                                              #
#                       NLLLOSS UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dlogp_i L = -1/N for the true class and 0 for all other classes.
# Upstream gradients are scaled by -factor and scattered to the target indices.


class NLLLossBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for NLL-style losses.
    # @details Accumulates gradients only on the provided target indices, scaled
    # by the factor encoding the averaging strategy.
    ##
    def __init__(self, target: np.ndarray, factor: float) -> None:
        self.target = target
        self.factor = factor

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        grad_value = -self.factor * out_grad
        grad_to_propagate = np.zeros_like(a.data, dtype=a.data.dtype)
        batch_indices = np.arange(a.data.shape[0])
        grad_to_propagate[batch_indices, self.target] = grad_value
        return grad_to_propagate
