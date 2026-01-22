##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Log-softmax backward operator
##

# **************************************************************************** #
#                                                                              #
#                      LOG SOFTMAX UNARY BACKWARD OP                           #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# ∂ log_softmax_i = grad_i - softmax_i * Σ_j grad_j, implementing the
# Jacobian-vector product with cached softmax probabilities.

class LogSoftmaxBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for log-softmax.
    # @details Subtracts softmax * sum(grad) to implement Jacobian-vector rule.
    # Used by Tensor.log_softmax and NLL-like losses.
    # Relies on cached softmax probabilities from forward().
    ##
    def __init__(self, softmax_output: np.ndarray) -> None:
        self.softmax_output = softmax_output

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        s = self.softmax_output
        sum_grad_per_sample = np.sum(out_grad, axis=-1, keepdims=True)
        return out_grad - sum_grad_per_sample * s