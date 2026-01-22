##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Softmax backward operator
##

# **************************************************************************** #
#                                                                              #
#                        SOFTMAX UNARY BACKWARD OP                             #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# Jacobian of softmax is diag(s) - s sᵀ, so the gradient is s * (grad - ⟨grad,s⟩)
# reusing cached probabilities per sample.


class SoftmaxBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for softmax.
    # @details Applies Jacobian-vector rule: s * (grad − ⟨grad,s⟩).
    # Uses cached softmax probabilities from forward().
    # Used in Tensor.softmax and cross-entropy models.
    ##
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        s = self.output
        return s * (out_grad - np.sum(out_grad * s))
