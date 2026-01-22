##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Sum backward operator
##

# **************************************************************************** #
#                                                                              #
#                           SUM UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# Sum reductions distribute the upstream gradient equally to every element that
# contributed, so the gradient is simply broadcast back to the input shape.

class SumBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for Tensor.sum.
    # @details Broadcasts upstream gradients to input shape.
    # Restores removed axes when keepdims=False.
    # Supports any combination of reduction axes.
    ##
    def __init__(self, input_shape, axis, keepdims):
        self.input_shape = input_shape
        self.axis = axis
        self.keepdims = keepdims

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        grad = out_grad
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in axes:
                grad = np.expand_dims(grad, axis=ax)
        return np.broadcast_to(grad, self.input_shape)
