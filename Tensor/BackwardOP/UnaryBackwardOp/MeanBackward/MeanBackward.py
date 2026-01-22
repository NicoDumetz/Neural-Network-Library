##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Mean backward operator
##

# **************************************************************************** #
#                                                                              #
#                          MEAN UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# Each element that participated in the mean receives grad / count, so upstream
# gradients are divided by the number of reduced elements and broadcast back.

class MeanBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for Tensor.mean.
    # @details Divides gradients by number of averaged elements.
    # Restores reduced dimensions if keepdims=False.
    # Broadcasts final gradient to original input shape.
    ##
    def __init__(self, input_shape, axis, keepdims):
        self.input_shape = input_shape
        self.axis = axis
        self.keepdims = keepdims
        if axis is None:
            self.factor = float(np.prod(input_shape))
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            size = 1
            for ax in axes:
                size *= input_shape[ax]
            self.factor = float(size)

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        grad = out_grad / self.factor
        if self.axis is not None and not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in axes:
                grad = np.expand_dims(grad, axis=ax)
        return np.broadcast_to(grad, self.input_shape)
