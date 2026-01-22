##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Reshape backward operator
##

# **************************************************************************** #
#                                                                              #
#                        RESHAPE UNARY BACKWARD OP                             #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# Reshape is a view, so gradients are only reordered back to the original shape
# with no scaling or masking applied.

class ReshapeBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for Tensor.reshape.
    # @details Restores gradients to original shape used before reshape().
    # Does not modify values, only layout.
    # Works because reshape is a view-like transformation.
    ##
    def __init__(self, original_shape: tuple[int, ...]) -> None:
        self.original_shape = original_shape

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad.reshape(self.original_shape)
