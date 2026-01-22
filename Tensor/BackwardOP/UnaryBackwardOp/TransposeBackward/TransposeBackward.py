##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Transpose backward operator
##

# **************************************************************************** #
#                                                                              #
#                       TRANSPOSE UNARY BACKWARD OP                            #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# Transpose is linear and orthonormal, so gradients are transposed using the
# same axis permutation applied during the forward pass.


class TransposeBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for transpose.
    # @details Applies the same axis permutation to the gradient array.
    # Mirrors Tensor.transpose behavior fully.
    # Works for both explicit and reversed permutations.
    ##
    def __init__(self, axes: tuple[int, ...] | None) -> None:
        self.axes = axes

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad.transpose(self.axes)
