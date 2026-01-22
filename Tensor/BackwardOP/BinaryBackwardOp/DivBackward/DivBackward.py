##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Division backward operator
##

# **************************************************************************** #
#                                                                              #
#                           DIV BINARY BACKWARD                                #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BinaryBackwardOp.BinaryBackwardOp import BinaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# For y = A / B, ∂y/∂A = 1/B while ∂y/∂B = -A/B², so gradients divide by the
# denominator for A and subtract the scaled quotient for B before reducing.

class DivBackward(BinaryBackwardOp):
    ##
    # @brief Backward operator for a / b.
    # @details Applies derivatives 1/b for numerator and -a/b² for denominator.
    # Implements element-wise quotient rule with broadcasting.
    # Used by Tensor.__truediv__ and reverse division.
    ##
    def _compute_grad_a(self, out_grad, a, b):
        safe_b_data = np.maximum(b.data, 1e-8)
        return self._reduce_grad(out_grad / safe_b_data, a.data.shape)

    def _compute_grad_b(self, out_grad, a, b):
        safe_b_data = np.maximum(b.data, 1e-8)
        grad = -out_grad * a.data / (safe_b_data ** 2)
        return self._reduce_grad(grad, b.data.shape)
