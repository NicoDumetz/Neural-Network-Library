##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Multiply backward operator
##

# **************************************************************************** #
#                                                                              #
#                           MUL BINARY BACKWARD                                #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BinaryBackwardOp.BinaryBackwardOp import BinaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dA (A * B) = B and d/dB (A * B) = A, meaning gradients are scaled by the
# opposite operand before being reshaped for broadcasting.

class MulBackward(BinaryBackwardOp):
    ##
    # @brief Backward operator for a * b.
    # @details Applies product rule: gradient to a is grad * b, to b is grad * a.
    # Supports broadcasting for scalar and tensor scaling.
    # Used by Tensor.__mul__ and elementwise operations.
    ##
    def _compute_grad_a(self, out_grad, a, b):
        return self._reduce_grad(out_grad * b.data, a.data.shape)

    def _compute_grad_b(self, out_grad, a, b):
        return self._reduce_grad(out_grad * a.data, b.data.shape)
