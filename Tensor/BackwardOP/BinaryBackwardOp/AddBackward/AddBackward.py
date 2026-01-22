##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Add backward operator
##

# **************************************************************************** #
#                                                                              #
#                            ADD BINARY BACKWARD                               #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BinaryBackwardOp.BinaryBackwardOp import BinaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dA (A + B) = 1 and d/dB (A + B) = 1, so upstream gradients are forwarded
# directly to each operand before any broadcast-aware reduction.

class AddBackward(BinaryBackwardOp):
    ##
    # @brief Backward operator for a + b.
    # @details Forwards the upstream gradient unchanged to both operands.
    # Implements derivative 1 for both inputs with broadcasting support.
    # Used by Tensor.__add__ and Tensor.__radd__.
    ##
    def _compute_grad_a(self, out_grad, a, b):
        return self._reduce_grad(out_grad, a.data.shape)

    def _compute_grad_b(self, out_grad, a, b):
        return self._reduce_grad(out_grad, b.data.shape)
