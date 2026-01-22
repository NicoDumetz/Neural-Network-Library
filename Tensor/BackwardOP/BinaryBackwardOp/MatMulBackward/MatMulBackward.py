##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Matrix multiplication backward operator
##

# **************************************************************************** #
#                                                                              #
#                          MATMUL BINARY BACKWARD                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BinaryBackwardOp.BinaryBackwardOp import BinaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# ∂(A @ B)/∂A = upstream @ Bᵀ and ∂/∂B = Aᵀ @ upstream, following matrix
# calculus rules; gradients are optionally reduced for broadcasting.


class MatMulBackward(BinaryBackwardOp):
    ##
    # @brief Backward operator for matrix multiplication.
    # @details Uses grad@Bᵀ for dA and Aᵀ@grad for dB following matrix calculus.
    # Supports batched and 2D matrix multiplications.
    # Drives Tensor.__matmul__ and linear layers.
    ##
    def _compute_grad_a(self, out_grad, a, b):
        b_T_batched = np.swapaxes(b.data, -2, -1)
        grad = out_grad.dot(b_T_batched)
        return self._reduce_grad(grad, a.data.shape)

    def _compute_grad_b(self, out_grad, a, b):
        a_T = a.data.T
        grad = a_T.dot(out_grad)
        return self._reduce_grad(grad, b.data.shape)