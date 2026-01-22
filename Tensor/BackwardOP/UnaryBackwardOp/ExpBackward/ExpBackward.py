##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Exponential backward operator
##

# **************************************************************************** #
#                                                                              #
#                           EXP UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dx exp(x) = exp(x), so the stored input is exponentiated again and used to
# scale the upstream gradient element-wise.


class ExpBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for exp(x).
    # @details Multiplies upstream gradients by exp(x) using input values.
    # Implements derivative d/dx exp(x) = exp(x) element-wise.
    # Used in Tensor.exp and internally by softmax.
    ##
    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * np.exp(a.data)
