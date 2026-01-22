##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Logarithm backward operator
##

# **************************************************************************** #
#                                                                              #
#                           LOG UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dx log(x) = 1/x, so gradients divide element-wise by the original input
# (clamped for stability) before flowing backward.


class LogBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for log(x).
    # @details Computes gradient by multiplying by 1/x element-wise.
    # Implements derivative d/dx log(x) = 1/x.
    # Used in Tensor.log and log-softmax internals.
    ##
    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
            safe_a_data = np.maximum(a.data, 1e-8)
            return out_grad / safe_a_data
