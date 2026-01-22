##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## ReLU backward operator
##

# **************************************************************************** #
#                                                                              #
#                          RELU UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# ReLU derivative is 1 when x>0 and 0 otherwise, so gradients are masked by
# the positive-activation indicator generated from the saved inputs.

class ReluBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for ReLU.
    # @details Zeroes gradients where input ≤ 0 and passes others unchanged.
    # Implements derivative of max(0,x) using a boolean mask.
    # Used by Tensor.relu and activation modules.
    ##
    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * (a.data > 0)
