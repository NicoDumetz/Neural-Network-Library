##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Power backward operator
##

# **************************************************************************** #
#                                                                              #
#                           POW UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# For y = x^p, ∂y/∂x = p * x^(p-1); gradients multiply this analytic derivative
# element-wise with the upstream signal.

class PowBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for Tensor.pow.
    # @details Uses derivative p * x^(p-1) applied element-wise.
    # Supports any scalar exponent passed during forward().
    # Multiplies this derivative with upstream gradient.
    ##
    def __init__(self, power: float) -> None:
        self.power = power

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * self.power * np.power(a.data, self.power - 1)
