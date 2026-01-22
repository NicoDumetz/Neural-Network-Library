##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Hyperbolic tangent backward operator
##

# **************************************************************************** #
#                                                                              #
#                          TANH UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dx tanh(x) = 1 - tanh(x)^2, so the cached activation supplies the factor
# multiplying the upstream gradient.

class TanhBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for tanh.
    # @details Uses cached tanh(x) to apply 1 − tanh(x)².
    # Multiplies this derivative with upstream gradient.
    # Enables Tensor.tanh and related layers.
    ##
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * (1 - self.output**2)
