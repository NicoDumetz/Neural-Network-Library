##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Sigmoid backward operator
##

# **************************************************************************** #
#                                                                              #
#                        SIGMOID UNARY BACKWARD OP                             #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# σ'(x) = σ(x)(1-σ(x)), so the stored sigmoid output is reused to compute the
# multiplicative factor applied to the upstream gradient.


class SigmoidBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for sigmoid.
    # @details Uses cached σ(x) to apply derivative σ(x)(1−σ(x)).
    # Multiplies this derivative with upstream gradient.
    # Used by Tensor.sigmoid and logistic modules.
    ##
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * self.output * (1 - self.output)
