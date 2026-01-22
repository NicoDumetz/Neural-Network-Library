##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Negation backward operator
##

# **************************************************************************** #
#                                                                              #
#                           NEG UNARY BACKWARD OP                              #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dx (-x) = -1, so the backward pass simply flips the sign of the incoming
# gradient before passing it along.


class NegBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for negation.
    # @details Multiplies upstream gradient by -1.
    # Used by Tensor.__neg__ and subtraction expressions.
    # Simple linear derivative, no extra state required.
    ##
    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return -out_grad
