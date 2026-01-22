##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Absolute value backward operator
##

# **************************************************************************** #
#                                                                              #
#                          ABS UNARY BACKWARD OP                               #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dx |x| = sign(x) for x != 0 and 0 for x == 0; implemented via np.sign().
# Upstream gradients are masked by the sign pattern element-wise.


class AbsBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for Tensor.abs.
    # @details Multiplies upstream gradients by sign(x), zeroing undefined points
    # at the origin via NumPy’s sign implementation.
    ##
    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * np.sign(a.data)
