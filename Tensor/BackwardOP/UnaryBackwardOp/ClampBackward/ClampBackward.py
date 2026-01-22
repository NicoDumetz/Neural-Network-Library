##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Clamp backward operator
##

# **************************************************************************** #
#                                                                              #
#                          CLAMP UNARY BACKWARD OP                             #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# d/dx clamp(x) equals 1 when min < x < max and 0 otherwise, so the upstream
# gradient is masked wherever the forward output saturated at the bounds.


class ClampBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for Tensor.clamp.
    # @details Passes gradients only where inputs were strictly inside bounds.
    # Entries clamped to min or max receive a zero gradient.
    # Gradient mask depends on the clamp range used in forward().
    ##
    def __init__(self, min_val: float | None, max_val: float | None) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        mask = np.ones_like(a.data, dtype=bool)
        if self.min_val is not None:
            mask &= a.data > self.min_val
        if self.max_val is not None:
            mask &= a.data < self.max_val
        return out_grad * mask
