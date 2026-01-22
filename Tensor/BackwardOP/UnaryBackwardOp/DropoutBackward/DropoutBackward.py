##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Dropout
##
# **************************************************************************** #
#                                                                              #
#                         DROPOUT UNARY BACKWARD OP                            #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

# Derivative summary:
# Dropout rescales surviving activations by scale, so the gradient is simply
# masked by the dropout pattern and multiplied by the same scale factor.


class DropoutBackward(UnaryBackwardOp):
    ##
    # @brief Backward operator for dropout masking.
    # @details Applies the sampled binary mask and scaling factor to the upstream
    # gradient to ensure expectation-preserving updates.
    ##
    def __init__(self, mask: np.ndarray, scale: float) -> None:
        self.mask = mask
        self.scale = scale

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return out_grad * self.mask * self.scale
