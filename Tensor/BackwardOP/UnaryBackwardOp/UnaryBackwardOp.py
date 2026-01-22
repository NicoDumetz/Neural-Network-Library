##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Unary backward operator base class
##

# **************************************************************************** #
#                                                                              #
#                          ABSTRACT UNARY BACKWARD OP                          #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BackwardOp import BackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor



class UnaryBackwardOp(BackwardOp):
    ##
    # @brief Base class for unary backward operators.
    # @details Forwards gradients through _compute_grad implementation.
    # Children override derivative logic for specific unary ops.
    # Handles gradient routing to a single parent tensor.
    ##
    def backward(self, out_grad: np.ndarray, parents: Tuple["Tensor"]) -> None:
        (a,) = parents
        if not a.requires_grad:
            return
        grad_to_parent = self._compute_grad(out_grad, a)
        self._accumulate(a, grad_to_parent)
        if a._op is not None and a.grad is not None:
             a._op.backward(a.grad, a._parents)

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        raise NotImplementedError
