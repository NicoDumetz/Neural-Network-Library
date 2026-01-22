##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## BackwardOp base definition
##

# **************************************************************************** #
#                                                                              #
#                          ABSTRACT BACKWARD OPERATOR                          #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor


class BackwardOp:
    def backward(self, out_grad: np.ndarray, parents: Tuple["Tensor", ...]) -> None:
        ##
        # @brief Apply the chain rule for this operation.
        # @details Receives the gradient w.r.t. the output and must compute each
        # parent gradient. Implemented in subclasses for specific operations.
        # @param out_grad Upstream gradient.
        # @param parents Parent tensors of the forward step.
        ##
        raise NotImplementedError

    ##
    # @brief Accumulate gradient into a tensor.
    # @details Adds grad to tensor.grad unless requires_grad is false. Ensures
    # correct gradient accumulation across graph branches. Initializes grad if
    # needed.
    ##
    def _accumulate(self, tensor: "Tensor", grad: np.ndarray) -> None:
        if not tensor.requires_grad:
            return
        if tensor.grad is None:
            tensor.grad = grad.copy() if isinstance(grad, np.ndarray) else grad 
        else:
            tensor.grad = tensor.grad + grad