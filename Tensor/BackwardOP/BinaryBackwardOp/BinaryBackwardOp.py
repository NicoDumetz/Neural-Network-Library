##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Binary backward operator base class
##

# **************************************************************************** #
#                                                                              #
#                         ABSTRACT BINARY BACKWARD OP                          #
#                                                                              #
# **************************************************************************** #
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BackwardOp import BackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

class BinaryBackwardOp(BackwardOp):
    def backward(self, out_grad, parents):
        a, b = parents
        if a.requires_grad:
            self._accumulate(a, self._compute_grad_a(out_grad, a, b))
        if b.requires_grad:
            self._accumulate(b, self._compute_grad_b(out_grad, a, b))

    def _reduce_grad(self, grad: np.ndarray, target_shape: tuple) -> np.ndarray:
        num_missing_dims = len(grad.shape) - len(target_shape)
        if num_missing_dims > 0:
            sum_axes = tuple(range(num_missing_dims))
            grad = np.sum(grad, axis=sum_axes, keepdims=False)
        for i in range(len(target_shape)):
            if target_shape[i] == 1 and grad.shape[i] > 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad.reshape(target_shape)

    def _compute_grad_a(self, out_grad, a, b):
        raise NotImplementedError

    def _compute_grad_b(self, out_grad, a, b):
        raise NotImplementedError