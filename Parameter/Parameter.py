##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Parameters
##
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from library.Tensor.Tensor import Tensor


# **************************************************************************** #
#                                                                              #
#                               PARAMETER CORE API                             #
#                                                                              #
# **************************************************************************** #

class Parameter(Tensor):
    ##
    # @brief Initialize a trainable parameter tensor.
    # @details Wraps a Tensor with gradient tracking enabled by default and
    # allocates a pruning mask of the same shape. Supports autograd, pruning, and
    # optimizer interaction as a fundamental building block of learnable modules.
    ##
    def __init__(self, data: ArrayLike, dtype: DTypeLike = np.float32) -> None:
        super().__init__(data, requires_grad=True, dtype=dtype)
        self.mask: np.ndarray = np.ones_like(self.data, dtype=np.bool_)

    ##
    # @brief Return a readable description of the parameter.
    # @details Displays the shape of the underlying data and the number of pruned
    # elements to assist with debugging, logging, and model inspection.
    ##
    def __repr__(self) -> str:
        return f"Parameter(shape={self.data.shape}, pruned={np.sum(self.mask == 0)}/{self.data.size})"