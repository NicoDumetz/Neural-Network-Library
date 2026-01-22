##
## EPITECH PROJECT, 2025
## library
## File description:
## Flatten
##
from __future__ import annotations
import numpy as np
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor


class Flatten(Module):
    ##
    # @brief Initialize a flatten layer.
    # @details Stores the dimension from which the tensor will be flattened,
    # mimicking torch.nn.Flatten semantics with N-dimensional support.
    ##
    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    ##
    # @brief Collapse tensor dimensions starting from start_dim.
    # @details Reshapes the tensor while preserving leading axes, keeping the
    # operation differentiable through Tensor.reshape.
    ##
    def forward(self, x: Tensor) -> Tensor:
        dims = x.data.ndim
        start = self.start_dim if self.start_dim >= 0 else dims + self.start_dim
        if start < 0 or start >= dims:
            raise ValueError(f"start_dim must be within [0, {dims}), got {self.start_dim}.")
        leading = x.data.shape[:start]
        trailing = x.data.shape[start:]
        flat_dim = int(np.prod(trailing))
        new_shape = leading + (flat_dim,)
        return x.reshape(new_shape)
