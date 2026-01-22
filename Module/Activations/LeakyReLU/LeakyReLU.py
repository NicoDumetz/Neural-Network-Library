##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## LeakyReLU activation module
##

from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                             ACTIVATION CORE API                              #
#                                                                              #
# **************************************************************************** #

class LeakyReLU(Module):
    ##
    # @brief Initialize a LeakyReLU activation module.
    # @details Stores the negative slope used to scale negative inputs during
    # the activation. A small slope prevents neurons from becoming inactive
    # compared to the standard ReLU.
    ##
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    ##
    # @brief Apply the LeakyReLU activation to the input tensor.
    # @details Computes ReLU(x) for positive values and applies a scaled ReLU(-x)
    # for negative values. This avoids zero gradients on negative input regions
    # while keeping the behavior similar to standard ReLU.
    ##
    def forward(self, x: Tensor) -> Tensor:
        pos = x.relu()
        neg = (-x).relu()
        return pos - neg * self.negative_slope

    ##
    # @brief Return a readable string representation of the module.
    # @details Displays the configured negative slope to help identify the
    # activation module during debugging or model inspection.
    ##
    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"
