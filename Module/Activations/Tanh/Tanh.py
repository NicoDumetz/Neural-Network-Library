##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Tanh activation module
##

from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                             ACTIVATION CORE API                              #
#                                                                              #
# **************************************************************************** #


class Tanh(Module):
    ##
    # @brief Initialize a Tanh activation module.
    # @details Sets up the module as a parameter-free activation layer. The tanh
    # function maps inputs into the interval [-1,1] and provides stronger
    # non-linearity than sigmoid while remaining smooth and differentiable.
    ##
    def __init__(self) -> None:
        super().__init__()

    ##
    # @brief Apply the hyperbolic tangent activation to the input tensor.
    # @details Computes the element-wise tanh transformation, producing outputs
    # centered around zero. This can improve optimization stability in various
    # neural architectures.
    ##
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    ##
    # @brief Return a string representation of the Tanh module.
    # @details Provides a concise identifier useful for debugging and inspecting
    # the structure of a model.
    ##
    def __repr__(self) -> str:
        return "Tanh()"
