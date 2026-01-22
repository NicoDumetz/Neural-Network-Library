##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Sigmoid activation module
##

from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                             ACTIVATION CORE API                              #
#                                                                              #
# **************************************************************************** #


class Sigmoid(Module):
    ##
    # @brief Initialize a Sigmoid activation module.
    # @details Prepares the module for use as a parameter-free activation layer.
    # The sigmoid function maps inputs into the interval [0,1], making it suitable
    # for binary classification and gating mechanisms.
    ##
    def __init__(self) -> None:
        super().__init__()

    ##
    # @brief Apply the sigmoid activation to the input tensor.
    # @details Computes the element-wise sigmoid function 1 / (1 + exp(-x)),
    # producing smooth, bounded outputs and well-defined gradients. Commonly
    # used in output layers and recurrent neural networks.
    ##
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


    ##
    # @brief Return a string representation of the Sigmoid module.
    # @details Provides a clean identifier useful for debugging and model
    # visualization.
    ##
    def __repr__(self) -> str:
        return "Sigmoid()"
