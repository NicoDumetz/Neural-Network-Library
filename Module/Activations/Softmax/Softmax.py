##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Softmax activation module
##

from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                             ACTIVATION CORE API                              #
#                                                                              #
# **************************************************************************** #


class Softmax(Module):
    ##
    # @brief Initialize a Softmax activation module.
    # @details Sets up the module without parameters since softmax is a
    # parameter-free transformation. Prepares the module to convert logits
    # into normalized probability distributions.
    ##
    def __init__(self) -> None:
        super().__init__()

    ##
    # @brief Apply the softmax transformation to the input tensor.
    # @details Computes the exponentiated and normalized values along the last
    # axis, ensuring numerical stability. Produces outputs that sum to one,
    # commonly used for multi-class classification.
    ##
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax()

    ##
    # @brief Return a string representation of the Softmax module.
    # @details Provides a simple readable identifier for debugging or model
    # inspection purposes.
    ##
    def __repr__(self) -> str:
        return "Softmax()"
