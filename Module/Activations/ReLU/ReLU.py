##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Relu
##
from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                             ACTIVATION CORE API                              #
#                                                                              #
# **************************************************************************** #

class ReLU(Module):
    ##
    # @brief Initialize a ReLU activation module.
    # @details Sets up the module without additional parameters since ReLU is a
    # parameter-free activation function. Prepares the module for use inside a
    # computation graph.
    ##
    def __init__(self) -> None:
        super().__init__()

    ##
    # @brief Apply the ReLU activation to the input tensor.
    # @details Computes the element-wise rectified linear activation, returning
    # each value if positive and zero otherwise. Frequently used due to its
    # simplicity and efficient gradient propagation.
    ##
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    ##
    # @brief Return a string representation of the ReLU module.
    # @details Provides a concise identifier for debugging or model inspection.
    ##
    def __repr__(self) -> str:
        return "ReLU()"
