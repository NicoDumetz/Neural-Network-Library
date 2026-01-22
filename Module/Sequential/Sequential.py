##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Sequential
##
from __future__ import annotations
from typing import List
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                               SEQUENTIAL CORE API                            #
#                                                                              #
# **************************************************************************** #


class Sequential(Module):
    ##
    # @brief Initialize a sequential container of modules.
    # @details Stores an ordered list of child modules and registers each of them
    # as an attribute for proper parameter tracking. Allows stacking arbitrary
    # layers to build feed-forward architectures.
    ##
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: List[Module] = []
        for i, layer in enumerate(layers):
            self.layers.append(layer)
            setattr(self, f"layer_{i}", layer)

    ##
    # @brief Execute the forward pass through all contained modules in order.
    # @details Iteratively feeds the output of each module into the next, enabling
    # the construction of deep models by function composition. The final output is
    # the result of the last module.
    ##
    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    ##
    # @brief Return a formatted string representation of the sequential container.
    # @details Displays each module on its own line to help with model inspection
    # and debugging of complex layer stacks.
    ##
    def __repr__(self) -> str:
        rep = "Sequential(\n"
        for layer in self.layers:
            rep += f"  {layer}\n"
        rep += ")"
        return rep
